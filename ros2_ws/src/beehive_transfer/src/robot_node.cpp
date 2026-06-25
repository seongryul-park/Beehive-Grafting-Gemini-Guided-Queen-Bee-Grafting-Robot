// =============================================================================
//  robot_node.cpp  —  motion node (Stages 5-7).
//
//  Consumes ONLY /selected_pose (geometry_msgs/PoseStamped, already in the robot
//  base frame, produced by the vision stage). It performs NO pixel->world
//  conversion of its own — all of that lives upstream in vision_node /
//  coord_transform. This node is pure motion + place logic.
//
//  Per received target pose (comb is a VERTICAL WALL; grasp is HORIZONTAL):
//      Home -> Approach (stand off in front of the wall along -X) -> Pick (move
//           in to the comb face, attach a small virtual larva to the gripper)
//           -> Retreat (back off the wall) -> move to the next empty tray slot
//           -> Place (detach, leaving the larva in the slot) -> Return Home.
//
//  The single fixed drop location was replaced by a tray of predefined slots
//  (default 5x5 = 25). Each successfully picked larva is placed into the next
//  empty slot, filled sequentially, then the arm returns home.
//
//  Pick is virtual (no real gripper): the larva is a small collision box that we
//  attachObject() to the gripper after Pick and detachObject() at the slot, so
//  it rides the end-effector in RViz and then stays in the tray.
//
//  Notes (Jazzy): MoveIt headers are .hpp. MoveGroupInterface needs a spinning
//  node, so the executor runs on its own thread and setup() runs after spin-up.
// =============================================================================

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

static const std::string PLANNING_GROUP = "panda_arm";

class RobotNode : public rclcpp::Node
{
public:
  RobotNode() : Node("robot_node")
  {
    // ---- parameters ------------------------------------------------------
    target_topic_   = declare_parameter<std::string>("target_topic", "/selected_pose");
    hand_link_      = declare_parameter<std::string>("hand_link", "panda_hand");
    home_target_    = declare_parameter<std::string>("home_named_target", "ready");
    approach_height_= declare_parameter<double>("approach_height", 0.10);
    lift_height_    = declare_parameter<double>("lift_height", 0.15);
    larva_size_     = declare_parameter<double>("larva_size", 0.02);

    // wall-facing grasp: tool Z points +X (into the comb wall). rpy default
    // (0, pi/2, 0). The arm stands off along approach_dir (-X) before moving in,
    // then retreats back along it after the virtual pick.
    grasp_rpy_      = declare_parameter<std::vector<double>>(
                        "grasp_rpy", std::vector<double>{0.0, M_PI / 2.0, 0.0});
    approach_dir_   = declare_parameter<std::vector<double>>(
                        "approach_dir", std::vector<double>{-1.0, 0.0, 0.0});
    approach_standoff_ = declare_parameter<double>("approach_standoff", 0.10);
    retreat_standoff_  = declare_parameter<double>("retreat_standoff", 0.12);

    // tray of placement slots (a grid). Defaults: 5x5 beside the comb.
    tray_x_  = declare_parameter<double>("tray_x", 0.45);
    tray_y_  = declare_parameter<double>("tray_y", -0.30);
    tray_z_  = declare_parameter<double>("tray_z", 0.05);
    tray_rows_ = declare_parameter<int>("tray_rows", 5);
    tray_cols_ = declare_parameter<int>("tray_cols", 5);
    tray_dx_ = declare_parameter<double>("tray_dx", 0.035);
    tray_dy_ = declare_parameter<double>("tray_dy", 0.035);

    subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        target_topic_, 10,
        std::bind(&RobotNode::onTarget, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "robot_node ready — waiting on %s",
                target_topic_.c_str());
  }

  // MoveGroupInterface can't be built in the constructor (needs shared_from_this).
  void setup()
  {
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), PLANNING_GROUP);
    move_group_->setMaxVelocityScalingFactor(0.3);
    move_group_->setMaxAccelerationScalingFactor(0.3);

    buildSlots();
    addTray();

    RCLCPP_INFO(get_logger(),
                "MoveGroup ready. planning_frame=%s, %zu tray slots.",
                move_group_->getPlanningFrame().c_str(), slots_.size());

    goHome();   // always START from a predefined home pose
  }

private:
  // ------------------------------------------------------------------ target
  void onTarget(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (busy_) {
      RCLCPP_WARN(get_logger(), "busy with a previous target — ignoring this one");
      return;
    }
    if (slot_index_ >= static_cast<int>(slots_.size())) {
      RCLCPP_WARN(get_logger(), "tray is FULL (%zu slots) — ignoring target",
                  slots_.size());
      return;
    }
    busy_ = true;

    // The pose is already in the base frame; we do NOT convert pixels here.
    if (!msg->header.frame_id.empty() &&
        msg->header.frame_id != move_group_->getPlanningFrame()) {
      RCLCPP_WARN(get_logger(),
                  "target frame '%s' != planning frame '%s' — assuming aligned",
                  msg->header.frame_id.c_str(),
                  move_group_->getPlanningFrame().c_str());
    }

    const geometry_msgs::msg::Pose target = msg->pose;
    RCLCPP_INFO(get_logger(), "target #%d at (%.3f, %.3f, %.3f)",
                slot_index_, target.position.x, target.position.y,
                target.position.z);

    const std::string larva_id = "larva_" + std::to_string(slot_index_);

    // 1) spawn the larva to be picked
    addLarvaBox(larva_id, target.position);

    const geometry_msgs::msg::Quaternion grasp_q = graspOrientation();

    // 2) pre-grasp standoff IN FRONT of the wall (along approach_dir, default -X)
    geometry_msgs::msg::Pose approach = target;
    approach.orientation = grasp_q;
    approach.position.x += approach_dir_[0] * approach_standoff_;
    approach.position.y += approach_dir_[1] * approach_standoff_;
    approach.position.z += approach_dir_[2] * approach_standoff_;
    if (!moveTo(approach, "approach")) return release();

    // 3) move in to the comb face to pick (horizontal, facing the wall)
    geometry_msgs::msg::Pose pick = target;
    pick.orientation = grasp_q;
    if (!moveTo(pick, "pick")) return release();

    // 4) virtual grip: attach the larva to the gripper
    RCLCPP_INFO(get_logger(), ">>> [pick] gripper close (virtual) <<<");
    attachLarva(larva_id);

    // 5) retreat straight back from the wall (reverse of the approach)
    geometry_msgs::msg::Pose lift = pick;
    lift.position.x += approach_dir_[0] * retreat_standoff_;
    lift.position.y += approach_dir_[1] * retreat_standoff_;
    lift.position.z += approach_dir_[2] * retreat_standoff_;
    if (!moveTo(lift, "retreat from wall")) return release();

    // 6) move above the next empty tray slot
    geometry_msgs::msg::Pose slot = slots_[slot_index_];
    geometry_msgs::msg::Pose above_slot = slot;
    above_slot.position.z += approach_height_;
    if (!moveTo(above_slot, "move to slot")) return release();

    // 7) descend and place
    if (!moveTo(slot, "place")) return release();
    RCLCPP_INFO(get_logger(), ">>> [place] gripper open (virtual) <<<");
    detachLarva(larva_id);   // larva stays in the slot

    // 8) retreat and return home
    if (!moveTo(above_slot, "retreat")) return release();
    goHome();

    RCLCPP_INFO(get_logger(), "=== placed larva in slot #%d (%d/%zu used) ===",
                slot_index_, slot_index_ + 1, slots_.size());
    slot_index_++;
    busy_ = false;
  }

  void release() { busy_ = false; }   // bail out of a cycle, stay responsive

  // ------------------------------------------------------------------- motion
  bool moveTo(const geometry_msgs::msg::Pose &target, const std::string &label)
  {
    RCLCPP_INFO(get_logger(), "[%s] planning...", label.c_str());
    move_group_->setPoseTarget(target);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (!static_cast<bool>(move_group_->plan(plan))) {
      RCLCPP_ERROR(get_logger(), "[%s] planning FAILED", label.c_str());
      return false;
    }
    move_group_->execute(plan);
    RCLCPP_INFO(get_logger(), "[%s] done", label.c_str());
    return true;
  }

  void goHome()
  {
    RCLCPP_INFO(get_logger(), "[home] returning to '%s'", home_target_.c_str());
    move_group_->setNamedTarget(home_target_);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (static_cast<bool>(move_group_->plan(plan))) {
      move_group_->execute(plan);
      RCLCPP_INFO(get_logger(), "[home] done");
    } else {
      RCLCPP_WARN(get_logger(),
                  "[home] could not plan to named target '%s' "
                  "(is it in the SRDF?)", home_target_.c_str());
    }
  }

  // ------------------------------------------------------------- scene helpers
  geometry_msgs::msg::Quaternion downOrientation()
  {
    // top-down grasp: 180deg about X -> tool Z points down (-Z base).
    // Used for placing flat into the horizontal tray.
    geometry_msgs::msg::Quaternion q;
    q.x = 1.0; q.y = 0.0; q.z = 0.0; q.w = 0.0;
    return q;
  }

  static geometry_msgs::msg::Quaternion rpyToQuat(double r, double p, double y)
  {
    const double cr = std::cos(r / 2), sr = std::sin(r / 2);
    const double cp = std::cos(p / 2), sp = std::sin(p / 2);
    const double cy = std::cos(y / 2), sy = std::sin(y / 2);
    geometry_msgs::msg::Quaternion q;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;
    q.w = cr * cp * cy + sr * sp * sy;
    return q;
  }

  // wall-facing grasp orientation (tool Z -> +X by default) from grasp_rpy_
  geometry_msgs::msg::Quaternion graspOrientation()
  {
    return rpyToQuat(grasp_rpy_[0], grasp_rpy_[1], grasp_rpy_[2]);
  }

  void buildSlots()
  {
    slots_.clear();
    for (int r = 0; r < tray_rows_; ++r) {
      for (int c = 0; c < tray_cols_; ++c) {
        geometry_msgs::msg::Pose p;
        p.position.x = tray_x_ + (c - (tray_cols_ - 1) / 2.0) * tray_dx_;
        p.position.y = tray_y_ + (r - (tray_rows_ - 1) / 2.0) * tray_dy_;
        p.position.z = tray_z_;
        p.orientation = downOrientation();
        slots_.push_back(p);
      }
    }
  }

  void addTray()
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = move_group_->getPlanningFrame();
    obj.id = "graft_tray";

    shape_msgs::msg::SolidPrimitive box;
    box.type = box.BOX;
    box.dimensions = {tray_cols_ * tray_dx_ + 0.03,
                      tray_rows_ * tray_dy_ + 0.03,
                      0.01};
    geometry_msgs::msg::Pose pose;
    pose.orientation.w = 1.0;
    pose.position.x = tray_x_;
    pose.position.y = tray_y_;
    pose.position.z = tray_z_ - 0.011;   // tray surface just below the slots

    obj.primitives.push_back(box);
    obj.primitive_poses.push_back(pose);
    obj.operation = obj.ADD;
    scene_.applyCollisionObjects({obj});
    RCLCPP_INFO(get_logger(), "added graft tray (%dx%d slots)",
                tray_rows_, tray_cols_);
  }

  void addLarvaBox(const std::string &id, const geometry_msgs::msg::Point &p)
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = move_group_->getPlanningFrame();
    obj.id = id;

    shape_msgs::msg::SolidPrimitive box;
    box.type = box.BOX;
    box.dimensions = {larva_size_, larva_size_, larva_size_};

    geometry_msgs::msg::Pose pose;
    pose.orientation.w = 1.0;
    pose.position = p;

    obj.primitives.push_back(box);
    obj.primitive_poses.push_back(pose);
    obj.operation = obj.ADD;
    scene_.applyCollisionObjects({obj});
  }

  void attachLarva(const std::string &id)
  {
    // fingers/hand are allowed to touch the attached larva
    std::vector<std::string> touch = {hand_link_, "panda_leftfinger",
                                      "panda_rightfinger"};
    move_group_->attachObject(id, hand_link_, touch);
  }

  void detachLarva(const std::string &id)
  {
    move_group_->detachObject(id);   // remains in the world at the slot
  }

  // ---- members ----
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr subscription_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  moveit::planning_interface::PlanningSceneInterface scene_;

  std::vector<geometry_msgs::msg::Pose> slots_;
  int slot_index_ = 0;
  bool busy_ = false;

  std::string target_topic_, hand_link_, home_target_;
  double approach_height_, lift_height_, larva_size_;
  double approach_standoff_, retreat_standoff_;
  std::vector<double> grasp_rpy_, approach_dir_;
  double tray_x_, tray_y_, tray_z_, tray_dx_, tray_dy_;
  int tray_rows_, tray_cols_;
};

// =============================================================================
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RobotNode>();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() { executor.spin(); });

  node->setup();        // build MoveGroup + tray, then go home

  spin_thread.join();
  rclcpp::shutdown();
  return 0;
}
