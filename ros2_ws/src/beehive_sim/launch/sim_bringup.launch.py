# =============================================================================
#  sim_bringup.launch.py  —  Stage 1/2/3 bringup.
#
#  Brings up:
#    - Gazebo Harmonic with worlds/beehive.sdf (table + textured honeycomb
#      plane + fixed downward camera)
#    - ros_gz_bridge: Gazebo camera image/camera_info + /clock -> ROS
#    - vision_node: live camera frame -> EXISTING beehive pipeline (on demand)
#    - (optional) Panda spawned for visualization (no motion yet)
#    - (optional) rqt_image_view on the annotated topic
#
#  Run (camera -> vision, the thing to verify first):
#    export BEEHIVE_WEB_DIR=/abs/path/to/beehive_project/web
#    ros2 launch beehive_sim sim_bringup.launch.py
#  then, in another terminal:
#    ros2 service call /vision_node/analyze std_srvs/srv/Trigger
#    ros2 run rqt_image_view rqt_image_view /vision_node/annotated
#
#  Notes:
#    - motion planning is intentionally NOT started here (Stage 5+).
#    - spawn_robot defaults to true but is skipped gracefully if the Panda
#      description package isn't installed, so camera->vision still comes up.
# =============================================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            SetEnvironmentVariable, GroupAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _maybe_robot_nodes():
    """Build robot_state_publisher + spawn nodes IF the Panda description exists.

    Returns a list of actions (possibly empty). Done in Python (not a launch
    condition) so a missing description package never breaks the whole bringup.
    """
    try:
        import xacro
        cfg = get_package_share_directory("moveit_resources_panda_moveit_config")
    except Exception:
        print("[beehive_sim] Panda description not found — skipping robot spawn. "
              "Install moveit_resources_panda_moveit_config to visualize the arm.")
        return []

    xacro_path = os.path.join(cfg, "config", "panda.urdf.xacro")
    if not os.path.isfile(xacro_path):
        print(f"[beehive_sim] {xacro_path} missing — skipping robot spawn.")
        return []

    try:
        robot_desc = xacro.process_file(xacro_path).toxml()
    except Exception as e:  # noqa: BLE001
        print(f"[beehive_sim] could not process Panda xacro ({e}) — skipping.")
        return []

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_desc, "use_sim_time": True}],
    )
    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description", "-name", "panda",
                   "-x", "0", "-y", "0", "-z", "0"],
        condition=IfCondition(LaunchConfiguration("spawn_robot")),
    )
    return [
        GroupAction([rsp], condition=IfCondition(LaunchConfiguration("spawn_robot"))),
        spawn,
    ]


def generate_launch_description():
    pkg = get_package_share_directory("beehive_sim")
    world = os.path.join(pkg, "worlds", "beehive.sdf")
    models = os.path.join(pkg, "models")

    # Make model:// URIs (honeycomb_plane, work_surface) resolvable + find texture.
    set_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=models + os.pathsep + os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
    )

    args = [
        DeclareLaunchArgument("mock", default_value="true",
                              description="Mock Gemini (no API key needed)."),
        DeclareLaunchArgument("detector", default_value="geometry",
                              description="OpenCV detector: geometry|opencv|color."),
        DeclareLaunchArgument("web_dir",
                              default_value=os.environ.get("BEEHIVE_WEB_DIR", ""),
                              description="Absolute path to beehive_project/web."),
        DeclareLaunchArgument("spawn_robot", default_value="true",
                              description="Spawn the Panda for visualization."),
        DeclareLaunchArgument("rqt", default_value="false",
                              description="Open rqt_image_view on the annotated topic."),
    ]

    # Gazebo Harmonic (ros_gz_sim). "-r" runs the sim immediately.
    # Headless option: on WSL/software-GL the Gazebo GUI competes with the camera
    # sensor for the llvmpipe renderer, so the camera stalls / starves of frames.
    # Set BEEHIVE_GAZEBO_GUI=0 to run server-only (-s): the camera renders reliably
    # and you watch the arm in RViz instead. Default keeps the GUI on.
    _gui = os.environ.get("BEEHIVE_GAZEBO_GUI", "1") != "0"
    _gz_flags = "-r -v 3" if _gui else "-s -r -v 3"
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])),
        launch_arguments={"gz_args": f"{_gz_flags} {world}"}.items(),
    )

    # Bridge: Gazebo camera + clock -> ROS, remapped to clean /beehive names.
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/honeycomb_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/honeycomb_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        remappings=[
            ("/honeycomb_camera/image", "/beehive/camera/image"),
            ("/honeycomb_camera/camera_info", "/beehive/camera/camera_info"),
        ],
        parameters=[{"use_sim_time": True}],
    )

    vision = Node(
        package="beehive_sim",
        executable="vision_node",
        name="vision_node",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "image_topic": "/beehive/camera/image",
            "web_dir": LaunchConfiguration("web_dir"),
            "mock": LaunchConfiguration("mock"),
            "detector": LaunchConfiguration("detector"),
        }],
    )

    rqt = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        arguments=["/vision_node/annotated"],
        condition=IfCondition(LaunchConfiguration("rqt")),
    )

    # --- Stage 4 FIXED extrinsics, published as static TF ---
    # These must match worlds/beehive.sdf. The pixel->base projection reads them
    # from the TF tree, so there is no online calibration.
    def stf(name, args):
        return Node(package="tf2_ros", executable="static_transform_publisher",
                    name=name, arguments=args, output="screen")

    # robot base == world origin (robot spawned at 0,0,0)
    tf_world_base = stf("tf_world_base", [
        "--frame-id", "world", "--child-frame-id", "panda_link0",
        "--x", "0", "--y", "0", "--z", "0",
        "--roll", "0", "--pitch", "0", "--yaw", "0"])

    # camera body pose in the world (MUST match the <model> pose in the SDF):
    # middle of the Panda (base center, mid-height) looking forward (+X) at the
    # comb wall. Identity rotation -> camera +X = world +X, +Z = up.
    tf_world_cam = stf("tf_world_cam", [
        "--frame-id", "world", "--child-frame-id", "honeycomb_camera_link",
        "--x", "0", "--y", "0", "--z", "0.40",
        "--roll", "0", "--pitch", "0", "--yaw", "0"])

    # camera body -> optical frame (REP 103: x-forward body -> z-forward optical)
    tf_cam_optical = stf("tf_cam_optical", [
        "--frame-id", "honeycomb_camera_link",
        "--child-frame-id", "honeycomb_camera_optical_frame",
        "--x", "0", "--y", "0", "--z", "0",
        "--roll", "-1.5708", "--pitch", "0", "--yaw", "-1.5708"])

    static_tf = [tf_world_base, tf_world_cam, tf_cam_optical]

    return LaunchDescription(
        [set_resource_path] + args + [gz, bridge, vision, rqt]
        + static_tf + _maybe_robot_nodes()
    )
