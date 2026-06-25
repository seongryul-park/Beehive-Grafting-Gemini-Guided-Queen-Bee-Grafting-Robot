cmake_minimum_required(VERSION 3.8)
project(beehive_transfer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- 의존성 ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(shape_msgs REQUIRED)

# =====================================================================
#  robot_node : /larva_targets 구독 -> MoveIt 다중 pick & place
#               + /robot_status publish
# =====================================================================
add_executable(robot_node src/robot_node.cpp)
ament_target_dependencies(robot_node
  rclcpp
  geometry_msgs
  std_msgs
  moveit_ros_planning_interface
  moveit_msgs
  shape_msgs
)

install(TARGETS
  robot_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
