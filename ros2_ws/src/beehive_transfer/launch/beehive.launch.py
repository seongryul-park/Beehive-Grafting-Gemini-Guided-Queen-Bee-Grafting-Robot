# =============================================================================
#  beehive.launch.py  —  robot_node (+ optional rosbridge).
# -----------------------------------------------------------------------------
#  주의: Panda + MoveIt(move_group + RViz) 데모는 별도 터미널에서 먼저!
#        ros2 launch moveit_resources_panda_moveit_config demo.launch.py
#
#  실행:
#        ros2 launch beehive_transfer beehive.launch.py
#        ros2 launch beehive_transfer beehive.launch.py rosbridge:=true   # web UI
# =============================================================================

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # robot_node: subscribes /selected_pose -> MoveIt pick & place into tray slots.
    # It consumes ONLY the base-frame pose from the vision stage (no pixel math).
    robot_node = Node(
        package="beehive_transfer",
        executable="robot_node",
        name="robot_node",
        output="screen",
        parameters=[{"target_topic": "/selected_pose"}],
    )

    actions = [
        # rosbridge is only for the web dashboard (:9090); the Gazebo demo does
        # NOT need it. Default OFF so robot_node always comes up cleanly.
        DeclareLaunchArgument(
            "rosbridge", default_value="false",
            description="Also start rosbridge_websocket (:9090) for the web UI."),
        robot_node,
    ]

    # rosbridge ships an XML launch file, so it MUST be included with
    # AnyLaunchDescriptionSource. The previous PythonLaunchDescriptionSource tried
    # to parse the XML as Python -> 'invalid syntax (…launch.xml, line 1)', which
    # aborted the whole launch (and robot_node never started). Guarded so a missing
    # rosbridge_server package can't take robot_node down either.
    try:
        rb_xml = os.path.join(
            get_package_share_directory("rosbridge_server"),
            "launch", "rosbridge_websocket_launch.xml")
        actions.append(IncludeLaunchDescription(
            AnyLaunchDescriptionSource(rb_xml),
            condition=IfCondition(LaunchConfiguration("rosbridge")),
        ))
    except Exception as e:  # noqa: BLE001
        print(f"[beehive_transfer] rosbridge_server not found ({e}); skipping.")

    return LaunchDescription(actions)
