"""Launch the node and bring it to the active state. Parameters come from the command line."""
import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    node = launch_ros.actions.LifecycleNode(
        package="jetson_vision_ros2", executable="vision_node", name="jetson_vision",
        namespace="",
        parameters=[{
            "manifest": LaunchConfiguration("manifest"),
            "backend": LaunchConfiguration("backend"),
            "image_topic": LaunchConfiguration("image_topic"),
            "input_fps": LaunchConfiguration("input_fps"),
        }],
        output="screen",
    )
    configure = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=launch.events.matchers.matches_action(node),
        transition_id=Transition.TRANSITION_CONFIGURE))
    activate = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=launch.events.matchers.matches_action(node),
        transition_id=Transition.TRANSITION_ACTIVATE))
    return launch.LaunchDescription([
        DeclareLaunchArgument("manifest", default_value=""),
        DeclareLaunchArgument("backend", default_value="tensorrt"),
        DeclareLaunchArgument("image_topic", default_value="image_raw"),
        DeclareLaunchArgument("input_fps", default_value="30.0"),
        node,
        RegisterEventHandler(OnProcessStart(target_action=node, on_start=[configure])),
        RegisterEventHandler(OnStateTransition(
            target_lifecycle_node=node, goal_state="inactive", entities=[activate])),
    ])
