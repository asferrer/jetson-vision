# Smoke test in a ros:humble container (CPU, fake backend). Run: see README, section ROS 2.
source /opt/ros/humble/setup.bash
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq ros-humble-vision-msgs python3-opencv python3-yaml >/dev/null 2>&1
mkdir -p /ws/src && cp -r /repo/adapters/ros2/* /ws/src/ && cd /ws
colcon build > /tmp/build.log 2>&1; tail -4 /tmp/build.log
source /ws/install/setup.bash
export PYTHONPATH=/repo:$PYTHONPATH
ros2 run jetson_vision_ros2 vision_node --ros-args -p manifest:=/repo/models/rfdetr_int8g.yaml -p backend:=fake -p image_topic:=/image_raw > /tmp/node.log 2>&1 &
sleep 4
python3 /repo/adapters/ros2/jetson_vision_ros2/test/fake_camera.py > /tmp/pub.log 2>&1 &
sleep 2
echo "--- lifecycle"
ros2 lifecycle get /jetson_vision
ros2 lifecycle set /jetson_vision configure
ros2 lifecycle set /jetson_vision activate
sleep 2
echo "--- topics"
ros2 topic list
echo "--- sub QoS"
ros2 topic info -v /image_raw | grep -E "Node name|Reliability|History|Depth" 
echo "--- detections header"
timeout 10 ros2 topic echo --once /jetson_vision/detections --field header
echo "--- first detection"
timeout 10 ros2 topic echo --once /jetson_vision/detections --field detections | head -45
echo "--- frame_meta"
timeout 10 ros2 topic echo --once /jetson_vision/frame_meta
echo "--- diagnostics"
timeout 10 ros2 topic echo --once /diagnostics
echo "--- hz"
timeout 6 ros2 topic hz /jetson_vision/detections | head -3
echo "--- deactivate/cleanup"
ros2 lifecycle set /jetson_vision deactivate
ros2 lifecycle set /jetson_vision cleanup
echo "--- node log"; cat /tmp/node.log | tail -5
