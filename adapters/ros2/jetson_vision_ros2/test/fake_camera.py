import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
rclpy.init()
n = Node("fake_cam")
pub = n.create_publisher(Image, "/image_raw", qos_profile_sensor_data)
import numpy as np  # noqa: E402
img = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
def tick():
    m = Image()
    m.header.stamp.sec = 1234
    m.header.stamp.nanosec = 567
    m.header.frame_id = "cam_test"
    m.height, m.width, m.encoding, m.step = 48, 64, "rgb8", 64 * 3
    m.data = img.tobytes()
    pub.publish(m)
n.create_timer(0.2, tick)
rclpy.spin(n)
