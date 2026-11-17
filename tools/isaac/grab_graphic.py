#!/usr/bin/env python3
"""save one frame of each given image topic as PNG. usage: grab_graphic.py <out prefix> <topic> [<topic> ...]"""
import sys, time, numpy as np, cv2, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
rclpy.init(); n = Node('grab_graphic'); prefix, topics = sys.argv[1], sys.argv[2:]; got = {}
def cb(topic):
    def f(m):
        if topic in got: return
        a = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.step)[:, :m.width * (3 if m.encoding in ('rgb8', 'bgr8') else 1)]
        img = a.reshape(m.height, m.width, -1) if m.encoding in ('rgb8', 'bgr8') else a.reshape(m.height, m.width)
        if m.encoding == 'rgb8': img = img[:, :, ::-1]
        name = '%s_%s.png' % (prefix, topic.strip('/').replace('/', '_'))
        cv2.imwrite(name, img); got[topic] = name; print('saved', name, m.encoding, m.width, m.height, flush=True)
    return f
for t in topics:
    n.create_subscription(Image, t, cb(t), qos_profile_sensor_data)
end = time.time() + 30
while time.time() < end and len(got) < len(topics):
    rclpy.spin_once(n, timeout_sec=0.5)
print('missing:', [t for t in topics if t not in got])
