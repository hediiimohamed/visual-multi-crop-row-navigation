#!/usr/bin/env python3
"""save /vs_nav/graphic (+ raw camera, + status) every STEP m of odom x within [X0, X1], and at every new SCAN/lost status.
usage: grab_along.py <out dir> <x0> <x1> [step]"""
import sys, os, time, numpy as np, cv2, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSDurabilityPolicy
out, x0, x1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); step = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
os.makedirs(out, exist_ok=True)
rclpy.init(); n = Node('grab_along'); st = {'x': None, 'y': None, 'next': x0, 'status': '', 'last_status_saved': '', 'img': None, 'raw': None}
def toimg(m):
    a = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
    return a[:, :, ::-1] if m.encoding == 'rgb8' else a
def save(tag):
    if st['img'] is None: return
    name = '%s/x%05.2f_%s.png' % (out, st['x'], tag)
    img = st['img'].copy()
    cv2.putText(img, 'odom x %.2f y %+.2f | %s' % (st['x'], st['y'], st['status'][:90]), (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(name, img)
    if st['raw'] is not None: cv2.imwrite(name.replace('.png', '_raw.jpg'), st['raw'])
    print('saved', name, st['status'][:100], flush=True)
def odom(m):
    st['x'], st['y'] = m.pose.pose.position.x, m.pose.pose.position.y
    if x0 <= st['x'] <= x1 and st['x'] >= st['next']:
        save('step'); st['next'] = st['x'] + step
def status(m):
    st['status'] = m.data
    key = m.data.split(':')[0] + m.data.split(':')[1][:25] if ':' in m.data else m.data
    if st['x'] is not None and x0 - 2 <= st['x'] <= x1 + 2 and ('SCAN' in m.data[:5] or 'rejected' in m.data) and key != st['last_status_saved']:
        st['last_status_saved'] = key; save('event')
n.create_subscription(Odometry, '/odom', odom, 10)
n.create_subscription(String, '/vs_nav/status', status, QoSProfile(depth=5, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
n.create_subscription(Image, '/vs_nav/graphic', lambda m: st.update(img=toimg(m)), 1)
n.create_subscription(Image, '/visual_nav_camera/rgb/image_raw', lambda m: st.update(raw=toimg(m)), qos_profile_sensor_data)
end = time.time() + 3 * 3600
while time.time() < end and (st['x'] is None or st['x'] < x1 + 1.0):
    rclpy.spin_once(n, timeout_sec=0.2)
print('done', flush=True)
