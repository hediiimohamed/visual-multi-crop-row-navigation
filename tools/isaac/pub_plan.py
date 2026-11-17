#!/usr/bin/env python3
"""Publish (latched) the planned pass of the robot in the Isaac nav field, as a coverage planner would: a straight line
along the mean bed line, base_link target, odom frame (the robot spawns centred on bed 0, heading +x, so the pass is
y = shift, x from -3 to 56 m). The real rows wander around it (manual seeding). usage: pub_plan.py [shift_m] [topic]"""
import sys, rclpy, numpy as np
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
rclpy.init(); n = Node('pub_plan')
shift = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
topic = sys.argv[2] if len(sys.argv) > 2 else '/vnav/plan'
pub = n.create_publisher(Path, topic, QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
m = Path(); m.header.frame_id = 'odom'
for x in np.arange(-3.0, 56.0, 0.5):
    p = PoseStamped(); p.header.frame_id = 'odom'; p.pose.position.x = float(x); p.pose.position.y = shift; p.pose.orientation.w = 1.0
    m.poses.append(p)
pub.publish(m); n.get_logger().info('plan published on %s: y = %.2f m' % (topic, shift)); rclpy.spin(n)
