#!/usr/bin/env python3
"""Analyse the Isaac field test bags: true lateral error of base_link from the centre of the bed it should follow
(computed from the real plant positions around the robot, like the reference of show_field but per local window),
distance driven off the lane, losses / failures from vs_nav/status.
usage (in eterry_host, env.sh sourced): python3 itest_analyze.py <itest dir> <field:rows> ..."""
import sys, math, json, collections
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32
from visualization_msgs.msg import MarkerArray

BED_OFF = 0.0885          # bed centre left of base_link when the robot is centred (field.json)


def read(bag):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    odom, status, crops, pdev = [], [], None, []
    while r.has_next():
        t, d, ts = r.read_next()
        if t == '/odom':
            m = deserialize_message(d, Odometry); p = m.pose.pose
            yaw = math.atan2(2 * (p.orientation.w * p.orientation.z), 1 - 2 * p.orientation.z ** 2)
            odom.append((ts * 1e-9, p.position.x, p.position.y, yaw))
        elif t == '/vs_nav/status':
            status.append((ts * 1e-9, deserialize_message(d, String).data))
        elif t == '/vs_nav/path_deviation':
            pdev.append(deserialize_message(d, Float32).data)
        elif t == '/field/crops' and crops is None:
            m = deserialize_message(d, MarkerArray)
            crops = np.array([[q.x, q.y] for mk in m.markers if not mk.ns.startswith('weeds') for q in mk.points])
    return np.array(odom), status, crops, pdev


def bedError(x, y, yaw, crops, rows, furrow):
    """robot error (m, + = left) from the centre of the bed under it: plant rows within [-0.5, 1.5] m along the robot,
    grouped into beds by gaps > 1.4 furrow, the bed whose centre is closest to the target"""
    c, s = math.cos(yaw), math.sin(yaw)
    d = crops - [x, y]
    rx, ry = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    ys = np.sort(ry[(rx > -0.5) & (rx < 1.5) & (np.abs(ry) < 2.0)])
    if len(ys) < 6:
        return None
    groups = np.split(ys, np.flatnonzero(np.diff(ys) > 1.4 * furrow) + 1)
    centres = [(g.min() + g.max()) / 2 for g in groups
               if len(g) >= 4 and abs((g.max() - g.min()) - (rows - 1) * furrow) < 0.8 * furrow]
    if not centres:
        return None
    e = min(centres, key=lambda v: abs(v - BED_OFF))
    return -(e - BED_OFF)


def main():
    out = sys.argv[1]
    for fr in sys.argv[2:]:
        f, rows = fr.split(':'); rows = int(rows); furrow = 0.35 if rows == 3 else 0.30
        try:
            odom, status, crops, pdev = read('%s/%s_bag' % (out, f))
        except Exception as e:
            print('%-7s no bag (%s)' % (f, e)); continue
        if crops is None or not len(odom):
            print('%-7s no crops / odom in the bag' % f); continue
        errs, dist, off, last = [], 0.0, 0.0, None
        for t, x, y, yaw in odom[::5]:
            if last is not None:
                step = math.hypot(x - last[0], y - last[1])
                dist += step
            e = bedError(x, y, yaw, crops, rows, furrow)
            if e is not None:
                errs.append(e)
                if abs(e) > furrow / 2 and last is not None:
                    off += step
            last = (x, y)
        states = [s.split(':')[0] for _, s in status]
        pairs = [(a, b) for a, b in zip(states, states[1:]) if a != b]
        losses = sum(1 for a, b in pairs if a in ('TRACKING', 'EXIT') and b == 'SCAN')
        final = status[-1][1][:110] if status else '-'
        e = np.abs(errs) if errs else np.zeros(1)
        pd = ' | plan dev max %.2f m' % max(pdev) if pdev else ''
        print('%-7s driven %5.1f m | true error mean %4.1f p95 %4.1f max %5.1f cm | off lane %5.1f m | lane losses %d%s | final: %s'
              % (f, dist, 100 * e.mean(), 100 * np.percentile(e, 95), 100 * e.max(), off, losses, pd, final))


main()
