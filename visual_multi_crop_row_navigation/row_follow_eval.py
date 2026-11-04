#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# How well did the robot follow the row?
#
# Compares the robot position with a reference line (a row of the field file, or two
# points) and reports lateral / heading errors, oscillations, stops and recoveries.
#
#   simulation: /odom is the Gazebo ground truth, the field file rows are in map = odom
#     ros2 run visual_multi_crop_row_navigation row_follow_eval --ros-args \
#       -p field_file:=<simulation_navigation>/maps/output.xml -p use_sim_time:=true
#   real robot: wheel odometry drifts, use an RTK GNSS pose in the field file frame, e.g.
#       -p pose_source:=tf -p frame:=map
#
# Ctrl+C (or the end of the row with stop_on_row_end) prints the summary and writes
# <output_prefix>.csv and <output_prefix>.png.
import math
import time
import signal
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import tf2_ros


def readRows(path):
    """rows of a GAOS field file (output.xml): {id: (x1, y1, x2, y2)}"""
    rows = {}
    for row in ET.parse(path).getroot().iter():
        if not row.tag.endswith('Row'):
            continue
        coords = [e.text for e in row.iter() if e.tag.endswith('coordinates')]
        if coords:
            pts = [tuple(map(float, p.split(','))) for p in coords[0].split()]
            rows[row.get('id')] = (pts[0][0], pts[0][1], pts[-1][0], pts[-1][1])
    return rows


def yawOf(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RowFollowEval(Node):

    def __init__(self):
        super().__init__('row_follow_eval')
        self.declare_parameter('field_file', '')            # GAOS output.xml with <Row> lines
        self.declare_parameter('row_id', '')                # '' = the row closest to the robot
        self.declare_parameter('line', [0.0, 0.0, 0.0, 0.0])  # x1 y1 x2 y2, instead of a field file
        self.declare_parameter('reference_offset', 0.0)     # m, target left of the reference line
        # curved rows: lateral shift of the target (m, left > 0) as a function of x, the distance
        # along the reference line, e.g. the y_function of a CropCraft bed. '' = straight
        self.declare_parameter('reference_function', '')
        self.declare_parameter('pose_source', 'odom')       # 'odom' (topic) or 'tf'
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('frame', 'odom')             # tf: reference frame (map with RTK)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('status_topic', '/vs_nav/status')
        self.declare_parameter('sample_distance', 0.05)     # m between two samples
        self.declare_parameter('skip_start', 2.0)           # m ignored in the statistics (convergence)
        self.declare_parameter('print_every', 2.0)          # m
        self.declare_parameter('stop_on_row_end', True)     # summary at ROW_END / FAILED
        self.declare_parameter('output_prefix', 'row_follow')
        g = lambda n: self.get_parameter(n).value

        self.offset = g('reference_offset')
        expr = g('reference_function')
        env = {k: getattr(math, k) for k in ('sin', 'cos', 'tan', 'pi', 'tau', 'exp', 'sqrt', 'atan')}
        self.shape = (lambda x: eval(expr, {'__builtins__': {}}, dict(env, x=x))) if expr else (lambda x: 0.0)
        self.shape(0.0)  # fails here on a bad expression
        self.sampleDistance = g('sample_distance')
        self.skipStart = g('skip_start')
        self.printEvery = g('print_every')
        self.stopOnRowEnd = g('stop_on_row_end')
        self.prefix = g('output_prefix')
        self.rowId = g('row_id')
        self.rows = readRows(g('field_file')) if g('field_file') else {}
        line = list(g('line'))
        if not self.rows and any(line):
            self.rows = {'line': tuple(line)}
        if not self.rows:
            raise RuntimeError('give field_file or line')
        self.ref = None                                     # (x1, y1, ux, uy, length, id)

        self.samples = []       # along, lateral, heading, omega, speed, time, state
        self.omega = 0.0
        self.state = ''
        self.states = []        # (along, state, text) on each change
        self.recoveries = 0     # TRACKING / EXIT -> SCAN: rows lost
        self.lastPose = None
        self.nextPrint = 0.0
        self.done = False

        if g('pose_source') == 'tf':
            self.tfBuffer = tf2_ros.Buffer()
            self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)
            self.frames = (g('frame'), g('base_frame'))
            self.create_timer(0.05, self.tfPose)
        else:
            self.create_subscription(Odometry, g('odom_topic'), self.odomCallback, 20)
        self.create_subscription(Twist, g('cmd_vel_topic'), lambda m: setattr(self, 'omega', m.angular.z), 10)
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, g('status_topic'), self.statusCallback, latched)
        self.get_logger().info('Waiting for the robot pose...')

    # ------------------------------------------------------------------ inputs
    def odomCallback(self, msg):
        p = msg.pose.pose
        self.addPose(p.position.x, p.position.y, yawOf(p.orientation),
                     math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y))

    def tfPose(self):
        try:
            tf = self.tfBuffer.lookup_transform(self.frames[0], self.frames[1], Time())
        except tf2_ros.TransformException:
            return
        t = tf.transform.translation
        speed = 0.0
        if self.lastPose is not None:
            dt = max(self.now() - self.lastPose[3], 1e-3)
            speed = math.hypot(t.x - self.lastPose[0], t.y - self.lastPose[1]) / dt
        self.addPose(t.x, t.y, yawOf(tf.transform.rotation), speed)

    def statusCallback(self, msg):
        text = msg.data
        state = text.split(':')[0].split(' ')[0]
        if state != self.state:
            if state == 'SCAN' and self.state in ('TRACKING', 'EXIT'):
                self.recoveries += 1
            along = self.samples[-1][0] if self.samples else 0.0
            self.states.append((along, state, text))
            self.get_logger().info(f'{along:6.2f} m  {text[:100]}')
            self.state = state
            if self.stopOnRowEnd and state in ('ROW_END', 'FAILED'):
                self.done = True

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ geometry
    def chooseRow(self, x, y):
        """reference: the given row, or the row whose line passes closest to the robot"""
        best = None
        for rid, (x1, y1, x2, y2) in self.rows.items():
            if self.rowId and rid != self.rowId:
                continue
            L = math.hypot(x2 - x1, y2 - y1)
            ux, uy = (x2 - x1) / L, (y2 - y1) / L
            d = abs(-(x - x1) * uy + (y - y1) * ux)
            if best is None or d < best[0]:
                best = (d, (x1, y1, ux, uy, L, rid))
        if best is None:
            raise RuntimeError('row %s not in the field file' % self.rowId)
        self.ref = best[1]
        self.get_logger().info('Reference: row %s, %.1f m long, robot %.2f m from it'
                               % (self.ref[5], self.ref[4], best[0]))

    def addPose(self, x, y, yaw, speed):
        t = self.now()
        if self.ref is None:
            self.chooseRow(x, y)
        x1, y1, ux, uy, L, _ = self.ref
        along = (x - x1) * ux + (y - y1) * uy
        # left of the target line > 0; curved rows: target shifted by shape(along), heading
        # measured against its local direction
        lateral = -(x - x1) * uy + (y - y1) * ux - self.offset - self.shape(along)
        direction = math.atan2(uy, ux) + math.atan((self.shape(along + 0.05) - self.shape(along - 0.05)) / 0.1)
        heading = math.degrees(math.atan2(math.sin(yaw - direction), math.cos(yaw - direction)))
        if self.samples and abs(along - self.samples[-1][0]) < self.sampleDistance \
                and abs(t - self.samples[-1][5]) < 1.0:
            self.lastPose = (x, y, yaw, t)
            return
        self.samples.append((along, lateral, heading, self.omega, speed, t, self.state))
        self.lastPose = (x, y, yaw, t)
        if along >= self.nextPrint:
            self.nextPrint = along + self.printEvery
            print(f'along {along:6.2f} m | lateral {100 * lateral:+6.1f} cm | heading {heading:+5.1f} deg'
                  f' | omega {self.omega:+.3f} | {self.state}', flush=True)

    # ------------------------------------------------------------------ results
    def summary(self):
        if len(self.samples) < 5:
            print('Not enough samples (no pose received?)')
            return
        a = np.array([s[:6] for s in self.samples], dtype=float)
        along, lat, hdg, omega, speed, t = a.T
        m = along >= along[0] + self.skipStart
        if np.count_nonzero(m) < 5:
            m = np.ones(len(a), dtype=bool)
        L, H = 100 * lat[m], hdg[m]

        def crossings(x, hyst):
            """sign changes of x around its mean, with hysteresis: left-right corrections"""
            x = x - np.mean(x)
            n, sign = 0, 0
            for v in x:
                s = 1 if v > hyst else -1 if v < -hyst else 0
                if s and sign and s != sign:
                    n += 1
                if s:
                    sign = s
            return n

        dist = along[m][-1] - along[m][0]
        per10 = 10.0 / max(dist, 1e-3)
        stopped = float(np.sum(np.diff(t[m])[speed[m][1:] < 0.02]))
        lines = [
            '',
            '================ ROW FOLLOWING ================',
            'reference         row %s (offset %+.2f m)' % (self.ref[5], self.offset),
            'distance          %.2f m evaluated (%.2f -> %.2f m along the row), %.0f s'
            % (dist, along[m][0], along[m][-1], t[m][-1] - t[m][0]),
            'lateral error     mean %+.1f cm | std %.1f cm | RMS %.1f cm | max |%.1f| cm | 95%% within %.1f cm'
            % (L.mean(), L.std(), np.sqrt(np.mean(L ** 2)), np.abs(L).max(), np.percentile(np.abs(L), 95)),
            'within +/-2 cm    %.0f %%   within +/-5 cm  %.0f %%'
            % (100 * np.mean(np.abs(L) <= 2), 100 * np.mean(np.abs(L) <= 5)),
            'heading error     mean %+.2f deg | std %.2f deg | max |%.1f| deg' % (H.mean(), H.std(), np.abs(H).max()),
            'oscillation       %.1f left/right heading swings per 10 m, %.1f steering sign changes per 10 m'
            % (crossings(H, 0.3) * per10, crossings(omega[m], 0.005) * per10),
            'stops             %.0f s stopped, %d recovery(ies)' % (stopped, self.recoveries),
            'end               %s' % (self.states[-1][2][:90] if self.states else self.state),
            '===============================================',
        ]
        print('\n'.join(lines), flush=True)

        with open(self.prefix + '.csv', 'w') as f:
            f.write('along_m,lateral_m,heading_deg,omega_rad_s,speed_m_s,time_s,state\n')
            for s in self.samples:
                f.write('%.3f,%.4f,%.3f,%.4f,%.3f,%.3f,%s\n' % s)
        self.plot(along, lat, hdg, omega, m)
        print('written %s.csv, %s.png' % (self.prefix, self.prefix), flush=True)

    def plot(self, along, lat, hdg, omega, m):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        ax[0].plot(along, 100 * lat, lw=1)
        ax[0].axhspan(-5, 5, color='green', alpha=0.08)
        ax[0].set_ylabel('lateral [cm]')
        ax[1].plot(along, hdg, lw=1)
        ax[1].set_ylabel('heading [deg]')
        ax[2].plot(along, omega, lw=1)
        ax[2].set_ylabel('omega [rad/s]')
        ax[2].set_xlabel('along the row [m]')
        for a in ax:
            a.axhline(0, color='k', lw=0.5)
            a.grid(alpha=0.3)
            for x, s, _ in self.states:
                a.axvline(x, color='red' if s in ('SCAN', 'ADVANCE', 'FAILED') else 'gray', lw=0.6, ls='--')
        ax[0].set_title('row %s: lateral std %.1f cm, max %.1f cm' % (
            self.ref[5], 100 * lat[m].std(), 100 * np.abs(lat[m]).max()))
        fig.tight_layout()
        fig.savefig(self.prefix + '.png', dpi=110)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(True))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    node = RowFollowEval()
    try:
        while rclpy.ok() and not stop and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.done:
            # a few more samples after the row end (robot stopping)
            end = time.monotonic() + 1.0
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
