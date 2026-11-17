#!/usr/bin/env python3
"""crop safety: per wheel, smallest distance (m) from the wheel's centre line to a crop plant (truth), along the run.
Wheel contact points in base_link (gen5 track 1500: pods 1/3 right at y -0.661, pod 2 left at +0.838; x from the URDF).
usage: itest_wheels.py <dir> <field_rN> [...]"""
import sys, math, numpy as np
src = open('/home/brainy/isaac_sim_stack/vnav_ws/tmp/itest_analyze.py').read().replace('\nmain()\n', '\n')
ia = {}; exec(compile(src, 'ia', 'exec'), ia)
WHEELS = {'front-right': (2.094, -0.661), 'middle-left': (0.0, 0.838), 'rear-right': (-2.094, -0.661)}
for f in sys.argv[2:]:
    odom, status, crops, _ = ia['read']('%s/%s_bag' % (sys.argv[1], f))
    worst = {k: (9.0, 0.0) for k in WHEELS}
    hist = {k: [] for k in WHEELS}
    for t, x, y, yaw in odom[::5]:
        c, s = math.cos(yaw), math.sin(yaw)
        for k, (wx, wy) in WHEELS.items():
            px, py = x + c * wx - s * wy, y + s * wx + c * wy
            d = np.hypot(crops[:, 0] - px, crops[:, 1] - py)
            near = d < 0.6
            # lateral distance to the plants within 15 cm along the wheel (the wheel rolls over them)
            dx = (crops[near, 0] - px) * c + (crops[near, 1] - py) * s
            dy = -(crops[near, 0] - px) * s + (crops[near, 1] - py) * c
            m = np.abs(dy[np.abs(dx) < 0.15]).min() if np.any(np.abs(dx) < 0.15) else 9.0
            hist[k].append(m)
            if m < worst[k][0]: worst[k] = (m, x)
    print(f)
    for k in WHEELS:
        h = np.array(hist[k]); h = h[h < 9]
        print('  %-12s min %5.1f cm at x %5.1f m | under 20 cm on %4.1f %% of the run' % (k, 100 * worst[k][0], worst[k][1], 100 * np.mean(h < 0.20) if len(h) else 0))
