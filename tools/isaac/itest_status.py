#!/usr/bin/env python3
"""distinct consecutive vs_nav/status messages of a bag with the true error. usage: itest_status.py <bag> <rows> [max]"""
import sys, math
src = open('/home/brainy/isaac_sim_stack/vnav_ws/tmp/itest_analyze.py').read().replace('\nmain()\n', '\n')
ia = {}; exec(compile(src, 'ia', 'exec'), ia)
bag, rows = sys.argv[1], int(sys.argv[2]); mx = int(sys.argv[3]) if len(sys.argv) > 3 else 60
furrow = 0.35 if rows == 3 else 0.30
odom, status, crops, pdev = ia['read'](bag)
import numpy as np
oi, last, n = 0, None, 0
for t, s in status:
    key = s.split('[')[0][:60]
    if key == last:
        continue
    last = key
    while oi + 1 < len(odom) and odom[oi + 1][0] <= t:
        oi += 1
    _, x, y, yaw = odom[oi]
    e = ia['bedError'](x, y, yaw, crops, rows, furrow)
    print('t %7.1f x %6.2f y %+5.2f err %s | %s' % (t - status[0][0], x, y, ('%+5.1f' % (100 * e)) if e is not None else '  n/a', s[:150]))
    n += 1
    if n >= mx:
        break
