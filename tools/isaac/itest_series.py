#!/usr/bin/env python3
"""per metre: driven distance, true error (cm), last status. usage: itest_series.py <dir> <field:rows> [step]"""
import sys, math, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location('ia', '/home/brainy/isaac_sim_stack/vnav_ws/tmp/itest_analyze.py')
src = open('/home/brainy/isaac_sim_stack/vnav_ws/tmp/itest_analyze.py').read().replace('\nmain()\n', '\n')
ia = {}; exec(compile(src, 'ia', 'exec'), ia)
out, fr = sys.argv[1], sys.argv[2]; step = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
f, rows = fr.split(':'); rows = int(rows); furrow = 0.35 if rows == 3 else 0.30
odom, status, crops, pdev = ia['read']('%s/%s_bag' % (out, f))
st = [(t, s) for t, s in status]
dist, last, nextd, si, cur = 0.0, None, 0.0, 0, '-'
for t, x, y, yaw in odom[::3]:
    if last is not None:
        dist += math.hypot(x - last[0], y - last[1])
    last = (x, y)
    while si < len(st) and st[si][0] <= t:
        cur = st[si][1]; si += 1
    if dist >= nextd:
        e = ia['bedError'](x, y, yaw, crops, rows, furrow)
        print('%5.1f m  x %6.2f y %+5.2f  err %s | %s' % (dist, x, y, ('%+6.1f' % (100 * e)) if e is not None else '   n/a', cur[:95]))
        nextd += step
