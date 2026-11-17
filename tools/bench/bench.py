#!/usr/bin/env python3
"""Offline bench of the crop-row tracker (no ROS): RowTracker driving pure pursuit on a simulated bed field, seen by
the Isaac ZED camera model (params_isaac_unified.yaml: 960x540 processing, view topOffset/bottomOffset 160), with
the weeding box hiding the bottom centre of the image. Fast: 30 seeds of a field in a few minutes (run_seeds.py).

usage: python3 -I tools/bench/bench.py [tracker.py] <seed>
  tracker.py defaults to visual_multi_crop_row_navigation/groundTracker.py of this repository.
Field and run settings are environment variables (defaults of the FIELD preset):
  FIELD=beds3|seed3|weeds2|onion2   preset (3 rows 0.35 m late | 3 rows 0.35 m seedlings | 4 rows 0.30 m, weeds |
                             4 rows 0.30 m onions every 10 cm, fan of narrow detections SPRY, weeds WP)
  R SP                       rows per bed, row spacing (m); beds are 1.5 m apart, 5 beds
  WF WB WR                   lateral wander amplitude (m) of the field / of each bed / of each row (sines, 3-45 m)
  JY JX GAPS PS              plant jitter across / along (m), share of missing plants, plant spacing (m)
  ND PD CW                   detections per crop plant, detection probability, crop width (m)
  WD WEEDW                   weeds per m2 (uniform + one patch per bed per 10 m), weed width (m)
  SPRY                       spread (m) of the detections of a crop plant across the row (onion leaves)
  WP                         >0: Isaac generator weeds instead (make_beds_from_big1.py --weeds WD --weed-patches WP)
  NF                         false detections per image
  L                          field length (m; changes the random field), the run stops 8 m before its end
  LANEROWS FURROW            tracker lane_rows / furrow_width (0: measured)
  TOP BOT                    image rows hidden at the top / bottom (view), 160 / 160 as in Isaac
  Y0 YAW0                    robot placed off the lane centre (m) / turned (deg) at the start
  CTRL=pp|oracle             pure pursuit on the tracked lane | on the TRUE lane centre (perfect follower)
  LD                         pure pursuit lookahead (m)
  VBC=clean                  vision box under the robot (5 % misclassified, VBERR) feeding RowTracker.boxDetections
  CONTACT                    wheel axis to plant centre distance counted as a contact (m, 0.1575 = stem under the tyre)
  TRACE=x0,x1                print the tracker state between x0 and x1
Prints one result line ('ROWSFOUND n of R | [FAILED] seed s: dist m, losses k, frames off lane N, err ...'),
a WHEELS line (closest wheel-to-crop distances, contact frames) and FAILSTATUS when the run stops."""
import os, sys, math, random, importlib.util
import numpy as np

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
trackerPath = args[0] if len(args) > 1 else os.path.join(HERE, '..', '..', 'visual_multi_crop_row_navigation', 'groundTracker.py')
seed = int(args[-1])
spec = importlib.util.spec_from_file_location('gt', trackerPath)
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)

FIELDS = {   # rows per bed, spacing, wander field/bed/row, jitter y/x, gaps, plant spacing, detections per plant,
             # detection prob, crop width (m), weeds per m2
    'beds3': dict(R=3, SP=0.35, WF=0.30, WB=0.05, WR=0.05, JY=0.02, JX=0.05, GAPS=0.08, PS=0.20, ND=3, PD=1.0, CW=0.12, WD=0.0),
    'seed3': dict(R=3, SP=0.35, WF=0.30, WB=0.05, WR=0.05, JY=0.02, JX=0.05, GAPS=0.08, PS=0.30, ND=1, PD=0.7, CW=0.02, WD=0.0),
    'weeds2': dict(R=4, SP=0.30, WF=0.45, WB=0.06, WR=0.05, JY=0.025, JX=0.06, GAPS=0.12, PS=0.30, ND=3, PD=1.0, CW=0.15, WD=3.0),
    # Isaac onion2: onions every 10 cm, thin leaves in a fan: a few narrow detections per plant, spread across the row
    'onion2': dict(R=4, SP=0.30, WF=0.45, WB=0.06, WR=0.05, JY=0.025, JX=0.06, GAPS=0.12, PS=0.10, ND=2, PD=0.9, CW=0.03, WD=3.0,
                   WP=1.0, SPRY=0.04),
}
F = dict(FIELDS[E('FIELD', 'beds3')])
for k in F:
    F[k] = type(F[k])(E(k, F[k]))
L = float(E('L', 50.0)); NF = int(E('NF', 3)); WEEDW = float(E('WEEDW', 0.04))
WP = float(E('WP', F.get('WP', 0.0))); SPRY = float(E('SPRY', F.get('SPRY', 0.0)))   # SPRY: spread (m) of a plant's detections
rnd = random.Random(seed)
nrng = np.random.default_rng(seed)


def wander(amp, lams):
    comps = [(w * amp, lam * rnd.uniform(0.8, 1.25), rnd.uniform(0, 2 * math.pi)) for w, lam in zip((0.5, 0.33, 0.17), lams)]
    return lambda x: sum(A * math.sin(2 * math.pi * x / l + ph) for A, l, ph in comps)


R, SP = F['R'], F['SP']
fw = wander(F['WF'], (30.0, 15.0, 8.0))
beds = range(-2, 3)
bw = {m: wander(F['WB'], (45.0, 22.0, 12.0)) for m in beds}
rw = {(m, j): wander(F['WR'], (11.0, 6.0, 3.0)) for m in beds for j in range(R)}


def rowY(m, j, x):
    return 1.5 * m + (j - (R - 1) / 2) * SP + fw(x) + bw[m](x) + rw[(m, j)](x)


def laneY(x):
    return np.mean([rowY(0, j, x) for j in range(R)])


pl = []    # x, y, width, 1 crop / 0 weed
for m in beds:
    for j in range(R):
        for x in np.arange(-3.0, L, F['PS']):
            if rnd.random() < F['GAPS']:
                continue
            xx = x + rnd.gauss(0, F['JX'])
            pl.append((xx, rowY(m, j, xx) + rnd.gauss(0, F['JY']), F['CW'], 1.0))
if WP <= 0:
    # weeds: uniform, plus one patch per bed per 10 m
    area = L * 7.5
    for _ in range(int(F['WD'] * area * 0.7)):
        x = rnd.uniform(-3, L)
        pl.append((x, rnd.uniform(-3.75, 3.75) + fw(x), WEEDW, 0.0))
    for m in beds:
        for _ in range(int(L / 10)):
            cx = rnd.uniform(0, L); cy = 1.5 * m + rnd.uniform(-0.75, 0.75) + fw(cx)
            for _ in range(int(F['WD'] * area * 0.3 / (len(beds) * L / 10))):
                pl.append((cx + rnd.gauss(0, 0.4), cy + rnd.gauss(0, 0.2), WEEDW, 0.0))
else:
    # as make_beds_from_big1.py: per bed, WD weeds/m2 over the bed and half of each wheel path (+-0.75 m around the
    # wandering bed centre), and WP patches per 10 m, each 10-40 weeds in a disc of radius 0.2-0.6 m
    def addWeed(m, x, u):
        if -3.0 <= x <= L:
            pl.append((x, 1.5 * m + u + fw(x) + bw[m](x), WEEDW, 0.0))
    for m in beds:
        for _ in range(int(round(F['WD'] * (L + 3.0) * 1.5))):
            addWeed(m, rnd.uniform(-3.0, L), rnd.uniform(-0.75, 0.75))
        for _ in range(int(round(WP * L / 10))):
            px, pu, pr = rnd.uniform(0, L), rnd.uniform(-0.6, 0.6), rnd.uniform(0.2, 0.6)
            for _ in range(rnd.randint(10, 40)):
                rr, th = pr * math.sqrt(rnd.random()), rnd.uniform(0, 2 * math.pi)
                addWeed(m, px + rr * math.cos(th), max(-0.75, min(0.75, pu + rr * math.sin(th))))
pl = np.array(pl)
CROPS = pl[pl[:, 3] > 0.5]

f = 480 / math.tan(math.radians(50.5))
P = gt.GroundProjector(0.13, 0.135, 1.24 - 0.12, -0.60)
P.setIntrinsics(f, f, 479.5, 269.5)
TOP = int(E('TOP', 160)); BOT = int(E('BOT', 160))
p = dict(minRows=2, minPoints=3, furrowWidth=float(E('FURROW', 0.0)), furrowTolerance=0.25, minFurrowWidth=0.2,
         maxRowAngleSpread=10.0, historySize=400, historyLength=float(E('HL', 1.0)), mergeRadius=0.03,
         maxPointWeight=10.0, gateRatio=float(E('GR', 0.25)), minGate=0.05, maxLateralJump=0.05, maxLateralError=0.0,
         maxHeadingError=15.0, maxBadFrames=5, maxRowGap=1.0, maxFurrowWidth=1.2, laneRows=int(E('LANEROWS', 0)),
         initConfirmFrames=3, rowEndFrames=5, rowEndDistance=0.5, targetOffset=0.0, maxCurvature=float(E('MC', 0.2)),
         rowConsistency=float(E('RC', 0.35)), cropWidthRatio=float(E('CWR', 0.4)))
if E('GS'):
    p['growStep'] = float(E('GS'))
T = gt.RowTracker(P, p)
vbrng = np.random.default_rng(seed + 2000)

x, y = 0.0, laneY(0.0) + float(E('Y0', 0.0))
yaw = math.atan((laneY(0.1) - laneY(0.0)) / 0.1) + math.radians(float(E('YAW0', 0.0)))
v, dt, Ld, maxOm = 0.2, 1 / 15, float(E('LD', 2.5)), 0.15
xNear, xFar = T.viewRange(540, TOP)
xRef = xNear
CTRL = E('CTRL', 'pp')
WHEELS = [(2.094, -0.75), (0.0, 0.75), (-2.094, -0.75)]   # front-right, middle-left, rear-right, from the lane centre
CONTACT = float(E('CONTACT', 0.20))
wheelMin, wheelHit = [9.0, 9.0, 9.0], [0, 0, 0]
errs, losses, wrong, frames, inited = [], 0, 0, 0, False
lostAt, waited, recov, failed = [], 0, 0.0, False
TRACE = [float(t) for t in E('TRACE').split(',')] if E('TRACE') else None
while x < L - 8:
    frames += 1
    pose = (x, y, yaw)
    # plants in view: detections (several per crop plant), noise, false detections, pixels, vegetation widths
    rx, ry = gt.toRobot(pose, pl[:, 0], pl[:, 1])
    wg, crop = pl[:, 2], pl[:, 3] > 0.5
    sel = (rx > 0.3) & (rx < 12) & (nrng.random(len(rx)) < F['PD'])
    rx, ry, wg, crop = rx[sel], ry[sel], wg[sel], crop[sel]
    nd = np.where(crop, F['ND'], 1)
    rx, ry, wg, crop = np.repeat(rx, nd), np.repeat(ry, nd), np.repeat(wg, nd), np.repeat(crop, nd)
    if SPRY > 0:
        ry = ry + np.where(crop, nrng.normal(0, SPRY, len(ry)), 0.0)
    big = wg > 0.05
    rx = rx + nrng.normal(0, 1, len(rx)) * np.where(big, 0.03, 0.01)
    ry = ry + nrng.normal(0, 1, len(ry)) * np.where(big, 0.03, 0.01)
    fx = nrng.uniform(0.9, 8.0, NF); fy = nrng.uniform(-3.0, 3.0, NF)
    rx = np.r_[rx, fx]; ry = np.r_[ry, fy]; wg = np.r_[wg, np.full(NF, 0.01)]
    u, vv = P.groundToPixel(rx, ry)
    ul, _ = P.groundToPixel(rx, ry + wg / 2)
    ur, _ = P.groundToPixel(rx, ry - wg / 2)
    wpx = np.maximum(np.abs(ur - ul), 1.0)
    ok = (u >= 0) & (u < 960) & (vv >= TOP) & (vv < 540 - BOT) & ~((u > 400) & (u < 690) & (vv > 365))
    pix, wpx = np.c_[u[ok], vv[ok]], wpx[ok]
    if not inited:
        # recovery as the node: scan (15 images), then 10 cm steps, FAILED after 1 m; off lane: stop at once
        inited = T.initialize(pix, pose, xRef, wpx)
        if not inited and getattr(T, 'robotOffLane', lambda: False)():
            failed = True
            print('FAILSTATUS x %.2f: off lane: %s' % (x, T.status))
            break
        if not inited:
            waited += 1
            if waited % 15 == 0:
                x += 0.10 * math.cos(yaw); y += 0.10 * math.sin(yaw); recov += 0.10
                if recov > 1.0 + 1e-6:
                    failed = True
                    print('FAILSTATUS x %.2f: %s' % (x, T.status))
                    break
            continue
        waited, recov = 0, 0.0
        if TRACE and TRACE[0] < x < TRACE[1]:
            print('x %5.2f INIT %s true err %+.1f' % (x, T.status, 100 * (y - laneY(x))))
    else:
        # plants of each row: the closest row line within half a furrow (the tracking boxes)
        rows, _ = T.rowLines(pose, xRef)
        gx, gy = P.pixelToGround(pix[:, 0], pix[:, 1])
        d = np.array([np.abs(gy - (a + b * (gx - xRef))) for a, b in rows])
        k = np.argmin(d, axis=0)
        inBox = d[k, np.arange(len(k))] < 0.5 * T.furrowWidth
        rowPixels = [pix[(k == i) & inBox] for i in range(len(rows))]
        rowWidths = [wpx[(k == i) & inBox] for i in range(len(rows))]
        if E('VBC') and hasattr(T, 'boxDetections'):
            # vision box under the robot: plants in its view, crop / weed with VBERR errors, 1 cm noise
            bx, by = gt.toRobot(pose, pl[:, 0], pl[:, 1])
            inb = (bx > 0.13) & (bx < 0.69) & (by > -1.2) & (by < 0.6)
            cr = (pl[inb, 3] > 0.5) ^ (vbrng.random(np.count_nonzero(inb)) < float(E('VBERR', 0.05)))
            T.boxDetections(pose, np.c_[bx[inb], by[inb]] + vbrng.normal(0, 0.01, (np.count_nonzero(inb), 2)), cr)
        if not T.update(rowPixels, pose, xRef, xNear, (xNear + xFar) / 2, False, rowWidths):
            if TRACE and TRACE[0] < x < TRACE[1]:
                print('x %5.2f LOST %s true err %+.1f' % (x, T.status, 100 * (y - laneY(x))))
            losses += 1
            lostAt.append((round(x, 1), T.status[:70]))
            inited = False
            T.reset()
            continue
    # steering
    a, b = T.center
    c = getattr(T, 'curvature', 0.0)
    if CTRL == 'oracle':
        tx = x + Ld * math.cos(yaw)
        yL = -(tx - x) * math.sin(yaw) + (laneY(tx) - y) * math.cos(yaw)
    else:
        yL = a + b * (Ld - xRef) + c * (Ld - xRef) ** 2
    om = max(-maxOm, min(maxOm, v * 2 * yL / (Ld ** 2 + yL ** 2)))
    # truth: lateral error, wheels to the nearest crop plant (within 15 cm along the wheel)
    e = y - laneY(x)
    errs.append(e)
    if abs(e) > SP / 2:
        wrong += 1
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    for wi, (wx, wy) in enumerate(WHEELS):
        px, py = x + cy_ * wx - sy_ * wy, y + sy_ * wx + cy_ * wy
        dxs, dys = CROPS[:, 0] - px, CROPS[:, 1] - py
        m_ = np.abs(dxs * cy_ + dys * sy_) < 0.15
        if m_.any():
            lat = float(np.min(np.abs(-dxs[m_] * sy_ + dys[m_] * cy_)))
            wheelMin[wi] = min(wheelMin[wi], lat)
            if lat < CONTACT:
                wheelHit[wi] += 1
    if TRACE and TRACE[0] < x < TRACE[1] and frames % 10 == 0:
        print('x %5.2f true err %+5.1f | %s' % (x, 100 * e, T.status[:70]))
    yaw += om * dt
    x += v * dt * math.cos(yaw)
    y += v * dt * math.sin(yaw)
errs = np.abs(errs) if errs else np.zeros(1)
print(f"ROWSFOUND {getattr(T, 'laneRowsFound', -1)} of {R} | " + ('FAILED ' if failed else '')
      + f"seed {seed}: {x:5.1f} m, losses {losses}, frames off lane (>{50 * SP:.1f}cm) {wrong}, "
      f"err mean {100 * errs.mean():.1f} p95 {100 * np.percentile(errs, 95):.1f} max {100 * errs.max():.1f} cm", lostAt[:3])
print('WHEELS min %s cm, contact frames %s of %d' % ([round(100 * w, 1) for w in wheelMin], wheelHit, frames))
