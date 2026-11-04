#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Crop row tracking on the ground plane.
#
# Plant centers detected in the image are projected onto the ground and kept, per
# row, in the odom frame (fixed in the field). Each row line is fitted on its last
# plants instead of on the current image only, and every estimate is checked in
# metres: a crop row cannot move sideways by more than a few centimetres between
# two images, and the robot is never further than half a furrow from the lane.
import math
import numpy as np


class GroundProjector:
    """Pinhole camera looking at a flat ground, robot frame: x forward, y left, z up."""

    def __init__(self, forward, lateral, height, tilt, yaw=0.0, roll=0.0):
        self.c = np.array([forward, lateral, height], dtype=float)
        # tilt < 0: pitched down (camera_tilt parameter); pitch > 0 is nose down in ROS.
        # yaw > 0: camera turned left, roll > 0: right side down (ROS rpy, applied roll first)
        p = -tilt
        Ry = np.array([[math.cos(p), 0.0, math.sin(p)],
                       [0.0, 1.0, 0.0],
                       [-math.sin(p), 0.0, math.cos(p)]])
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                       [math.sin(yaw), math.cos(yaw), 0.0],
                       [0.0, 0.0, 1.0]])
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, math.cos(roll), -math.sin(roll)],
                       [0.0, math.sin(roll), math.cos(roll)]])
        self.R = Rz @ Ry @ Rx
        self.K = None

    def setIntrinsics(self, fx, fy, cx, cy):
        self.K = (fx, fy, cx, cy)

    def pixelToGround(self, u, v, maxRange=15.0):
        """pixels (u right, v down) -> ground points (x, y) in the robot frame, NaN above the horizon"""
        fx, fy, cx, cy = self.K
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        # optical (x right, y down, z forward) -> camera body (x forward, y left, z up)
        d = np.vstack([np.ones_like(u), -(u - cx) / fx, -(v - cy) / fy])
        d = self.R @ d
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(d[2] < -1e-6, -self.c[2] / d[2], np.nan)
        x, y = self.c[0] + s * d[0], self.c[1] + s * d[1]
        far = np.hypot(x - self.c[0], y - self.c[1]) > maxRange
        x[far], y[far] = np.nan, np.nan
        return x, y

    def groundToPixel(self, x, y):
        fx, fy, cx, cy = self.K
        p = np.vstack([np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                       np.zeros(np.size(x))]) - self.c[:, None]
        b = self.R.T @ p
        return cx - fx * b[1] / b[0], cy - fy * b[2] / b[0]


def toOdom(pose, x, y):
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return px + c * x - s * y, py + s * x + c * y


def toRobot(pose, x, y):
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = x - px, y - py
    return c * dx + s * dy, -s * dx + c * dy


def fitLine(x, y, w):
    """weighted least squares y = a + b*x (rows run along the robot x axis), refitted once
    without the outliers (leaves sticking out, a weed next to the row)"""
    def lsq(m):
        A = np.c_[np.ones(np.count_nonzero(m)), x[m]] * np.sqrt(w[m])[:, None]
        return np.linalg.lstsq(A, y[m] * np.sqrt(w[m]), rcond=None)[0]
    m = np.ones(len(x), dtype=bool)
    a, b = lsq(m)
    if len(x) >= 6:
        r = np.abs(y - (a + b * x))
        keep = r <= max(2.5 * 1.4826 * np.median(r), 0.02)
        if np.count_nonzero(keep) >= 3 and not keep.all():
            a, b = lsq(keep)
    return a, b


class RowTracker:

    def __init__(self, projector, params):
        self.projector = projector
        self.p = params
        self.forgetLane()
        self.reset()

    def forgetLane(self):
        """new row: forget the lane found before"""
        self.lane = None
        self.initialized = False
        self.candidates = []        # lanes found in the last images, before confirmation

    def reset(self):
        # lane of this row (row offsets, furrow, last center in odom), kept across a recovery
        # so that the rows found again are matched to the same lane
        if getattr(self, 'initialized', False):
            self.lane = (list(self.offsets), self.furrowWidth, self.centerOdom)
        self.initialized = False
        self.rows = []              # per row: {'pts': N x 4 [x_odom, y_odom, weight, odometer seen]}
        self.offsets = []           # lateral position of each row w.r.t. the lane center
        self.furrowWidth = None
        self.odometer = 0.0
        self.lastPose = None
        self.center = None          # (a, b): lane center, y = a at xRef with slope b, robot frame
        self.centerOdom = None      # same line in the odom frame: (x, y, direction)
        self.badFrames = 0
        self.status = 'not initialized'
        self.noPlantAhead = 0      # images in a row without row plants in the far half of the view
        self.plantAhead = 0        # images in a row with some
        self.noPlantSeen = 0       # images in a row without any row plant in the whole view
        self.lastAheadOdometer = 0.0   # odometer when row plants were last seen ahead

    # ------------------------------------------------------------------ geometry
    def viewRange(self, imgHeight, topOffset):
        """ground x of the nearest and farthest tracked image rows, along the image center"""
        cx = self.projector.K[2]
        x, _ = self.projector.pixelToGround([cx, cx], [imgHeight - 1, topOffset])
        return x[0], x[1]

    def lineToImage(self, a, b, xNear, xFar):
        """robot-frame ground line -> image line x_pix = m*y_pix + c, as [c, m] (CropRows layout)"""
        u, v = self.projector.groundToPixel([xNear, xFar], [a + b * xNear, a + b * xFar])
        m = (u[1] - u[0]) / (v[1] - v[0])
        return [u[0] - m * v[0], m]

    def _advance(self, pose):
        if self.lastPose is not None:
            self.odometer += math.hypot(pose[0] - self.lastPose[0], pose[1] - self.lastPose[1])
        self.lastPose = pose

    def _project(self, pixels, pose):
        if len(pixels) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
        pixels = np.asarray(pixels, dtype=float)
        x, y = self.projector.pixelToGround(pixels[:, 0], pixels[:, 1])
        ok = np.isfinite(x)
        robot = np.c_[x[ok], y[ok]]
        ox, oy = toOdom(pose, robot[:, 0], robot[:, 1])
        return robot, np.c_[ox, oy]

    def _rowLine(self, k, pose, xRef):
        """robot-frame line y = a + b*x of row k, from the lane center when its history is too short"""
        if len(self.rows[k]) >= self.p['minPoints']:
            return self._fitRow(self.rows[k], pose)
        b = self.center[1]
        return self.center[0] + self.offsets[k] - b * xRef, b

    def _fitRow(self, pts, pose):
        x, y = toRobot(pose, pts[:, 0], pts[:, 1])
        return fitLine(x, y, np.minimum(pts[:, 2], self.p['maxPointWeight']))

    # ------------------------------------------------------------------ initialization
    def initialize(self, pixels, pose, xRef, widths=None):
        """find the rows on the ground from all plant points (u, v) of one image:
        orientation that lines the plants up best, row spacing from the periodicity of the
        lateral histogram, one row per peak, then the evenly spaced rows around the robot.
        widths: vegetation width (pixels) at each point, wide rows outweigh small weeds.
        True once the same lane was found in initConfirmFrames images in a row."""
        if not self._findLane(pixels, pose, xRef, widths):
            self.candidates = []
            return False
        # confirmation: same rows (odom), same furrow, in consecutive images
        lane = (self.centerOdom, len(self.rows), self.furrowWidth)
        if self.candidates:
            (ox, oy, direction), n, furrow = self.candidates[-1]
            # previous candidate center seen from the current pose, at xRef
            x, y = toRobot(pose, ox, oy)
            previous = y + math.tan(direction - pose[2]) * (xRef - x)
            if not (n == lane[1] and abs(furrow - lane[2]) <= 0.1 * furrow
                    and abs(previous - self.center[0]) <= self.p['maxLateralJump']):
                self.candidates = []
        self.candidates.append(lane)
        need = self.p.get('initConfirmFrames', 1)
        if len(self.candidates) < need:
            self.initialized = False
            self.status = 'init: lane found %d/%d times' % (len(self.candidates), need)
            return False
        self.candidates = []
        self.initialized = True
        return True

    def _findLane(self, pixels, pose, xRef, widths):
        self.reset()
        self._advance(pose)
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        robot, odom = self._project(pixels, pose)
        if len(robot) < self.p['minRows'] * self.p['minPoints']:
            return self._fail('%d plants on the ground' % len(robot))
        weight = np.ones(len(robot))
        if widths is not None and len(widths) == len(pixels):
            # ground width of each vegetation run: a weed is a few cm, a crop row much wider
            w = np.asarray(widths, dtype=float)
            ok = np.isfinite(self.projector.pixelToGround(pixels[:, 0], pixels[:, 1])[0])
            xl, yl = self.projector.pixelToGround(pixels[ok, 0] - w[ok] / 2, pixels[ok, 1])
            xr, yr = self.projector.pixelToGround(pixels[ok, 0] + w[ok] / 2, pixels[ok, 1])
            weight = np.nan_to_num(np.clip(np.hypot(xr - xl, yr - yl), 0.01, 0.3), nan=0.01)

        binSize = 0.02
        kernel = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
        best = None
        for heading in np.radians(np.arange(-self.p['maxHeadingError'], self.p['maxHeadingError'] + 0.5, 1.0)):
            lat = robot[:, 1] - math.tan(heading) * (robot[:, 0] - xRef)
            lo = np.floor(lat.min() / binSize) * binSize - 5 * binSize
            hist, edges = np.histogram(lat, bins=np.arange(lo, lat.max() + 6 * binSize, binSize), weights=weight)
            hist = np.convolve(hist, kernel, mode='same')
            score = np.sum(hist ** 2)
            if best is None or score > best[0]:
                best = (score, heading, hist, edges)
        _, heading, hist, edges = best
        lat = robot[:, 1] - math.tan(heading) * (robot[:, 0] - xRef)
        centers = (edges[:-1] + edges[1:]) / 2

        # furrow: given, or the period of the lateral histogram (autocorrelation), which
        # does not depend on a leaf cluster giving two close peaks
        if self.p['furrowWidth'] > 0:
            furrow, ref = self.p['furrowWidth'], 'furrow_width parameter'
        else:
            h0 = hist - hist.mean()
            ac = np.correlate(h0, h0, mode='full')[len(h0) - 1:]
            lags = np.arange(len(ac)) * binSize
            band = (lags >= self.p['minFurrowWidth']) & (lags <= self.p.get('maxFurrowWidth', 1.2))
            if not band.any() or ac[band].max() <= 0:
                if self.p['minRows'] > 1:
                    return self._fail('no periodic rows')
                return self._fail('set furrow_width when following a single row')
            i = np.flatnonzero(band)[np.argmax(ac[band])]
            # parabolic refinement of the period
            if 0 < i < len(ac) - 1 and (ac[i - 1] - 2 * ac[i] + ac[i + 1]) != 0:
                i = i + 0.5 * (ac[i - 1] - ac[i + 1]) / (ac[i - 1] - 2 * ac[i] + ac[i + 1])
            furrow, ref = float(i * binSize), 'measured furrow'
        if furrow < self.p['minFurrowWidth']:
            return self._fail('furrow %.2f m < %.2f m' % (furrow, self.p['minFurrowWidth']))

        # peaks: local maxima with enough plants, strongest first, at least 0.6 furrow apart
        peaks = []
        for i in np.argsort(-hist):
            if hist[i] <= 0 or (i > 0 and hist[i - 1] > hist[i]) or (i + 1 < len(hist) and hist[i + 1] > hist[i]):
                continue
            if any(abs(centers[i] - q) < 0.6 * furrow for q in peaks):
                continue
            if np.count_nonzero(np.abs(lat - centers[i]) < 0.25 * furrow) >= self.p['minPoints'] \
                    and hist[i] >= 0.25 * hist.max():
                peaks.append(centers[i])
        peaks = sorted(peaks)
        if len(peaks) < self.p['minRows']:
            return self._fail('%d row(s) found, %d needed' % (len(peaks), self.p['minRows']))

        # rows of one bed: evenly spaced chain around the robot (neighbour beds are further apart)
        tol = self.p['furrowTolerance'] * furrow
        k0 = int(np.argmin(np.abs(np.array(peaks) - self.p['targetOffset'])))
        lo = hi = k0
        while lo > 0 and abs(peaks[lo] - peaks[lo - 1] - furrow) <= tol:
            lo -= 1
        while hi + 1 < len(peaks) and abs(peaks[hi + 1] - peaks[hi] - furrow) <= tol:
            hi += 1
        chain = peaks[lo:hi + 1]
        if len(chain) < self.p['minRows']:
            return self._fail('row spacing %s m does not match the %s %.2f m'
                              % (np.round(np.diff(peaks), 2).tolist(), ref, furrow))
        laneRows = self.p.get('laneRows', 0)
        if laneRows > 0 and self.lane is None:
            # the lane is made of laneRows rows: the ones around the robot's target
            if len(chain) < laneRows:
                return self._fail('%d row(s) of the %d of the lane visible' % (len(chain), laneRows))
            starts = range(len(chain) - laneRows + 1)
            k = min(starts, key=lambda j: abs(np.mean(chain[j:j + laneRows]) - self.p['targetOffset']))
            chain = chain[k:k + laneRows]

        # each row: its plants, line fitted on them
        gate = min(0.4 * furrow, max(self.p['gateRatio'] * furrow, self.p['minGate']))
        fits, rows = [], []
        for c in chain:
            near = np.abs(lat - c) <= gate
            if np.count_nonzero(near) < self.p['minPoints']:
                return self._fail('row at %+.2f m has too few plants' % c)
            a, b = fitLine(robot[near, 0], robot[near, 1], weight[near])
            fits.append((a + b * xRef, b))
            rows.append(np.c_[odom[near], np.ones(np.count_nonzero(near)),
                              np.full(np.count_nonzero(near), self.odometer)])
        lat0 = np.array([f[0] for f in fits])
        if self.lane is not None:
            # same row as before the loss: place the rows found into the known lane
            match = self._matchLane(lat0, pose, xRef)
            if match is None:
                return self._fail('rows found do not match the lane followed before')
            keep, offsets, furrow = match
            fits = [fits[j] for j in keep]
            rows = [rows[j] for j in keep]
            lat0 = lat0[keep]
        headings = np.degrees(np.arctan([f[1] for f in fits]))
        if np.max(np.abs(headings - np.median(headings))) > self.p['maxRowAngleSpread']:
            return self._fail('rows not parallel (%s deg)' % np.round(headings, 1).tolist())
        if self.lane is None:
            center = float(np.mean(lat0))
            offsets = list(lat0 - center)
        else:
            center = float(np.mean(lat0 - np.array(offsets)))
        slope = float(np.mean([f[1] for f in fits]))
        offset = self.robotOffset(center, slope, xRef)
        if abs(offset) > furrow / 2 + tol:
            return self._fail('robot %.2f m from the lane center, more than half a furrow' % offset)

        self.furrowWidth = furrow
        self.rows = rows
        self.offsets = list(offsets)
        self._setCenter(center, slope, pose, xRef)
        self.badFrames = 0
        self.status = 'tracking %d rows, furrow %.2f m (%s)' % (len(rows), furrow,
                                                              'set' if self.p['furrowWidth'] > 0 else 'measured')
        return True

    def _matchLane(self, lat, pose, xRef):
        """rows found (lateral at xRef, sorted) -> (indices kept, their lane offsets, furrow),
        aligned on the known lane so that the center is the closest to the previous one"""
        laneOffsets, furrow, centerOdom = self.lane
        self.centerOdom = centerOdom
        predicted = self._predictCenter(pose, xRef)[0]
        pattern = np.array(laneOffsets)
        best = None
        for shift in range(-len(lat) + 1, len(pattern)):
            pairs = [(j, j + shift) for j in range(len(lat)) if 0 <= j + shift < len(pattern)]
            if len(pairs) < self.p['minRows']:
                continue
            center = np.mean([lat[j] - pattern[i] for j, i in pairs])
            err = abs(center - predicted)
            if best is None or err < best[0] - 1e-6 or (abs(err - best[0]) < 1e-6 and len(pairs) > len(best[1])):
                best = (err, pairs)
        if best is None or best[0] > furrow / 2:
            return None
        return [j for j, _ in best[1]], [pattern[i] for _, i in best[1]], furrow

    def _fail(self, reason):
        self.initialized = False
        self.status = 'init failed: ' + reason
        return False

    # ------------------------------------------------------------------ tracking
    def update(self, rowPixels, pose, xRef, xNear, xAhead=None, exiting=False):
        """returns True while the lane is tracked; self.status tells why not.
        xAhead: plants of the rows beyond it count as 'row continues' (row end detection).
        exiting: past the last plants, keep the lane from the history / odometry, never lost"""
        self._advance(pose)
        gate = max(self.p['gateRatio'] * self.furrowWidth, self.p['minGate'])
        candidates = []
        ahead = 0
        # farthest row plant of this image (odom), seen from here: the closer, the more accurate
        self.farthestOdom, farthestX = None, -np.inf
        for k, pixels in enumerate(rowPixels):
            robot, odom = self._project(pixels, pose)
            a, b = self._rowLine(k, pose, xRef)
            # plants far from the row's own line belong to a neighbour row or are weeds. The
            # gate widens with the distance: a small heading error of the line moves it by
            # several cm 6 m ahead, the far plants must still be accepted (or the line gets
            # fitted on the near plants only, its heading gets worse, and so on)
            gateAt = np.minimum(gate + np.maximum(robot[:, 0] - xNear, 0.0) * math.tan(math.radians(2.0)),
                                max(gate, 0.45 * self.furrowWidth))
            near = np.abs(robot[:, 1] - (a + b * robot[:, 0])) <= gateAt
            candidates.append(odom[near])
            if np.any(near) and np.max(robot[near, 0]) > farthestX:
                i = int(np.argmax(np.where(near, robot[:, 0], -np.inf)))
                farthestX, self.farthestOdom = robot[i, 0], odom[i]
            if xAhead is not None:
                # plants assigned to this row in the image, before the gate: the row goes on
                ahead += int(np.count_nonzero(robot[:, 0] > xAhead))
        seen = sum(len(c) for c in candidates)
        self.noPlantSeen = 0 if seen >= self.p.get('minPointsAhead', 2) else self.noPlantSeen + 1
        if ahead >= self.p.get('minPointsAhead', 2):
            self.noPlantAhead, self.plantAhead = 0, self.plantAhead + 1
            self.lastAheadOdometer = self.odometer
        else:
            self.noPlantAhead, self.plantAhead = self.noPlantAhead + 1, 0

        newRows = [self._merge(self.rows[k], candidates[k]) for k in range(len(self.rows))]
        newRows = [self._prune(r, pose, xNear) for r in newRows]
        valid = [k for k, r in enumerate(newRows)
                 if len(r) >= self.p['minPoints']
                 and self.odometer - np.max(r[:, 3]) <= self.p['maxRowGap']]
        # rows with enough plants seen in the last 0.5 m (not only old ones about to be pruned)
        self.freshRows = sum(1 for r in newRows
                             if np.count_nonzero(r[:, 3] >= self.odometer - 0.5) >= self.p['minPoints'])
        if len(valid) < self.p['minRows']:
            self.rows = newRows
            if exiting:
                return self._deadReckoning(pose, xRef, '%d row(s) left' % len(valid))
            self.status = 'lost: %d row(s) visible, %d needed' % (len(valid), self.p['minRows'])
            return False

        fits = [self._fitRow(newRows[k], pose) for k in valid]
        # lane center from each visible row, shifted by its offset measured at initialization
        a = float(np.mean([f[0] + f[1] * xRef - self.offsets[k] for f, k in zip(fits, valid)]))
        b = float(np.mean([f[1] for f in fits]))
        lateral, heading = a, math.degrees(math.atan(b))
        # robot offset from where the controller drives it (camera above the lane center),
        # at the robot (x = 0): independent of the heading, unlike the value at xRef
        offset = self.robotOffset(a, b, xRef)

        # the field does not move: compare with the previous estimate carried over in odom,
        # so that the robot turning (which moves the point at xRef) is not taken for a jump
        predicted = self._predictCenter(pose, xRef)
        reason = None
        if abs(lateral - predicted[0]) > self.p['maxLateralJump']:
            reason = 'lateral jump %.0f cm -> %.0f cm' % (100 * predicted[0], 100 * lateral)
        elif abs(offset) > self.maxLateralError():
            reason = 'lateral error %.0f cm > %.0f cm' % (100 * offset, 100 * self.maxLateralError())
        elif abs(heading) > self.p['maxHeadingError']:
            reason = 'heading error %.1f deg > %.1f deg' % (heading, self.p['maxHeadingError'])
        if reason is not None and exiting:
            self.rows = newRows
            return self._deadReckoning(pose, xRef, reason)
        if reason is not None:
            # implausible: keep the history as it was, so one bad image cannot pull the lines
            self.badFrames += 1
            self.center = predicted
            self.status = 'rejected (%d/%d): %s' % (self.badFrames, self.p['maxBadFrames'], reason)
            if self.badFrames > self.p['maxBadFrames']:
                self.status = 'lost: ' + reason
                return False
            return True

        self.rows = newRows
        self.badFrames = 0
        self._setCenter(lateral, b, pose, xRef)
        self.status = 'tracking %d/%d rows, lateral %+.0f cm, heading %+.1f deg' % (
            len(valid), len(self.rows), 100 * offset, heading)
        return True

    def _deadReckoning(self, pose, xRef, reason):
        """keep the last lane center line (odom): straight on with the odometry"""
        self.center = self._predictCenter(pose, xRef)
        self.status = 'exit, lane from odometry (%s)' % reason
        return True

    def rowEnd(self):
        """no row plant ahead in rowEndFrames images and over rowEndDistance metres"""
        return self.noPlantAhead >= self.p.get('rowEndFrames', 5) and \
            self.odometer - self.lastAheadOdometer >= self.p.get('rowEndDistance', 0.5)

    def blind(self):
        """no row plant in the whole view for rowEndFrames images"""
        return self.noPlantSeen >= self.p.get('rowEndFrames', 5)

    def rowContinues(self):
        """plants ahead again in rowEndFrames images and enough rows to track (after a gap)"""
        return self.plantAhead >= self.p.get('rowEndFrames', 5) and \
            getattr(self, 'freshRows', 0) >= self.p['minRows']

    def alongLane(self, x, y):
        """distance along the lane center line (odom), from its reference point"""
        ox, oy, direction = self.centerOdom
        return (x - ox) * math.cos(direction) + (y - oy) * math.sin(direction)

    def lastPlantAlong(self):
        """position along the lane of the farthest plant of the tracked rows"""
        pts = [r[:, :2] for r in self.rows if len(r)]
        if not pts:
            return None
        pts = np.vstack(pts)
        return float(np.max(self.alongLane(pts[:, 0], pts[:, 1])))

    def _setCenter(self, a, b, pose, xRef):
        self.center = (a, b)
        ox, oy = toOdom(pose, xRef, a)
        self.centerOdom = (ox, oy, pose[2] + math.atan(b))

    def _predictCenter(self, pose, xRef):
        """previous lane center line, seen from the current pose: (y at xRef, slope)"""
        ox, oy, direction = self.centerOdom
        x, y = toRobot(pose, ox, oy)
        b = math.tan(direction - pose[2])
        return y + b * (xRef - x), b

    def robotOffset(self, a, b, xRef):
        """lateral offset of the robot from its target line, lane center (a at xRef, slope b)"""
        return -((a - b * xRef) - self.p['targetOffset'])

    def maxLateralError(self):
        if self.p['maxLateralError'] > 0:
            return self.p['maxLateralError']
        return self.furrowWidth / 2

    def rowLines(self, pose, xRef):
        """robot-frame lines (a at xRef, b) of all rows, and of the lane center"""
        rows = []
        for k in range(len(self.rows)):
            a, b = self._rowLine(k, pose, xRef)
            rows.append((a + b * xRef, b))
        return rows, self.center

    def _merge(self, row, odom):
        """add plant centers; one seen again (within merge radius) is averaged, not duplicated"""
        row = row.copy()
        r2 = self.p['mergeRadius'] ** 2
        for pt in odom:
            if len(row):
                d2 = np.sum((row[:, :2] - pt) ** 2, axis=1)
                i = int(np.argmin(d2))
                if d2[i] <= r2:
                    w = row[i, 2]
                    row[i, :2] = (row[i, :2] * w + pt) / (w + 1)
                    row[i, 2] = w + 1
                    row[i, 3] = self.odometer
                    continue
            row = np.vstack([row, [pt[0], pt[1], 1.0, self.odometer]])
        return row

    def _prune(self, row, pose, xNear):
        """keep the last historySize plants, at most historyLength metres behind the view"""
        if len(row) == 0:
            return row
        x, _ = toRobot(pose, row[:, 0], row[:, 1])
        row = row[x >= xNear - self.p['historyLength']]
        if len(row) > self.p['historySize']:
            row = row[np.argsort(row[:, 3], kind='stable')[-self.p['historySize']:]]
        return row
