#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Lane check from the vision box (the phenotyping cameras looking down at the plants).
#
# The navigation camera sees the rows from 1 to 3.4 m ahead and cannot tell a crop from a
# weed: a lane that slid by a furrow (or onto a weed line in a wheel path) can look exactly
# like the right one. The vision box sees the plants around the tools, under the robot, and
# tells crops from weeds.
#
# The crop rows it sees are compared with the lane, and with the same lane shifted by one or
# two furrows to each side: crops on the lane rows count for a placement, a lane row seen
# empty and a crop row on the place one furrow outside the lane (where a bed has its wheel
# path) count against it. When a shifted lane explains the crops clearly better than the
# tracked one, the tracked lane slid, and the shift says where the right one is. In fields of
# uniform rows (no wheel paths) every placement scores the same: no decision, no false alarm.
#
# Nothing depends on the crop, the number of rows or where the tools are: the lane comes
# from the row tracker, the part each camera covers from the plants it reports.
import numpy as np


class VisionBoxCheck:

    def __init__(self, length=2.0, minCrops=3, mergeRadius=0.04):
        self.length = length            # m driven over which the plants are kept (lettuce: 0.3 m apart)
        self.minCrops = minCrops        # plants making a crop row
        self.mergeRadius = mergeRadius  # m, the same plant re-published is counted once
        self.reset()

    def reset(self):
        # per plant: x, y (odom), crop votes, weed votes, distance when last seen, camera,
        # lateral position from the lane centre when seen (mean over the times it was seen)
        self.points = np.zeros((0, 7))
        self.lastOffset = None          # m, crop rows' mean lateral offset from the lane rows
        self.shift = 0.0                # m, lane shift that explains the crops best (last check)
        self.status = 'no vision box data'
        self.lastOdometer, self.driven = None, 0.0

    def _distance(self, odometer):
        """own distance driven: the caller's odometer may restart from 0 (tracker reset)"""
        if self.lastOdometer is not None:
            self.driven += max(odometer - self.lastOdometer, 0.0)
        self.lastOdometer = odometer
        return self.driven

    def add(self, odomXY, rel, isCrop, odometer, camera=0):
        """plants of one vision box message from one camera (topic index): odom positions and
        lateral positions from the lane centre now (the plants are close to the robot, where
        the lane is measured; later the lane follows the field, the position stays valid)"""
        driven = self._distance(odometer)
        self.points = self.points[self.points[:, 4] >= driven - self.length]
        odomXY = np.asarray(odomXY, dtype=float).reshape(-1, 2)
        rel = np.asarray(rel, dtype=float).reshape(-1)
        isCrop = np.asarray(isCrop, dtype=bool).reshape(-1)
        r2 = self.mergeRadius ** 2
        new = []
        for (x, y), r, crop in zip(odomXY, rel, isCrop):
            if len(self.points):
                d2 = (self.points[:, 0] - x) ** 2 + (self.points[:, 1] - y) ** 2
                i = int(np.argmin(d2))
                if d2[i] <= r2:
                    # seen again: one plant, one more vote
                    votes = self.points[i, 2] + self.points[i, 3]
                    self.points[i, 6] = (self.points[i, 6] * votes + r) / (votes + 1)
                    self.points[i, 2 if crop else 3] += 1
                    self.points[i, 4] = driven
                    continue
            new.append((x, y, 1.0 if crop else 0.0, 0.0 if crop else 1.0, driven, camera, r))
        if new:
            self.points = np.vstack([self.points, new])

    def check(self, offsets, furrow):
        """offsets: lane rows from the lane centre. Returns None (no decision), True (lane
        confirmed) or a reason (str) when a shifted lane explains the crops clearly better;
        self.shift is then that shift (m, + = left)."""
        self.shift = 0.0
        if len(self.points) == 0:
            self.status = 'no vision box data'
            return None
        rel = self.points[:, 6]                             # lateral from the lane centre
        crop = self.points[:, 2] > self.points[:, 3]
        rows = self._cropRows(rel[crop], rel[~crop], furrow)
        if len(rows) < 2 or np.count_nonzero(crop) < 4 * self.minCrops:
            # not enough seen yet (start of a row, a few plants entering the view)
            self.status = 'vision box: too few crops seen (%d plants, %d rows)' % (np.count_nonzero(crop), len(rows))
            return None
        # what each camera covers: from its own plants (a gap between two cameras is not seen)
        spans = [(rel[self.points[:, 5] == c].min(), rel[self.points[:, 5] == c].max())
                 for c in np.unique(self.points[:, 5])]

        def covered(y):
            return any(lo + 0.25 * furrow <= y <= hi - 0.25 * furrow for lo, hi in spans)

        lane = np.sort(np.asarray(offsets, dtype=float))

        def faults(s):
            """for the lane shifted by s: lane rows seen empty, crop rows on a place one furrow
            outside the lane. Only contradictions count: a placement gains nothing from
            seeing more rows (the cameras cover one side of the lane more than the other)"""
            n, seen = 0, 0
            for y in lane + s:
                if np.min(np.abs(rows - y)) <= 0.5 * furrow:
                    seen += 1
                elif covered(y):
                    n, seen = n + 1, seen + 1
            for y in (lane[0] + s - furrow, lane[-1] + s + furrow):
                if np.min(np.abs(rows - y)) <= 0.35 * furrow:
                    n += 1
            return n, seen

        shifts = furrow * np.array([0, -1, 1, -2, 2])
        results = [faults(s) for s in shifts]
        base, seen = results[0]
        if seen == 0:
            self.status = 'vision box: lane rows not covered'
            return None
        matched = [y for y in lane if np.min(np.abs(rows - y)) <= 0.5 * furrow]
        self.lastOffset = float(np.median([rows[np.argmin(np.abs(rows - y))] - y for y in matched])) \
            if matched else None
        f = np.array([n for n, _ in results])
        best = int(np.argmin(f))
        # a clear winner: two contradictions fewer than the tracked lane, and the only one
        if best != 0 and f[best] <= base - 2 and np.count_nonzero(f == f[best]) == 1:
            self.shift = float(shifts[best])
            self.status = ('vision box: the crop rows fit the lane shifted %+.0f cm (%d contradictions vs %d): '
                           'the lane slid' % (100 * self.shift, f[best], base))
            return self.status
        self.status = 'vision box: lane confirmed (%d crop rows, offset %+.0f cm)' % (
            len(rows), 100 * (self.lastOffset or 0.0))
        return True

    def _cropRows(self, crops, weeds, furrow):
        """lateral positions of the crop rows: peaks of the crop plants, at least 0.6 furrow apart,
        each with enough plants and not a weed line with a few plants taken for crops"""
        if len(crops) < self.minCrops:
            return np.zeros(0)
        b = 0.02
        lo = np.floor(crops.min() / b) * b - 3 * b
        edges = np.arange(lo, crops.max() + 4 * b, b)
        h, _ = np.histogram(crops, bins=edges)
        h = np.convolve(h, np.exp(-0.5 * (np.arange(-3, 4) / 1.2) ** 2), mode='same')
        centers = (edges[:-1] + edges[1:]) / 2
        rows = []
        for i in np.argsort(-h):
            if h[i] <= 0:
                break
            if (i > 0 and h[i - 1] > h[i]) or (i + 1 < len(h) and h[i + 1] > h[i]):
                continue
            if any(abs(centers[i] - r) < 0.6 * furrow for r in rows):
                continue
            near = np.abs(crops - centers[i]) <= 0.35 * furrow
            n = np.count_nonzero(near)
            # a weed line has 5-15 % of its plants taken for crops; a crop row through a weed
            # patch keeps a good share of crops
            if n < self.minCrops or n < 0.3 * np.count_nonzero(np.abs(weeds - centers[i]) <= 0.35 * furrow):
                continue
            rows.append(float(np.mean(crops[near])))
        return np.sort(np.asarray(rows))
