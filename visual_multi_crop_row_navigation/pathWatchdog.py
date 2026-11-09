#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Reference path watchdog: an independent check of the visual row following.
#
# The camera alone cannot tell a wrong lane from the right one when both look the same
# (e.g. weeds in the middle of a wheel path two furrows wide: four evenly spaced "rows").
# The planned coverage path (or a GNSS track) does not depend on the image: the robot
# must stay within maxDeviation of it. The path covers the whole field, so the watchdog
# locks on the segment closest to the robot when the row starts and then only searches a
# window along the path from there: the neighbour pass, 1.5 m away, never counts.
import math
import numpy as np


class PathWatchdog:

    def __init__(self, maxDeviation, window=5.0):
        self.maxDeviation = maxDeviation
        self.window = window        # m along the path searched ahead of / behind the locked point
        self.path = None            # N x 2, in the path frame
        self.frame = None
        self.reset()

    def reset(self):
        """new row: lock again on the closest segment"""
        self.index = None           # segment the robot was last matched to
        self.deviation = None

    def setPath(self, points, frame):
        """new reference path; the same path again (re-published, or received twice) keeps the
        lock. Returns True when the path changed."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        # consecutive duplicates give zero-length segments
        keep = np.r_[True, np.any(np.abs(np.diff(points, axis=0)) > 1e-6, axis=1)]
        points = points[keep]
        if len(points) < 2:
            self.path = None
            return True
        if self.path is not None and frame == self.frame and points.shape == self.path.shape and \
                np.allclose(points, self.path):
            return False
        self.path, self.frame = points, frame
        self.cumulative = np.r_[0.0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
        self.reset()
        return True

    def _distances(self, first, last, x, y, yaw):
        """distance from (x, y) to segments first..last-1; segments going against the robot
        heading yaw (the neighbour pass of a back-and-forth coverage) are not candidates"""
        a, b = self.path[first:last], self.path[first + 1:last + 1]
        d = b - a
        t = ((x - a[:, 0]) * d[:, 0] + (y - a[:, 1]) * d[:, 1]) / np.sum(d * d, axis=1)
        # before the first pose / past the last one (robot still in the headland when the row
        # starts, or driving to the last plant): sideways distance to the extended path, not
        # the distance along it
        lo = np.where(np.arange(first, last) == 0, -np.inf, 0.0)
        hi = np.where(np.arange(first, last) == len(self.path) - 2, np.inf, 1.0)
        t = np.clip(t, lo, hi)
        dist = np.hypot(a[:, 0] + t * d[:, 0] - x, a[:, 1] + t * d[:, 1] - y)
        if yaw is not None:
            dist = np.where(d[:, 0] * math.cos(yaw) + d[:, 1] * math.sin(yaw) > 0, dist, np.inf)
        return dist

    def check(self, x, y, yaw=None):
        """distance (m) from the robot (x, y, yaw in the path frame) to its part of the path,
        None without path"""
        if self.path is None:
            return None
        segments = len(self.path) - 1
        if self.index is None:
            # start of the row: the closest segment of the whole path
            dist = self._distances(0, segments, x, y, yaw)
            if not np.isfinite(np.min(dist)):
                return None
            self.index = int(np.argmin(dist))
        else:
            # then only the path around the last match: the robot moves along it
            s = self.cumulative[self.index]
            first = int(np.searchsorted(self.cumulative, s - self.window, side='right')) - 1
            last = int(np.searchsorted(self.cumulative, s + self.window, side='left'))
            first, last = max(first, 0), min(max(last, first + 1), segments)
            dist = self._distances(first, last, x, y, yaw)
            self.index = first + int(np.argmin(dist))
            dist = dist[self.index - first:self.index - first + 1]
        self.deviation = float(np.min(dist))
        return self.deviation

    def exceeded(self):
        return self.deviation is not None and self.maxDeviation > 0 and self.deviation > self.maxDeviation
