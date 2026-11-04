 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Fixed imports - add package name
from visual_multi_crop_row_navigation.contours import *
from visual_multi_crop_row_navigation.geometric import *
from visual_multi_crop_row_navigation.featureMatching import *
from visual_multi_crop_row_navigation.movingVariance import *

class imageProc:
    def __init__(self, scannerParams, contourParams, roiParams, trackerParams):
        """# class FeatureExtractor to extract the the line Features: bottom point, angle

        Args:
            windowProp (_type_): _description_
            roiParams (_type_): _description_
            featureParams (_type_): _description_
        """
        self.roiParams = roiParams
        self.contourParams = contourParams
        self.scannerParams = scannerParams
        self.trackerParams = trackerParams
        # rows the image scan must find to start tracking (a lane center needs 2)
        self.minRows = trackerParams.get("minRows", 1)
        self.verbose = True
        # ground-plane row tracker (groundTracker.RowTracker), set by the node;
        # without it the rows are tracked in the image only
        self.groundTracker = None
        # robot pose (x, y, yaw) in the odom frame for the current image
        self.robotPose = None
        # past the last plants of the row: the ground tracker follows the lane by odometry
        self.exiting = False

        self.reset()

    def log(self, *args):
        if self.verbose:
            print(*args)

    def reset(self):
        self.log("**************Reset*************")
        if getattr(self, 'groundTracker', None) is not None:
            self.groundTracker.reset()
        # true when the ground tracker lost the lane (not when the initial scan fails)
        self.trackingLost = False
        self.trackStatus = ''
        self.rowPixels = []
        self.count = 0
        self.pointsInTop = 0
        self.pointsInBottom = 0

        # parameters
        self.bushy = False
        self.CropRows = []
        self.primaryRGBImg = []
        self.cropLaneFound = False
        self.isInitialized = False

        self.cropRowEnd = False
        # smallest gap between the rows found at initialization (merge detection)
        self.initRowSpacing = 0.0
        self.missedFrames = np.zeros(0, dtype=int)
        # a row may be unseen this many frames in a row (~2 s at 30 Hz) before re-scanning
        self.maxMissedFrames = 60
        self.P, self.ang = None, None

        self.trackingBoxLoc = []

        # features
        self.allLineEnd = []
        self.allLineStart = []
        self.mainLine_up = np.array([])
        self.mainLine_down = np.array([])

        # window parameters
        self.trackingWindowOffsTop = 0
        self.trackingWindowOffsBottom = 0
        self.trackingWindowTopScaleRatio = 0.4

        self.imgHeight, self.imgWidth = 720, 1280

        # CORRECTION: Utiliser scanSteps, pas scanWindowWidth
        self.scanFootSteps = np.linspace(
            self.scannerParams["scanStartPoint"],
            self.scannerParams["scanEndPoint"],
            self.scannerParams["scanSteps"])  # Changé ici
        
        self.rowTrackingBoxes = []
        self.updateTrackingBoxes()

        self.numOfCropRows = len(self.rowTrackingBoxes) 

    def findCropLane(self, rgbImg, depthImg, mode='RGB-D', bushy=False):
        """finds Crops Rows in the image based on RGB and depth data
        Args:
            bushy (bool, optional): _description_. Defaults to False.
        """
        self.primaryRGBImg = rgbImg.copy()
        if depthImg is not None:
            self.primaryDepthImg = depthImg.copy()
        else:
            self.primaryDepthImg = None
        prev_h, prev_w = self.imgHeight, self.imgWidth
        self.imgHeight, self.imgWidth, self.imgChannels = rgbImg.shape
        if prev_h != self.imgHeight or prev_w != self.imgWidth:
            self.updateTrackingBoxes()
        # if crop canopy type is busshy like Coriander
        self.bushy = bushy
        
        # check if current image is not Empty or None
        if self.primaryRGBImg is None or len(self.primaryRGBImg) == 0:
            self.log("#[ERR] CR-Scanner - Image is None, Check Topics or the Sensor !")
        else:
            # initial RGB image process to get mask GreenIDx and plant centers
            self.plantWidths2D = None
            self.mask, self.greenIDX, self.plantObjects2D, self.plantCenters2D = self.processRGBImage(self.primaryRGBImg)
            self.numPlantsInScene = len(self.plantCenters2D[0])
        if not self.isInitialized and self.groundTracker is not None:
            self.findCropRowsGround()
        elif not self.isInitialized:
            self.log("#[INF] Find Crop Lane")
            self.trackStatus = 'scan: no crop rows found'
            self.lines2D, self.linesROIs2D = self.findCropRows2D(self.primaryRGBImg)
            # self.lines3D, self.linesROIs3D = self.findCropRows3D(self.mask, self.plantCenters2D, self.primaryDepthImg)
            # merge both predictions to get more robust results!

        return self.cropLaneFound

    def findCropRowsGround(self):
        """initialize the ground tracker with the plants between scanStartPoint and scanEndPoint"""
        tracker = self.groundTracker
        u, v = self.plantCenters2D
        inScan = (u >= self.scannerParams["scanStartPoint"]) & (u <= self.scannerParams["scanEndPoint"])
        xNear, xFar = tracker.viewRange(self.imgHeight, self.trackerParams["topOffset"])
        widths = self.plantWidths2D[inScan] if self.plantWidths2D is not None else None
        self.cropLaneFound = tracker.initialize(np.c_[u[inScan], v[inScan]], self.robotPose, xNear, widths)
        self.trackStatus = tracker.status
        if not self.cropLaneFound:
            self.log('#[INF] Find Crop Lane: ' + tracker.status)
            return
        rows, center = tracker.rowLines(self.robotPose, xNear)
        self.CropRows = np.array([tracker.lineToImage(a - b * xNear, b, xNear, xFar) for a, b in rows])
        self.numOfCropRows = len(self.CropRows)
        self.trackingBoxLoc = np.zeros((self.numOfCropRows, 1))
        self.missedFrames = np.zeros(self.numOfCropRows, dtype=int)
        self.isInitialized = True
        self.log('#[INF] Controller Initialized - ' + tracker.status)

    def findCropRows3D(self, maskImg, plantCenters2D, depthImg):
        if depthImg is None:
            return [], []
        lines = []
        linesROIs = []
        # project the plant centers to 3D
        # cluster points 
        # fit line on each cluster
        # project back 3D points to image
        # return the resutls
        return lines, linesROIs

    def findCropRows2D(self, rgbImg):
        # search for lines at given window locations using LQ-Adjustment
        lines, linesROIs, _ = self.findLinesInImage()
        # if any feature is observed during the scan
        if len(lines) != 0:
            self.log(f"DEBUG: lines found: {len(lines)}")
            self.log(f"DEBUG: lines X positions: {[line[0] for line in lines[:10]]}")
            np.set_printoptions(suppress=True)
            
            # Vérifier qu'il y a assez de lignes pour calculer la moving std
            if len(lines[:, 0]) <= 5:
                self.log('--- Initialisation failed - Not enough lines for analysis ---')
                self.cropLaneFound = False
                return lines, linesROIs
            
            # compute moving standard deviation
            mvSignal = movingStd(lines[:, 0])
            
            # Vérifier que mvSignal n'est pas vide
            if len(mvSignal) == 0:
                self.log('--- Initialisation failed - Moving std returned empty ---')
                self.cropLaneFound = False
                return lines, linesROIs
            
            # find positive an negative peaks of the signal
            peaksPos, peaksNeg = findPicksTroughths(mvSignal, 0.5)
            # locate best lines
            self.CropRows, self.trackingBoxLoc = self.findCropRowsInMVSignal(peaksPos, peaksNeg,
                                                        mvSignal, lines, linesROIs)
            
            # Ajouter cette vérification
            if len(self.CropRows) == 0 or len(self.trackingBoxLoc) == 0:
                self.log('--- Initialisation failed - No valid crop rows found ---')
                self.cropLaneFound = False
                return lines, linesROIs
            
            self.numOfCropRows = len(self.trackingBoxLoc)
            self.lostCropRows = list(np.zeros(self.numOfCropRows))
            # consecutive frames each row was not seen (kept on its previous line)
            self.missedFrames = np.zeros(self.numOfCropRows, dtype=int)
            # if there is any proper line to follow
            if self.numOfCropRows < self.minRows:
                self.log('--- Initialisation failed - %d crop row(s) found, %d needed ---'
                         % (self.numOfCropRows, self.minRows))
                self.cropLaneFound = False
                self.trackStatus = 'scan: %d row(s) found, %d needed' % (self.numOfCropRows, self.minRows)
            elif self.numOfCropRows != 0:
                # set parameters indicating a successfull initialization
                self.isInitialized = True
                self.cropLaneFound = True
                self.initRowSpacing = self.minRowSpacing(self.CropRows)
                self.log('#[INF] Controller Initialized - Crop Rows:',
                        len(self.CropRows),
                        ', Window positions:',
                        self.CropRows[:,0].tolist())
            else:
                self.log(
                    '--- Initialisation failed - No lines detected due to peak detection ---')
                self.cropLaneFound = False
        else:
            self.log(
                '--- Initialisation failed - No lines detected by sweeping window ---')
            self.cropLaneFound = False
        return lines, linesROIs

    def findCropRowsInMVSignal(self, peaksPos, peaksNeg, mvSignal, lines, linesROIs):
        # if multiple lines
        if len(peaksPos) != 0 and len(peaksNeg) != 0 and len(mvSignal) != 0:
            # best line one each side of the crop row transition
            qualifiedLines = np.zeros((len(peaksPos)+1, 2))
            windowLocations = np.zeros((len(peaksPos)+1, 1))
            # get lines at each side of a positive peak (= crop row
            # transition and select the best)
            try:
                lidx = 0
                # every peak stands for 2 croprows
                for k in range(0, len(peaksPos) + 1):
                    # first peak
                    if k == 0:
                        peaksNegLine = peaksNeg[peaksNeg < peaksPos[k]]
                    else:
                        # second to last-1
                        if k < len(peaksPos):
                            linestmp = peaksNeg[peaksNeg < peaksPos[k]]
                            peaksNegLine = linestmp[linestmp > peaksPos[k-1]]
                        # last peak
                        else:
                            peaksNegLine = peaksNeg[peaksNeg > peaksPos[k-1]]

                    if len(peaksNegLine) != 0:
                        # best line (measured by variance)
                        bestLine = np.where(
                            mvSignal[peaksNegLine] == np.min(mvSignal[peaksNegLine]))[0][0]
                        # parameters of that line
                        qualifiedLines[lidx, :] = lines[peaksNegLine[bestLine]]
                        # location of the window which led to this line
                        if linesROIs[peaksNegLine[bestLine]] != [0]:
                            windowLocations[lidx,
                                                :] = linesROIs[peaksNegLine[bestLine]]
                            lidx += 1
            except:

                # fallback: just take all negative peaks
                qualifiedLines = lines[peaksNeg]
                windowLocations = np.mean(linesROIs[peaksNeg])

        # if there are no positive peaks but negative ones: no crop
        # row transition -> there might be just one line
        elif len(peaksPos) == 0 and len(peaksNeg) != 0 and len(mvSignal) != 0:
            peaksNegLine = peaksNeg
            bestLine = np.where(mvSignal[peaksNegLine] == np.min(mvSignal[peaksNegLine]))[0][0]

            qualifiedLines = np.zeros((1, 2))
            windowLocations = np.zeros((1, 1))

            qualifiedLines[0, :] = lines[peaksNegLine[bestLine]]
            windowLocations[0] = linesROIs[peaksNegLine[bestLine]]
        else:
            qualifiedLines = []
            windowLocations = []
        
        # Ajouter cette vérification
        if len(qualifiedLines) == 0 or qualifiedLines.size == 0:
            return np.array([]), np.array([])
        
        # Vérifier que qualifiedLines a au moins 1 dimension avant d'utiliser axis=1
        if qualifiedLines.ndim == 1:
            qualifiedLines = qualifiedLines.reshape(1, -1)
        
        qualifiedLines = qualifiedLines[~np.all(qualifiedLines == 0, axis=1)]
        
        if len(windowLocations) == 0 or windowLocations.size == 0:
            return qualifiedLines, np.array([])
        
        if windowLocations.ndim == 1:
            windowLocations = windowLocations.reshape(1, -1)
        
        windowLocations = windowLocations[~np.all(windowLocations == 0, axis=1)]

        return qualifiedLines, windowLocations

    def findLinesInImage(self):
        """function to get lines at given window locations 
        Args:
            self.scanFootSteps (_type_): _description_
        Returns:
            _type_: lines in image
        """
        # initialization
        lines = np.zeros((len(self.scanFootSteps), 2))
        trackingWindows = np.zeros((len(self.scanFootSteps), 1))
        meanLinesInWindows = np.zeros((len(self.scanFootSteps), 1))
        angleVariance = None
        # counting the points in the top and bottom half
        self.pointsInBottom = 0
        self.pointsInTop = 0
        # while tracking, each plant belongs only to the row whose previous line is
        # closest: rows converge towards the top of the image, so neighbouring boxes
        # overlap there and a shared plant pulls both lines together until they merge
        nearestRow = None
        if self.isInitialized and self.numPlantsInScene > 0:
            rowX = (self.CropRows[:self.numOfCropRows, 1][:, None] * self.plantCenters2D[1][None, :]
                    + self.CropRows[:self.numOfCropRows, 0][:, None])
            nearestRow = np.argmin(np.abs(rowX - self.plantCenters2D[0][None, :]), axis=0)
        self.rowPixels = []
        # for all windows
        for boxIdx in range(0, self.numOfCropRows, 1):
            # define window
            if self.isInitialized:
                # get x values, where the top and bottom of the window intersect
                # with the previous found line
                lineIntersection = lineIntersectWin(self.CropRows[boxIdx, 1],
                                                    self.CropRows[boxIdx, 0],
                                                    self.imgHeight,
                                                    self.trackerParams["topOffset"],
                                                    self.trackerParams["bottomOffset"])
                self.updateTrackingBoxes(boxIdx, lineIntersection)

            plantsInCropRow = []
            for ptIdx in range(self.numPlantsInScene):
                # checking #points in upper/lower half of the image
                self.checkPlantsLocTB(self.plantCenters2D[:, ptIdx])
                
                if nearestRow is not None and nearestRow[ptIdx] != boxIdx:
                    continue
                # if plant center is inside tracking box
                if self.rowTrackingBoxes[boxIdx].contains(Point(self.plantCenters2D[0, ptIdx], self.plantCenters2D[1, ptIdx])):
                    plantsInCropRow.append(self.plantCenters2D[:, ptIdx])
            self.rowPixels.append(plantsInCropRow)

            if len(plantsInCropRow) >= 2:
                # flipped points
                ptsFlip = np.flip(plantsInCropRow, axis=1)
                # get line at scanning window
                xM, xB = getLineRphi(ptsFlip)
                t_i, b_i = lineIntersectImgUpDown(xM, xB, self.imgHeight)
                l_i, r_i = lineIntersectImgSides(xM, xB, self.imgHeight)
                # self.log("row ID:", boxIdx, t_i, b_i, l_i, r_i )
                # if the linefit does not return None and the line-image intersections
                # are within the image bounds
                if xM is not None and b_i >= 0 and b_i <= self.imgWidth:
                    lines[boxIdx, :] = [xB, xM]
                    # store the window location, which generated these line
                    # parameters
                    if self.isInitialized == False:
                        trackingWindows[boxIdx] = self.scanFootSteps[boxIdx]
                    # mean x value of all points used for fitting the line
                    meanLinesInWindows[boxIdx] = np.median(plantsInCropRow, axis=0)[0]

            if self.isInitialized:
                if np.all(lines[boxIdx] == 0):
                    # gap in this row (missing plants): keep its previous line for now
                    self.missedFrames[boxIdx] += 1
                    lines[boxIdx, :] = self.CropRows[boxIdx]
                    meanLinesInWindows[boxIdx] = self.trackingBoxLoc[boxIdx] + self.trackerParams["trackingBoxWidth"]/2
                else:
                    self.missedFrames[boxIdx] = 0

        # if the feature extractor is initalized, adjust the window
        # size with the variance of the line fit
        if self.isInitialized and angleVariance is not None:
            self.trackerParams["trackingBoxWidth"] = max(3*angleVariance, self.trackerParams["trackingBoxWidth"])

        # delete zero rows from line parameters
        lines = lines[~np.all(lines == 0, axis=1)]
        trackingWindows = trackingWindows[~np.all(trackingWindows == 0, axis=1)]
        meanLinesInWindows = meanLinesInWindows[~np.all(meanLinesInWindows == 0, axis=1)]

        return lines, trackingWindows, meanLinesInWindows

    def updateTrackingBoxes(self, boxID=0, lineIntersection=None):

        if not self.isInitialized and lineIntersection is None:
            self.rowTrackingBoxes = []
            for i in range(len(self.scanFootSteps)):
                center_x = self.scanFootSteps[i]

                bottom_half = self.trackerParams["trackingBoxWidth"] / 2.0
                top_half = (self.trackerParams["trackingBoxWidth"] * self.trackerParams["sacleRatio"]) / 2.0

                boxBL_x = int(center_x - bottom_half)
                boxBR_x = int(center_x + bottom_half)
                boxTL_x = int(center_x - top_half)
                boxTR_x = int(center_x + top_half)

                boxT_y = self.trackerParams["bottomOffset"]
                boxB_y = self.imgHeight - self.trackerParams["topOffset"]

                self.rowTrackingBoxes.append(Polygon([
                    (boxBR_x, boxB_y),
                    (boxBL_x, boxB_y),
                    (boxTL_x, boxT_y),
                    (boxTR_x, boxT_y)
                ]))
        else:
            center_bottom = lineIntersection[0]
            center_top = lineIntersection[1]

            bottom_half = self.trackerParams["trackingBoxWidth"] / 2.0
            top_half = (self.trackerParams["trackingBoxWidth"] * self.trackerParams["sacleRatio"]) / 2.0

            boxBL_x = int(center_bottom - bottom_half)
            boxBR_x = int(center_bottom + bottom_half)
            boxTL_x = int(center_top - top_half)
            boxTR_x = int(center_top + top_half)

            boxT_y = self.trackerParams["bottomOffset"]
            boxB_y = self.imgHeight - self.trackerParams["topOffset"]

            self.rowTrackingBoxes[boxID] = Polygon([
                (boxBR_x, boxB_y),
                (boxBL_x, boxB_y),
                (boxTL_x, boxT_y),
                (boxTR_x, boxT_y)
            ])
    def minRowSpacing(self, lines):
        # smallest horizontal gap between two lines, at the top and bottom of the tracking boxes
        if len(lines) < 2:
            return np.inf
        ends = np.c_[lineIntersectWin(lines[:, 1], lines[:, 0], self.imgHeight,
                                      self.trackerParams["topOffset"],
                                      self.trackerParams["bottomOffset"])]
        return min(np.min(np.diff(np.sort(ends[:, 0]))), np.min(np.diff(np.sort(ends[:, 1]))))

    def checkPlantsInRows(self, cropRowID, plantsInCroRow):

        if len(plantsInCroRow) >= 2:
            return True
        else:
            self.lostCropRows[cropRowID] = 1
            return False   

    def checkPlantsLocTB(self, point):
        if point[1] < self.imgHeight/2:
            self.pointsInTop += 1
        else:
            self.pointsInBottom += 1

    def trackCropLane(self, mode=None):
        """function to get lines at given window locations 
        Returns:
            _type_: _description_
        """
        if self.groundTracker is not None:
            return self.trackCropLaneGround(mode)
        P, ang = None, None
        # if the feature extractor is initalized
        if self.cropLaneFound:
            # get the lines at windows defined through previous found lines
            lines, linesROIs, meanLinesInWindows = self.findLinesInImage()

            # rows missing in this frame keep their previous line (gaps in a row);
            # the lane is lost only when no row at all is seen, as at the end of the field
            allRowsMissing = np.all(self.missedFrames > 0)
            # previous lines too: a line squeezed out by its neighbour gets no plants
            spacing = min(self.minRowSpacing(self.CropRows), self.minRowSpacing(lines))
            merged = spacing < 0.5 * self.initRowSpacing
            if not allRowsMissing and (merged or np.max(self.missedFrames) > self.maxMissedFrames):
                # two lines collapsed onto the same row, or a row unseen for too long
                # (its frozen line would bias the steering): scan the image again
                if merged:
                    self.log("#[ERR] Crop rows merged (spacing %.0f px, initially %.0f px): re-initializing"
                          % (spacing, self.initRowSpacing))
                else:
                    self.log("#[ERR] A crop row was not seen for %d frames: re-initializing"
                          % np.max(self.missedFrames))
                self.isInitialized = False
                self.cropLaneFound = False
                self.updateTrackingBoxes()
                self.numOfCropRows = len(self.rowTrackingBoxes)
                return self.cropLaneFound, self.cropRowEnd, self.P, self.ang

            if not allRowsMissing:
                # the line parameters are the new tracked lines (for the next step)
                self.CropRows = lines
                # location is always the left side of the window
                self.trackingBoxLoc = meanLinesInWindows - self.trackerParams["trackingBoxWidth"]/2
                # average image intersections of all found lines
                avgOfLines = np.mean(np.c_[lineIntersectImgUpDown(
                    self.CropRows[:, 1], self.CropRows[:, 0], self.imgHeight)], axis=0)
                #  get AvgLine in image cords
                self.mainLine_up, self.mainLine_down = getImgLineUpDown(
                    avgOfLines, self.imgHeight)
                # compute all intersections between the image and each line
                allLineIntersect = np.c_[lineIntersectWin(self.CropRows[:, 1],
                                                          self.CropRows[:, 0],
                                                          self.imgHeight,
                                                          self.trackerParams["topOffset"],
                                                          self.trackerParams["bottomOffset"])]
                # store start and end points of lines - mainly for plotting
                self.allLineStart = np.c_[
                    allLineIntersect[:, 0], self.imgHeight - self.trackerParams["bottomOffset"] * np.ones((len(self.CropRows), 1))]
                self.allLineEnd = np.c_[
                    allLineIntersect[:, 1], self.trackerParams["topOffset"] * np.ones((len(self.CropRows), 1))]
                # main features
                self.P = self.cameraToImage([avgOfLines[1], self.imgHeight])
                self.ang = computeTheta(self.mainLine_up, self.mainLine_down)
                self.cropLaneFound = True

            else:
                self.log("#[ERR] Lost all crop rows")
                self.cropLaneFound = False
        else:
            self.log('Running rest()..')
        
        if self.pointsInBottom == 0 and self.pointsInTop == 0:
            self.cropLaneFound = False

        self.checkRowEnd(mode)
        return self.cropLaneFound, self.cropRowEnd, self.P, self.ang

    def checkRowEnd(self, mode):
        if self.pointsInBottom == 0 and mode in [2, 5]:
            self.cropRowEnd = True
        elif self.pointsInTop == 0 and mode in [1, 4]:
            self.cropRowEnd = True
        else:
            self.cropRowEnd = False

    def trackCropLaneGround(self, mode=None):
        """track the rows on the ground plane: line fit on the last plants of each row,
        metric plausibility checks (see groundTracker.RowTracker)"""
        tracker = self.groundTracker
        self.trackingLost = False
        if not self.cropLaneFound:
            return self.cropLaneFound, self.cropRowEnd, self.P, self.ang
        # plants of each row, inside the boxes around the previous lines
        self.findLinesInImage()
        xNear, xFar = tracker.viewRange(self.imgHeight, self.trackerParams["topOffset"])
        # lateral error and limits are measured at the bottom of the image
        xRef = xNear
        if not tracker.initialized:
            # tracker reset since the rows were found: find them again
            self.isInitialized = False
            self.cropLaneFound = False
            return self.cropLaneFound, self.cropRowEnd, self.P, self.ang
        # plants of the rows in the far half of the view: the row goes on
        xAhead = (xNear + xFar) / 2
        if not tracker.update(self.rowPixels, self.robotPose, xRef, xNear, xAhead, self.exiting):
            self.trackStatus = tracker.status
            self.log('#[ERR] ' + tracker.status)
            self.cropLaneFound = False
            self.trackingLost = True
            return self.cropLaneFound, self.cropRowEnd, self.P, self.ang
        self.trackStatus = tracker.status

        # ground lines back into the image: tracking boxes, drawing and visual servoing
        rows, center = tracker.rowLines(self.robotPose, xRef)
        self.CropRows = np.array([tracker.lineToImage(a - b * xRef, b, xNear, xFar) for a, b in rows])
        self.numOfCropRows = len(self.CropRows)
        c, m = tracker.lineToImage(center[0] - center[1] * xRef, center[1], xNear, xFar)
        # same features as the image tracker, from the lane center line
        avgOfLines = [m * self.imgHeight + c, c]
        self.mainLine_up, self.mainLine_down = getImgLineUpDown(avgOfLines, self.imgHeight)
        allLineIntersect = np.c_[lineIntersectWin(self.CropRows[:, 1], self.CropRows[:, 0], self.imgHeight,
                                                  self.trackerParams["topOffset"],
                                                  self.trackerParams["bottomOffset"])]
        self.allLineStart = np.c_[allLineIntersect[:, 0],
                                  self.imgHeight - self.trackerParams["bottomOffset"] * np.ones((len(self.CropRows), 1))]
        self.allLineEnd = np.c_[allLineIntersect[:, 1],
                                self.trackerParams["topOffset"] * np.ones((len(self.CropRows), 1))]
        self.P = self.cameraToImage([avgOfLines[1], self.imgHeight])
        self.ang = computeTheta(self.mainLine_up, self.mainLine_down)
        self.cropLaneFound = True
        # end of row: no plant of the tracked rows ahead any more (weeds or neighbour beds
        # in the top of the image do not count, unlike checkRowEnd)
        self.cropRowEnd = tracker.rowEnd()
        return self.cropLaneFound, self.cropRowEnd, self.P, self.ang

    def cameraToImage(self, P):
        """function to transform the feature point from camera to image frame
        Args:
            P (_type_): point in camera Fr
        Returns:
            _type_: point in image Fr
        """
        if len(P) >= 2:
            P[0] = P[0] - self.imgWidth/2
            P[1] = P[1] - self.imgHeight/2
        return P
    def applySimpleROI(self, img):
        """Applique une ROI rectangulaire simple pour la zone des rangées"""
        _img = img.copy()
        height, width = _img.shape[:2]
        
        # Garder seulement la partie centrale (éliminer les bords)
        roi_start_x = int(width * 0.15)   # 15% depuis la gauche
        roi_end_x = int(width * 0.85)     # 85% depuis la gauche
        roi_start_y = int(height * 0.3)   # 30% depuis le haut (éliminer le ciel)
        roi_end_y = height                # Garder jusqu'en bas
        
        # Créer un masque
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[roi_start_y:roi_end_y, roi_start_x:roi_end_x] = 255
        
        # Appliquer le masque (ne garder que la zone d'intérêt)
        _img = cv.bitwise_and(_img, _img, mask=mask)
        
        return _img
    def processRGBImage(self, rgbImg):
        """function to extract the greenidx and the contour center points
        Returns:
            _type_: _description_
        """
        self.log("\n" + "="*60)
        self.log(">>> PROCESS RGB IMAGE IS RUNNING (SOURCE FILE VERSION) <<<")
        self.log("="*60 + "\n")
        # change datatype to enable negative values for self.greenIDX calculation
        rgbImg = rgbImg.astype('int32')
        
        # # Appliquer ROI simple avant traitement
        # rgbImg = self.applySimpleROI(rgbImg)
        
        # Commenté temporairement - éviter l'erreur
        # apply ROI (polygon si configuré)
        # rgbImg = self.applyROI(rgbImg)
        
        #  get green index and binary mask
        mask, greenIDX = self.getExgMask(rgbImg)
            
        # Vérifier si le masque est valide
        if mask is None or mask.size == 0 or np.max(mask) == 0:
            self.log("Warning: Empty mask detected, returning empty results")
            return mask, greenIDX, [], np.zeros((2, 0))  # 2 x N like the plant centers
        
        if self.contourParams.get("pointExtraction", "contours") == "bands":
            # contours are only drawn; the points come from horizontal bands of the mask
            plantObjects = getPlantMasks(mask, self.contourParams["minContourArea"], maxSize=None)
            points = self.getBandPoints(mask)
            return mask, greenIDX, plantObjects, points

        # find contours
        plantObjects = getPlantMasks(
            mask, self.contourParams["minContourArea"], bushy=False)
        
        self.log(f"Number of plant objects found: {len(plantObjects)}")
    
        # Afficher la taille des 10 premiers contours
        for i, cnt in enumerate(plantObjects[:10]):
            area = cv.contourArea(cnt)
            x, y, w, h = cv.boundingRect(cnt)
            self.log(f"  Contour {i}: area={area:.0f}, bbox=({x},{y},{w},{h})")
        
        # get center of contours
        if len(plantObjects) == 0:
            self.log("No plant objects found")
            return mask, greenIDX, plantObjects, np.zeros((2, 0))
        
        contCenterPTS = getContourCenter(plantObjects)
        
        if len(contCenterPTS) == 0:
            return mask, greenIDX, plantObjects, np.zeros((2, 0))
        
        x = contCenterPTS[:, 1]
        y = contCenterPTS[:, 0]
        
        return mask, greenIDX, plantObjects, np.array([x, y])

    def getBandPoints(self, mask):
        """row points from the mask, independent of how the plants are separated: the image
        is cut in horizontal bands and each run of vegetation columns in a band gives one
        point. Touching plants or a continuous row (one big contour) give many points along
        the row instead of one center or none."""
        h, w = mask.shape[:2]
        band = max(int(self.contourParams.get("bandHeight", 12)), 2)
        minRun = max(int(self.contourParams.get("minRunWidth", 3)), 1)
        maxRun = int(self.contourParams.get("maxRunWidth", 0)) or w // 4
        fill = self.contourParams.get("bandFill", 0.3)
        xs, ys, ws = [], [], []
        binary = (mask > 0)
        for top in range(0, h - band + 1, band):
            # columns of this band with enough vegetation
            cols = np.count_nonzero(binary[top:top + band], axis=0) >= fill * band
            if not cols.any():
                continue
            edges = np.flatnonzero(np.diff(np.r_[0, cols.astype(np.int8), 0]))
            starts, ends = edges[0::2], edges[1::2]
            width = ends - starts
            # wider runs are several rows merged (far away) or a weed patch: not a row point
            ok = (width >= minRun) & (width <= maxRun)
            xs.extend(((starts[ok] + ends[ok] - 1) / 2.0).tolist())
            ys.extend([top + band / 2.0] * int(np.count_nonzero(ok)))
            ws.extend(width[ok].tolist())
        # vegetation width (pixels) of each point: weighs crop rows against small weeds
        self.plantWidths2D = np.array(ws, dtype=float)
        return np.array([xs, ys], dtype=float).reshape(2, -1)

    def getExgMask(self, img):
        """Function to compute the green index of the image
        """
        self.log(f"DEBUG getExgMask: img shape={img.shape}, dtype={img.dtype}")
        
        _img = img.copy()
        _img = _img.astype('int32')
        
        # Vegetation Mask
        r = _img[:, :, 0]
        g = _img[:, :, 1]
        b = _img[:, :, 2]

        # calculate Excess Green Index (symmetric in r and b: BGR or RGB order does not matter)
        greenIDX = 2*g - r - b
        if self.contourParams.get("exgNormalized", False):
            # chromatic ExG = (2G-R-B)/(R+G+B): independent of the brightness, so leaves in
            # the robot's shadow and in full sun get the same value; too dark pixels are noise
            total = r + g + b
            greenIDX = np.where(total > 3 * self.contourParams.get("exgDarkLevel", 20),
                                255.0 * greenIDX / np.maximum(total, 1), 0)
        # clip, not wrap: a bright leaf (ExG 360) must not become 104 in uint8
        greenIDX = np.clip(greenIDX, 0, 255).astype('uint8')
        
        self.log(f"DEBUG: greenIDX shape={greenIDX.shape}, min={greenIDX.min()}, max={greenIDX.max()}")

        # Otsu's thresholding after gaussian smoothing
        blur = cv.GaussianBlur(greenIDX, (5, 5), 0)
        threshold, threshIMG = cv.threshold(
            blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # Otsu always splits the image in two: on bare soil it would turn soil noise into
        # plants, so the threshold never goes below a minimal green level
        exgMin = self.contourParams.get("exgMin", 0.0) * 255
        if threshold < exgMin:
            threshold, threshIMG = cv.threshold(blur, exgMin, 255, cv.THRESH_BINARY)
        
        self.log(f"DEBUG: Otsu threshold={threshold}, threshIMG max={threshIMG.max()}")

        # CORRECTION: Dilatation beaucoup plus petite
        kernel = np.ones((3, 3), np.uint8)
        binaryMask = cv.dilate(threshIMG, kernel, iterations=1)
        
        self.log(f"DEBUG: binaryMask shape={binaryMask.shape}, non-zero pixels={np.count_nonzero(binaryMask)}")

        return binaryMask, greenIDX

    def applyROI(self, img):
        _img = img.copy()
        # defining ROI windown on the image
        if self.roiParams["enable_roi"]:
            try:
                r_pts = [self.roiParams["p1"], self.roiParams["p2"],
                        self.roiParams["p3"], self.roiParams["p4"]]
                l_pts = [self.roiParams["p5"], self.roiParams["p6"],
                        self.roiParams["p7"], self.roiParams["p8"]]
                
                # Vérifier que les points sont valides
                if len(r_pts) >= 3 and all(isinstance(p, (list, tuple)) for p in r_pts):
                    cv.fillPoly(_img, np.array([r_pts], dtype=np.int32), (0, 0, 0))
                if len(l_pts) >= 3 and all(isinstance(p, (list, tuple)) for p in l_pts):
                    cv.fillPoly(_img, np.array([l_pts], dtype=np.int32), (0, 0, 0))
            except Exception as e:
                self.log(f"Error in applyROI: {e}")
                pass
        return _img

    def drawGraphics(self):
        """function to draw the lines and the windows onto the image (self.primaryRGBImg)
        """
        # Vérifier que primaryRGBImg existe et n'est pas vide
        if self.primaryRGBImg is not None and len(self.primaryRGBImg) > 0:
            self.graphicsImg = self.primaryRGBImg.copy()
            
            # Vérifier que les données nécessaires existent
            if hasattr(self, 'mainLine_up') and hasattr(self, 'mainLine_down') and len(self.mainLine_up) > 0 and len(self.mainLine_down) > 0:
                # main line
                cv.line(self.graphicsImg, (int(self.mainLine_up[0]), int(self.mainLine_up[1])), (int(
                    self.mainLine_down[0]), int(self.mainLine_down[1])), (255, 0, 0), thickness=3)
            
            # contoures
            if hasattr(self, 'plantObjects2D') and self.plantObjects2D is not None and len(self.plantObjects2D) > 0:
                cv.drawContours(self.graphicsImg,
                                self.plantObjects2D, -1, (10, 50, 150), 3)
            
            # helper lines
            if hasattr(self, 'allLineStart') and len(self.allLineStart) > 0:
                for i in range(0, len(self.allLineStart)):
                    cv.line(self.graphicsImg, (int(self.allLineStart[i, 0]), int(self.allLineStart[i, 1])), (int(
                        self.allLineEnd[i, 0]), int(self.allLineEnd[i, 1])), (0, 255, 0), thickness=1)

            # tracking boxes
            if hasattr(self, 'rowTrackingBoxes') and self.rowTrackingBoxes is not None and len(self.rowTrackingBoxes) > 0:
                for i in range(min(self.numOfCropRows, len(self.rowTrackingBoxes))):
                    int_coords = lambda x: np.array(x).round().astype(np.int32)
                    exterior = [int_coords(self.rowTrackingBoxes[i].exterior.coords)]
                    cv.polylines(self.graphicsImg, exterior, True, (0, 255, 255))

            # plant centers
            if hasattr(self, 'plantCenters2D') and self.plantCenters2D is not None and len(self.plantCenters2D) > 0 and len(self.plantCenters2D[0]) > 0:
                for i in range(len(self.plantCenters2D[0])):
                    x = int(self.plantCenters2D[0, i])
                    y = int(self.plantCenters2D[1, i])
                    self.graphicsImg = cv.circle(
                        self.graphicsImg, (x, y), 3, (255, 0, 255), 5)
        else:
            self.log("No primaryRGBImg to draw")

    def handleKey(self, sleepTime=0):
        key = cv.waitKey(sleepTime)
        # Close program with keyboard 'q'
        if key == ord('q'):
            cv.destroyAllWindows()
            exit(1)