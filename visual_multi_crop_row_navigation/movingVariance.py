#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import copy

# funciton to compute the moving standard deviation for a given window size

def movingStd(data, winSize=5):
    _data = data.copy()
    
    # Vérifier qu'il y a assez de données
    if len(_data) <= winSize:
        print(f"Warning: Not enough data points ({len(_data)}) for window size {winSize}")
        return np.array([])
    
    # compute a moving standard deviation
    stdVec = np.zeros((len(_data)-winSize, 1))
    for i in range(0, len(stdVec)):
        sqsum = 0
        meanVal = np.mean(_data[i:i+winSize])
        for j in range(i, i+winSize):
            sqsum += (_data[j]-meanVal)**2
        stdVec[i] = np.sqrt(sqsum/(winSize))

    # normalize
    if np.max(stdVec) > 0:
        stdVec = stdVec/max(stdVec)
    stdVec = stdVec.reshape(len(stdVec),)
    return stdVec

def findPicksTroughths(movVarM, prominence=0.5, height=None):
    _movVarM = movVarM.copy()
    # find negative peaks (=least std, most stable line)
    peaksNeg, _ = find_peaks(-_movVarM, height=height)
    # find positive peaks (=least stable lines = crop row transition)
    peaksPos, _ = find_peaks(_movVarM, prominence=0.5, height=height)
    return peaksPos, peaksNeg