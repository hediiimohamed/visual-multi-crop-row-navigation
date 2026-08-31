
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from visual_multi_crop_row_navigation import contours
from visual_multi_crop_row_navigation.geometric import *

def getContourCenter(contours):
    """function to compute the center points of the contours

    Args:
        contours (_type_): contours from image

    Returns:
        _type_: contours centers
    """
    # get center of contours
    if len(contours) == 0:
        return np.array([])
    
    contCenterPTS = np.zeros((len(contours), 2))
    for i in range(0, len(contours)):
        # get contour
        c_curr = contours[i]
        
        # Vérifier que le contour est valide
        if c_curr is None or len(c_curr) == 0:
            continue
            
        # get moments
        M = cv.moments(c_curr)
        
        # compute center of mass
        if(M['m00'] != 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            contCenterPTS[i,:] = [cy, cx]

    # Filtrer les points non nuls
    if len(contCenterPTS) > 0:
        contCenterPTS = contCenterPTS[~np.all(contCenterPTS == 0, axis=1)]
    return contCenterPTS

def getPlantMasks(binrayMask, min_contour_area, bushy=False):
    # Vérifier que le masque est valide
    if binrayMask is None:
        return []
    
    # Convertir en uint8 si nécessaire
    if binrayMask.dtype != np.uint8:
        binrayMask = binrayMask.astype(np.uint8)
    
    # S'assurer que le masque contient des valeurs
    if np.max(binrayMask) == 0:
        return []
    
    # Trouver les contours
    try:
        contours, _ = cv.findContours(binrayMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    except Exception as e:
        print(f"Error finding contours: {e}")
        return []
    
    # Filtrer les contours
    filtered_contours = []
    for contour in contours:
        try:
            if len(contour) < 3:
                continue
            
            area = cv.contourArea(contour)
            
            # Rejeter les contours trop petits
            if area < min_contour_area:
                continue
            
            # FILTRE PLUS STRICT : Rejeter les gros blobs (herbe)
            if area > 1500:  # <-- Réduit de 1000 à 600
                continue
            
            x, y, w, h = cv.boundingRect(contour)
            
            # Rejeter les contours trop larges
            if w > 200 or h > 200:  # <-- Réduit de 80 à 50
                continue
            
            filtered_contours.append(contour)
            
        except Exception as e:
            continue
    
    return filtered_contours              
def splitContours(contour, x, y, w, h, max_coutour_height):
    """splits larg contours in smaller regions 

    Args:
        contour (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        w (_type_): _description_
        h (_type_): _description_

    Returns:
        _type_: sub polygons (seperated countours)
    """
    sub_polygon_num = h // max_coutour_height
    sub_polys = list()
    subContour = list()
    vtx_idx = list()
    contour = [contour.squeeze().tolist()]
    for subPoly in range(1, sub_polygon_num + 1):
        for vtx in range(len(contour[0])): 
            if  (subPoly - 1 * max_coutour_height) -1 <=  contour[0][vtx][1] and \
                (subPoly * max_coutour_height) -1 >= contour[0][vtx][1] and \
                vtx not in vtx_idx:
                subContour.append([contour[0][vtx]])
                vtx_idx.append(vtx)

        sub_polys.append(np.array(subContour))
        subContour = list()

    return sub_polys

def sortContours(contours, method="left-to-right"):
    """initialize the reverse flag and sort index

    Args:
        cnts (_type_): _description_
        method (str, optional): _description_. Defaults to "left-to-right".

    Returns:
        _type_: sorted countours, bboxes
    """
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return contours, boundingBoxes

def getContoursInWindow(contourCenters, box):
    """iflters out countours inside a box

    Args:
        contourCenters (_type_): _description_
        box (_type_): _description_

    Returns:
        _type_: contour centers
    """
    points = []
    for cnt in range(len(contourCenters[1])):
        x, y = contourCenters[0][cnt], contourCenters[1][cnt]
        if isInBox(list(box), [x, y]):
            points.append([x, y])
    return points