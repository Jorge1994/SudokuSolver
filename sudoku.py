# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:06:14 2020

@author: PJ
"""

import cv2
import numpy as np

def find_max_contour(contours):
    max_area = -1
    max_contour = None
  
    for i in range(len(contours)):
        temp = contours[i]
        area = cv2.contourArea(temp)
        if area > max_area:
            max_area = area
            max_contour = temp
   
    return max_area, max_contour

def get_corners_from_contour(contours, corner_amount=4):
    coefficient = 1
    iterations = 300
    while iterations > 0 and coefficient >= 0:
        iterations = iterations -1
        epsilon = coefficient * cv2.arcLength(contours, True)
        poly_approx = cv2.approxPolyDP(contours, epsilon, True)
        hull = cv2.convexHull(poly_approx)
   
        if len(hull) == corner_amount:
            return hull
        else:
            if len(hull) > corner_amount:
                coefficient += .01
            else:
                coefficient -= .01
                
    return None

def find_corners_locations(corners):
    max_sum = 1000000
    min_sum = 0
    rect = np.zeros((4, 2), dtype = "float32")
    index = -1
    
    # Top Left Corner -> smallest sum
    for i in range(len(corners)):
        temp_sum = corners[i][0] + corners[i][1]
        if temp_sum < max_sum:
            max_sum = temp_sum
            index = i
    rect[0] = corners[index]
    corners = np.delete(corners, index, 0)
    
    # Bottom Rigth -> biggest sum 
    for i in range(len(corners)):
        temp_sum = corners[i][0] + corners[i][1]
        if temp_sum > min_sum:
            min_sum = temp_sum
            index = i
    rect[2] = corners[index]
    corners = np.delete(corners, index, 0)
    
    if(corners[0][0] > corners[1][0]):
        rect[1] = corners[0]
        rect[3] = corners[1]
        
    else:
        rect[1] = corners[1]
        rect[3] = corners[0]

    rect = rect.reshape(4,2)
    return rect
        
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, max_contour = find_max_contour(contours)
    if max_contour is not None:
        corners = get_corners_from_contour(max_contour)
        if corners is not None:
            corners = corners.reshape(4,2)
            rect = find_corners_locations(corners)
            cv2.circle(frame, (rect[2][0], rect[2][1]), 5, (255,0,0), 5)
        
    #print(corners[0][0][0])
    
    cv2.drawContours(frame,[max_contour], 0,  (0,255,0), 3)
    cv2.imshow("Original", frame)
    cv2.imshow("Output", thresh)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()