# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:06:14 2020

@author: PJ
"""

import cv2
import numpy as np
import math

SIZE = 9

def get_connected_component (image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]
    
    if(len(sizes) <= 1):
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image
    
    max_label = 1
    max_size = sizes[1]
    
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2.fill(255)
    img2[output == max_label] = 0
    
    return img2

def angle_between_two_vectors(vector1, vector2):
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return math.degrees(angle)

def check_if_sides_are_equals(A, B, C, D, tolerance):
    AB = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    AD = math.sqrt((A[0]-D[0])**2 + (A[1]-D[1])**2)
    BC = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)
    CD = math.sqrt((C[0]-D[0])**2 + (C[1]-D[1])**2)
    shortest = min(AB, AD, BC, CD)
    longest = max(AB, AD, BC, CD)
    return longest > tolerance * shortest

def is_approx_90_degrees(angle, tolerance):
    return abs(angle - 90) < tolerance

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

def check_if_is_square(rect):
    #   A------B
    #   |      |
    #   |      |
    #   D------C
    A = rect[0]
    B = rect[1]
    C = rect[2]
    D = rect[3]
    
    # Calculate vectors
    AB = B - A
    AD = D - A
    BC = C - B
    DC = C - D
    
    # Calculate angles between vectors
    AB_AD = angle_between_two_vectors(AB, AD)
    AB_BC = angle_between_two_vectors(AB, BC)
    BC_DC = angle_between_two_vectors(BC, DC)
    AD_DC = angle_between_two_vectors(AD, DC)
    
    if not (is_approx_90_degrees(AB_AD, 15) and is_approx_90_degrees(AB_BC, 15) and 
            is_approx_90_degrees(BC_DC, 15) and is_approx_90_degrees(AD_DC, 15)):
        return False
    
    # Check if sides have the same length
    if check_if_sides_are_equals(A, B, C, D, 1.2):
        return False
    
    return True
    
        
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720) 

grid = []
for i in range(SIZE):
    row = []
    for j in range(SIZE):
        row.append(0)
        grid.append(row)

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
            if check_if_is_square(rect):
                
                # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
                (tl, tr, br, bl) = rect
                bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                
                max_width = max(int(bottom_width), int(top_width))
                max_height = max(int(right_height), int(left_height))
                
                dst = np.array([
            		[0, 0],
            		[max_width - 1, 0],
            		[max_width - 1, max_height - 1],
            		[0, max_height - 1]], dtype = "float32")
                
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(frame, M, (max_width, max_height))
                cv2.imshow("Sudoku-Original", warped)
                warped_copy = warped.copy()
                warped_copy = cv2.cvtColor(warped_copy, cv2.COLOR_BGR2GRAY)
                warped_copy = cv2.GaussianBlur(warped_copy, (7,7), 0)
                warped_copy = cv2.adaptiveThreshold(warped_copy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) 
                warped_copy = cv2.bitwise_not(warped_copy)
                cv2.imshow("Sudoku-Processed", warped_copy)
                
                cell_height = warped_copy.shape[0] // 9
                cell_width = warped_copy.shape[1] // 9
                
                offset_width = math.floor(cell_width / 10)    
                offset_height = math.floor(cell_height / 10)
                whites= 0
                        
                for i in range(SIZE):
                    for j in range(SIZE):
                        cell = warped_copy[cell_height*i+offset_height:cell_height*(i+1)-offset_height, cell_width*j+offset_width:cell_width*(j+1)-offset_width] 
                        ratio = 0.6        
                        # Top
                        while np.sum(cell[0]) <= (1-ratio) * cell.shape[1] * 255:
                            cell = cell[1:]
                        # Bottom
                        while np.sum(cell[:,-1]) <= (1-ratio) * cell.shape[1] * 255:
                            cell = np.delete(cell, -1, 1)
                        # Left
                        while np.sum(cell[:,0]) <= (1-ratio) * cell.shape[0] * 255:
                            cell = np.delete(cell, 0, 1)
                        # Right
                        while np.sum(cell[-1]) <= (1-ratio) * cell.shape[0] * 255:
                            cell = cell[:-1] 
                        
                        cell = cv2.bitwise_not(cell)
                        cell = get_connected_component(cell)
                        digit_pic_size = 28
                        cell_crop = cv2.resize(cell, (digit_pic_size,digit_pic_size))
                        cell_crop = cv2.dilate(cell_crop, np.ones((3, 3), np.uint8))
                        white_pixels = cv2.countNonZero(cell_crop)
                        ratio = white_pixels / float(cell_crop.shape[0]*cell_crop.shape[1])
                        x_pos = cell_width*i + cell_width/2
                        y_pos = cell_height*j + cell_height/2
                        
                        print(ratio)
                        #print(ratio)
                        if ratio > 0.9:
                            grid[i][j] = 0
                            whites = whites+1
                            cv2.circle(warped,(int(x_pos),int(y_pos)), 5,(0,0,255),5)
                            continue
                        else:
                            grid[i][j] = 1
                            cv2.circle(warped,(int(x_pos),int(y_pos)), 5,(0,255,0),5)
                        if cell.shape[0] > 0 and cell.shape[1] > 0:
                            k = i + j
                            cv2.imshow(str(k), cell_crop)
                cv2.imshow("white", warped)
                #print(whites)
                #print(grid)
                #print("\n")
                #print("=======")
                        
              
            cv2.circle(frame, (rect[0][0], rect[0][1]), 5, (0,0,255), 5)
            cv2.circle(frame, (rect[1][0], rect[1][1]), 5, (0,0,255), 5)
            cv2.circle(frame, (rect[2][0], rect[2][1]), 5, (0,0,255), 5)
            cv2.circle(frame, (rect[3][0], rect[3][1]), 5, (0,0,255), 5)
    
    cv2.drawContours(frame,[max_contour], 0,  (0,255,0), 3)
    cv2.imshow("Original", frame)
    cv2.imshow("Output", thresh)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()