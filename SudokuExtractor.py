# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:06:14 2020

@author: PJ
"""

import cv2
import numpy as np
import math
import Solver
import copy
from scipy import ndimage

SIZE = 9

def are_matrices_equals(matrix_1, matrix_2, row, col):
    for i in range(row):
        for j in range(col):
            if matrix_1[i][j] != matrix_2[i][j]:
                return False
    return True

def infer_grid(img):
    squares = []
    side = img.shape
    side_h = side[0] // 9
    side_w = side[1] // 9
    for i in range(9):
        for j in range(9):
            p1 = (i * side_w, j * side_h)  # Top left corner of a bounding box
            p2 = ((i + 1) * side_w, (j + 1) * side_h)  # Bottom right corner of bounding box
            #cv2.rectangle(img, (int(p1[0]), int(p1[1])),(int(p2[0]), int(p2[1])), (255,0,0), 3)
            squares.append((p1, p2))
    return squares

def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
	"""
	Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
	connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
	"""
	img = inp_img.copy()  # Copy the image, leaving the original untouched
	height, width = img.shape[:2]

	max_area = 0
	seed_point = (None, None)

	if scan_tl is None:
		scan_tl = [0, 0]

	if scan_br is None:
		scan_br = [width, height]

	# Loop through the image
	for x in range(scan_tl[0], scan_br[0]):
		for y in range(scan_tl[1], scan_br[1]):
			# Only operate on light or white squares
			if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
				area = cv2.floodFill(img, None, (x, y), 64)
				if area[0] > max_area:  # Gets the maximum bound area which should be the grid
					max_area = area[0]
					seed_point = (x, y)

	# Colour everything grey (compensates for features outside of our middle scanning range
	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 255 and x < width and y < height:
				cv2.floodFill(img, None, (x, y), 64)

	mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

	# Highlight the main feature
	if all([p is not None for p in seed_point]):
		cv2.floodFill(img, mask, seed_point, 255)

	top, bottom, left, right = height, 0, width, 0

	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 64:  # Hide anything that isn't the main feature
				cv2.floodFill(img, mask, (x, y), 0)

			# Find the bounding parameters
			if img.item(y, x) == 255:
				top = y if y < top else top
				bottom = y if y > bottom else bottom
				left = x if x < left else left
				right = x if x > right else right

	bbox = [[left, top], [right, bottom]]
	return np.array(bbox, dtype='float32')


def cut_from_rect(img, rect):
	"""Cuts a rectangle from an image using the top left and bottom right points."""
	return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def centre_pad(length,size):
  	"""Handles centering for a given length that may be odd or even."""
  	if length % 2 == 0:
  		side1 = int((size - length) / 2)
  		side2 = side1
  	else:
  		side1 = int((size - length) / 2)
  		side2 = side1 + 1
  	return side1, side2


def scale_and_centre(img, size, margin=0, background=0):
	"""Scales and centres an image onto a new background square."""
	h, w = img.shape[:2]

	if h > w:
		t_pad = int(margin / 2)
		b_pad = t_pad
		ratio = (size - margin) / h
		w, h = int(ratio * w), int(ratio * h)
		l_pad, r_pad = centre_pad(w,size)
	else:
		l_pad = int(margin / 2)
		r_pad = l_pad
		ratio = (size - margin) / w
		w, h = int(ratio * w), int(ratio * h)
		t_pad, b_pad = centre_pad(h,size)

	img = cv2.resize(img, (w, h))
	img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
	return cv2.resize(img, (size, size))

def extract_digit(img, rect, size):
	"""Extracts a digit (if one exists) from a Sudoku square."""

	digit = cut_from_rect(img, rect)  # Get the digit box from the whole square

	# Use fill feature finding to get the largest feature in middle of the box
	# Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
	h, w = digit.shape[:2]
	margin = int(np.mean([h, w]) / 2.5)
	bbox = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
	digit = cut_from_rect(digit, bbox)

	# Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
	w = bbox[1][0] - bbox[0][0]
	h = bbox[1][1] - bbox[0][1]

	# Ignore any small bounding boxes
	if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
		return scale_and_centre(digit, size, 4)
	else:
		return np.zeros((size, size), np.uint8)

def pre_process_image(img):
	proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
	proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	proc = cv2.bitwise_not(proc, proc)
	return proc

def get_digits(img, squares, size):
    digits = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = pre_process_image(img.copy())
    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits

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

def prepare(img_array):
    new_array = img_array.reshape(-1, 28, 28, 1)
    new_array = new_array.astype('float32')
    new_array /= 255
    return new_array

def image_to_array(array, model):
    grid = create_grid()
    cv2.imshow("wewewewe",array[28:28+28, 28:28+28])
    for i in range(9):
        for j in range (9):
            if np.sum(array[i*28:i*28+28, j*28:j*28+28]) > 0:
                digit_img = array[i*28:i*28+28, j*28:j*28+28]
                
                _, digit_img = cv2.threshold(digit_img, 200, 255, cv2.THRESH_BINARY) 
                digit_img = digit_img.astype(np.uint8)
                #digit_img = cv2.bitwise_not(digit_img)
                
                shift_x, shift_y = get_best_shift(digit_img)
                shifted = shift(digit_img,shift_x,shift_y)
                digit_img = shifted
                digit_img = cv2.bitwise_not(digit_img)
                
                digit_img2 = prepare(digit_img)
                
                prediction= model.predict(digit_img2)
                #cv2.imshow(str(np.argmax(prediction[0])), digit_img)
                #print(np.argmax(prediction[0]))
                grid[i][j] = np.argmax(prediction[0])+1
            else:
                grid[i][j] = 0
    #print(grid)
    return grid
    
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

def create_grid():
    """ Create an empty 9x9 matrix/grid (filled with 0) """
    grid = []
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            row.append(0)
        grid.append(row)
    return grid
        
def draw_solution_on_image(img, solved, unsolved):
    for i in range(SIZE):
        for j in range(SIZE):
            if unsolved[i][j] != 0:
                continue
            else:
                num = solved[i][j]
                X = img.shape[0]/9
                Y = img.shape[1]/9
                x_pos = X*i + X/2
                y_pos = Y*j + Y/2
                cv2.putText(img, str(num), (int(y_pos)-5,int(x_pos)+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) 
    
def show_digits(digits, colour=255):
    """ Shows the 81 extracted digits in a matrix/grid format """
    rows = []
    for i in range(9):
        row = np.concatenate(digits[i * 9:((i + 1) * 9)], axis=0)
        rows.append(row)
    img_grid = np.concatenate(rows, axis=1)
    return img_grid

def extract_and_solve_sudoku(frame, model, old_sudoku):
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
                (tl, tr, br, bl) = rect
                bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                
                max_width = max(int(bottom_width), int(top_width))
                max_height = max(int(right_height), int(left_height))
                rect = np.array([tl, tr, br, bl], dtype='float32')
                dst = np.array([
            		[0, 0],
            		[max_width - 1, 0],
            		[max_width - 1, max_height - 1],
            		[0, max_height - 1]], dtype = "float32")
                
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(frame, M, (max_width, max_height))
                
                warped_copy = warped.copy()
                squares = infer_grid(warped_copy)
                digits = get_digits(warped_copy, squares, 28)
                #cv2.imshow("WWEWE", digits[9])
                img_grid = show_digits(digits)
                cv2.imshow("Digits Grid", img_grid)
                grid = image_to_array(img_grid, model)
                unsolved_grid = copy.deepcopy(grid) 
                #zeros = 81 - np.count_nonzero(unsolved_grid)
            
                if(np.count_nonzero(unsolved_grid) >= 17):
                    if (not old_sudoku is None) and are_matrices_equals(old_sudoku, grid, SIZE, SIZE):
                       if(Solver.is_solved(grid)):
                           draw_solution_on_image(warped_copy, old_sudoku, unsolved_grid)
                    else:
                        print(grid)
                        Solver.solve_sudoku(grid)
                        if(Solver.is_solved(grid)):
                            draw_solution_on_image(warped_copy, grid, unsolved_grid)
                            old_sudoku = copy.deepcopy(grid)
                else:
                    return frame, None
                
                result_sudoku = cv2.warpPerspective(warped_copy, M, (frame.shape[1], frame.shape[0])
                                        , flags=cv2.WARP_INVERSE_MAP)
                result = np.where(result_sudoku.sum(axis=-1,keepdims=True)!=0, result_sudoku, frame)
               
                # Draw the 4 corners of the Sudoku puzzle
                cv2.circle(result, (rect[0][0], rect[0][1]), 5, (0,0,255), 5)
                cv2.circle(result, (rect[1][0], rect[1][1]), 5, (0,0,255), 5)
                cv2.circle(result, (rect[2][0], rect[2][1]), 5, (0,0,255), 5)
                cv2.circle(result, (rect[3][0], rect[3][1]), 5, (0,0,255), 5)
                
                # Draw the contour of the Sudoku puzzle
                cv2.drawContours(result,[max_contour], 0,  (0,255,0), 3)
                return result, old_sudoku
            else:
                return frame, None
        else:
            return frame, None
    else:
        return frame, None
                #cv2.imshow("solution", result)
                #cv2.imshow("Sudoku-Original", warped_copy)
                #cv2.imshow("Solved", result)
            #cv2.circle(frame, (rect[0][0], rect[0][1]), 5, (0,0,255), 5)
            #cv2.circle(frame, (rect[1][0], rect[1][1]), 5, (0,0,255), 5)
            #cv2.circle(frame, (rect[2][0], rect[2][1]), 5, (0,0,255), 5)
            #cv2.circle(frame, (rect[3][0], rect[3][1]), 5, (0,0,255), 5)
    
    #cv2.drawContours(frame,[max_contour], 0,  (0,255,0), 3)
    #cv2.imshow("Original", frame)
    #cv2.imshow("Output", thresh)
    
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

#cap.release()
#cv2.destroyAllWindows()