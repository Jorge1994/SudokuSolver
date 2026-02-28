"""
Sudoku Grid Extraction and Image Processing Module.

This module handles computer vision operations for detecting and extracting Sudoku puzzles
from video frames, recognizing digits, solving puzzles, and overlaying solutions.
"""

import cv2
import numpy as np
import math
import Solver
from scipy import ndimage

# Grid configuration
GRID_SIZE = 9

def are_grids_equal(grid_1, grid_2, num_rows, num_cols):
    """
    Compare two grids for equality.
    
    Args:
        grid_1: First grid to compare
        grid_2: Second grid to compare
        num_rows: Number of rows to compare
        num_cols: Number of columns to compare
    
    Returns:
        bool: True if grids are equal, False otherwise
    """
    for i in range(num_rows):
        for j in range(num_cols):
            if grid_1[i][j] != grid_2[i][j]:
                return False
    return True

def calculate_cell_boundaries(image):
    """
    Calculate bounding boxes for all 81 cells in a 9x9 Sudoku grid.
    
    Args:
        image: Warped Sudoku grid image
    
    Returns:
        list: List of 81 tuples with (top_left, bottom_right) coordinates for each cell
    """
    cell_rectangles = []
    image_dimensions = image.shape
    cell_height = image_dimensions[0] // 9
    cell_width = image_dimensions[1] // 9
    for i in range(9):
        for j in range(9):
            top_left = (i * cell_width, j * cell_height)
            bottom_right = ((i + 1) * cell_width, (j + 1) * cell_height)
            cell_rectangles.append((top_left, bottom_right))
    return cell_rectangles

def calculate_centering_shift(image):
    """
    Calculate shift needed to center an image based on its center of mass.
    
    Args:
        image: Input image
    
    Returns:
        tuple: (shift_x, shift_y) pixel shifts needed to center the image
    """
    center_y, center_x = ndimage.measurements.center_of_mass(image)
    rows, cols = image.shape
    shift_x = np.round(cols/2.0-center_x).astype(int)
    shift_y = np.round(rows/2.0-center_y).astype(int)
    return shift_x, shift_y

def apply_shift(image, shift_x, shift_y):
    """
    Apply a translation shift to an image.
    
    Args:
        image: Input image
        shift_x: Horizontal shift in pixels
        shift_y: Vertical shift in pixels
    
    Returns:
        np.ndarray: Shifted image
    """
    rows, cols = image.shape
    translation_matrix = np.float32([[1,0,shift_x],[0,1,shift_y]])
    shifted_image = cv2.warpAffine(image, translation_matrix, (cols,rows))
    return shifted_image

def find_largest_connected_component(input_image, scan_top_left=None, scan_bottom_right=None):
    """
    Find the largest connected white pixel structure in the image.
    
    Uses flood fill to identify connected components and isolates the largest one.
    This is typically used to extract the main Sudoku grid from noise.
    
    Args:
        input_image: Binary image to process
        scan_top_left: [x, y] coordinates to start scanning (optional)
        scan_bottom_right: [x, y] coordinates to end scanning (optional)
    
    Returns:
        np.ndarray: Bounding box as [[left, top], [right, bottom]]
    """
    processed_image = input_image.copy()
    height, width = processed_image.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_top_left is None:
        scan_top_left = [0, 0]

    if scan_bottom_right is None:
        scan_bottom_right = [width, height]

    # Find the largest connected component by flood filling
    for x in range(scan_top_left[0], scan_bottom_right[0]):
        for y in range(scan_top_left[1], scan_bottom_right[1]):
            if processed_image.item(y, x) == 255 and x < width and y < height:
                area = cv2.floodFill(processed_image, None, (x, y), 64)
                if area[0] > max_area:
                    max_area = area[0]
                    seed_point = (x, y)

    # Set all white pixels to grey
    for x in range(width):
        for y in range(height):
            if processed_image.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(processed_image, None, (x, y), 64)

    mask = np.zeros((height + 2, width + 2), np.uint8)

    # Highlight the main feature in white
    if all([p is not None for p in seed_point]):
        cv2.floodFill(processed_image, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if processed_image.item(y, x) == 64:
                cv2.floodFill(processed_image, mask, (x, y), 0)

            if processed_image.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bounding_box = [[left, top], [right, bottom]]
    return np.array(bounding_box, dtype='float32')


def extract_rectangle_region(image, rectangle):
    """
    Extract a rectangular region from an image.
    
    Args:
        image: Source image
        rectangle: Rectangle as [[x1, y1], [x2, y2]]
    
    Returns:
        np.ndarray: Extracted rectangular region
    """
    return image[int(rectangle[0][1]):int(rectangle[1][1]), int(rectangle[0][0]):int(rectangle[1][0])]


def calculate_center_padding(content_length, target_size):
    """
    Calculate padding needed to center content within a target size.
    
    Args:
        content_length: Current length of content
        target_size: Target size to fit content into
    
    Returns:
        tuple: (padding_side1, padding_side2)
    """
    if content_length % 2 == 0:
        padding_side1 = int((target_size - content_length) / 2)
        padding_side2 = padding_side1
    else:
        padding_side1 = int((target_size - content_length) / 2)
        padding_side2 = padding_side1 + 1
    return padding_side1, padding_side2


def resize_and_center_image(image, target_size, margin=0, background_color=0):
    """
    Resize and center an image onto a square background.
    
    Maintains aspect ratio while fitting into a square of target_size.
    
    Args:
        image: Input image to resize
        target_size: Target square size (width and height)
        margin: Margin to leave around content (default: 0)
        background_color: Background color value (default: 0)
    
    Returns:
        np.ndarray: Resized and centered image
    """
    height, width = image.shape[:2]

    if height > width:
        top_padding = int(margin / 2)
        bottom_padding = top_padding
        ratio = (target_size - margin) / height
        width, height = int(ratio * width), int(ratio * height)
        left_padding, right_padding = calculate_center_padding(width, target_size)
    else:
        left_padding = int(margin / 2)
        right_padding = left_padding
        ratio = (target_size - margin) / width
        width, height = int(ratio * width), int(ratio * height)
        top_padding, bottom_padding = calculate_center_padding(height, target_size)

    image = cv2.resize(image, (width, height))
    image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, None, background_color)
    return cv2.resize(image, (target_size, target_size))

def extract_digit_from_cell(image, cell_rectangle, digit_size):
    """
    Extract a digit from a Sudoku cell if one exists.
    
    Args:
        image: Preprocessed grid image
        cell_rectangle: Rectangle coordinates of the cell
        digit_size: Target size for the extracted digit
    
    Returns:
        np.ndarray: Extracted digit image or zeros if no digit found
    """
    digit_image = extract_rectangle_region(image, cell_rectangle)

    # Find largest feature in center region where digit should be
    height, width = digit_image.shape[:2]
    margin = int(np.mean([height, width]) / 2.5)
    bounding_box = find_largest_connected_component(digit_image, [margin, margin], [width - margin, height - margin])
    digit_image = extract_rectangle_region(digit_image, bounding_box)

    # Calculate bounding box dimensions
    box_width = bounding_box[1][0] - bounding_box[0][0]
    box_height = bounding_box[1][1] - bounding_box[0][1]

    # Ignore small bounding boxes (likely noise)
    if box_width > 0 and box_height > 0 and (box_width * box_height) > 100 and len(digit_image) > 0:
        return resize_and_center_image(digit_image, digit_size, 4)
    else:
        return np.zeros((digit_size, digit_size), np.uint8)

def preprocess_image_for_digit_extraction(image):
    """
    Preprocess image for digit extraction.
    
    Applies Gaussian blur and adaptive thresholding.
    
    Args:
        image: Input grayscale image
    
    Returns:
        np.ndarray: Preprocessed binary image
    """
    processed = cv2.GaussianBlur(image.copy(), (9, 9), 0)
    processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed = cv2.bitwise_not(processed, processed)
    return processed

def extract_all_digits_from_grid(image, cell_rectangles, digit_size):
    """
    Extract all 81 digits from the Sudoku grid.
    
    Args:
        image: Warped Sudoku grid image
        cell_rectangles: List of 81 cell rectangle coordinates
        digit_size: Size for extracted digit images
    
    Returns:
        list: List of 81 digit images
    """
    digit_images = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess_image_for_digit_extraction(image.copy())
    for cell_rect in cell_rectangles:
        digit_images.append(extract_digit_from_cell(image, cell_rect, digit_size))
    return digit_images

def extract_largest_component_from_binary_image(image):
    """
    Extract the largest connected component from a binary image.
    
    Used to isolate main digit from noise in a cell.
    
    Args:
        image: Binary image
    
    Returns:
        np.ndarray: Image with only the largest component
    """
    image = image.astype('uint8')
    num_components, component_labels, component_stats, component_centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    component_sizes = component_stats[:, -1]
    
    if(len(component_sizes) <= 1):
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image
    
    largest_component_label = 1
    largest_component_size = component_sizes[1]
    
    for i in range(2, num_components):
        if component_sizes[i] > largest_component_size:
            largest_component_label = i
            largest_component_size = component_sizes[i]

    output_image = np.zeros(component_labels.shape)
    output_image.fill(255)
    output_image[component_labels == largest_component_label] = 0
    
    return output_image

def calculate_angle_between_vectors(vector_1, vector_2):
    """
    Calculate angle in degrees between two vectors.
    
    Args:
        vector_1: First vector
        vector_2: Second vector
    
    Returns:
        float: Angle in degrees
    """
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle_radians = np.arccos(dot_product)
    return math.degrees(angle_radians)

def check_if_quadrilateral_sides_are_similar(point_A, point_B, point_C, point_D, tolerance):
    """
    Check if all sides of a quadrilateral have similar lengths.
    
    Args:
        point_A, point_B, point_C, point_D: Corner points
        tolerance: Maximum ratio of longest to shortest side
    
    Returns:
        bool: True if sides differ beyond tolerance, False if similar
    """
    side_AB = math.sqrt((point_A[0]-point_B[0])**2 + (point_A[1]-point_B[1])**2)
    side_AD = math.sqrt((point_A[0]-point_D[0])**2 + (point_A[1]-point_D[1])**2)
    side_BC = math.sqrt((point_B[0]-point_C[0])**2 + (point_B[1]-point_C[1])**2)
    side_CD = math.sqrt((point_C[0]-point_D[0])**2 + (point_C[1]-point_D[1])**2)
    shortest_side = min(side_AB, side_AD, side_BC, side_CD)
    longest_side = max(side_AB, side_AD, side_BC, side_CD)
    return longest_side > tolerance * shortest_side

def is_angle_approximately_90_degrees(angle, tolerance):
    """
    Check if an angle is approximately 90 degrees.
    
    Args:
        angle: Angle in degrees
        tolerance: Acceptable deviation from 90 degrees
    
    Returns:
        bool: True if within tolerance of 90 degrees
    """
    return abs(angle - 90) < tolerance

def find_largest_contour(contours):
    """
    Find the contour with the largest area.
    
    Args:
        contours: List of contours
    
    Returns:
        tuple: (max_area, max_contour)
    """
    max_area = -1
    max_contour = None
  
    for i in range(len(contours)):
        current_contour = contours[i]
        area = cv2.contourArea(current_contour)
        if area > max_area:
            max_area = area
            max_contour = current_contour
   
    return max_area, max_contour

def approximate_contour_to_polygon(contour, num_corners=4):
    """
    Approximate a contour to a polygon with specified number of corners.
    
    Args:
        contour: Input contour
        num_corners: Desired number of corners (default: 4)
    
    Returns:
        np.ndarray or None: Corner points if found, None otherwise
    """
    approximation_coefficient = 1
    max_iterations = 300
    while max_iterations > 0 and approximation_coefficient >= 0:
        max_iterations = max_iterations - 1
        epsilon = approximation_coefficient * cv2.arcLength(contour, True)
        polygon_approximation = cv2.approxPolyDP(contour, epsilon, True)
        convex_hull = cv2.convexHull(polygon_approximation)
   
        if len(convex_hull) == num_corners:
            return convex_hull
        else:
            if len(convex_hull) > num_corners:
                approximation_coefficient += .01
            else:
                approximation_coefficient -= .01
                
    return None

def order_corner_points_clockwise(corners):
    """
    Order four corner points in clockwise order starting from top-left.
    
    Returns corners as [top_left, top_right, bottom_right, bottom_left].
    
    Args:
        corners: Array of 4 corner points in any order
    
    Returns:
        np.ndarray: Ordered corner points (4, 2)
    """
    max_sum = 1000000
    min_sum = 0
    ordered_corners = np.zeros((4, 2), dtype = "float32")
    current_index = -1
    
    # Top Left Corner has smallest sum of coordinates
    for i in range(len(corners)):
        coordinate_sum = corners[i][0] + corners[i][1]
        if coordinate_sum < max_sum:
            max_sum = coordinate_sum
            current_index = i
    ordered_corners[0] = corners[current_index]
    corners = np.delete(corners, current_index, 0)
    
    # Bottom Right Corner has biggest sum of coordinates
    for i in range(len(corners)):
        coordinate_sum = corners[i][0] + corners[i][1]
        if coordinate_sum > min_sum:
            min_sum = coordinate_sum
            current_index = i
    ordered_corners[2] = corners[current_index]
    corners = np.delete(corners, current_index, 0)
    
    # Of remaining two: larger x is top-right, smaller x is bottom-left
    if(corners[0][0] > corners[1][0]):
        ordered_corners[1] = corners[0]
        ordered_corners[3] = corners[1]
    else:
        ordered_corners[1] = corners[1]
        ordered_corners[3] = corners[0]

    ordered_corners = ordered_corners.reshape(4,2)
    return ordered_corners

def normalize_image_for_model(image_array):
    """
    Normalize image array for model prediction.
    
    Args:
        image_array: Image array to normalize
    
    Returns:
        np.ndarray: Normalized array ready for prediction
    """
    normalized_array = image_array.reshape(-1, 28, 28, 1)
    normalized_array = normalized_array.astype('float32')
    normalized_array /= 255
    return normalized_array

def convert_digit_images_to_sudoku_grid(digit_grid_image, model):
    """
    Convert a 252x252 image of 81 digits into a 9x9 Sudoku grid.
    
    Recognizes digits using the trained model and populates the grid.
    
    Args:
        digit_grid_image: 252x252 image containing all 81 digits
        model: Trained digit recognition model
    
    Returns:
        list: 9x9 grid with recognized digits
    """
    # Create empty 9x9 grid
    grid = create_empty_sudoku_grid()

    # Store digits to predict and their positions
    digits_to_predict = []
    grid_positions = []

    # Process all 81 cells
    for i in range(9):
        for j in range(9):
            # Extract 28x28 cell image
            cell_image = digit_grid_image[i*28:i*28+28, j*28:j*28+28]

            # Check if cell contains a digit
            if np.sum(cell_image) > 0:
                # Binarize the image
                _, cell_image = cv2.threshold(cell_image, 200, 255, cv2.THRESH_BINARY)

                # Ensure correct format
                cell_image = cell_image.astype(np.uint8)

                # Center the digit
                shift_x, shift_y = calculate_centering_shift(cell_image)
                cell_image = apply_shift(cell_image, shift_x, shift_y)

                # Invert image (black digit on white background)
                cell_image = cv2.bitwise_not(cell_image)

                # Normalize and reshape
                cell_image = cell_image.astype('float32') / 255.0
                cell_image = cell_image.reshape(28, 28, 1)

                # Add to batch
                digits_to_predict.append(cell_image)
                grid_positions.append((i, j))

    # Batch predict all digits
    if digits_to_predict:
        batch = np.array(digits_to_predict)
        predictions = model.predict(batch, verbose=0)

        # Assign predictions to grid
        for idx, prediction in enumerate(predictions):
            i, j = grid_positions[idx]
            grid[i][j] = np.argmax(prediction) + 1

    return grid
    
def validate_quadrilateral_is_square_like(corner_points):
    """
    Validate that a quadrilateral is approximately square-shaped.
    
    Checks that all corners are ~90 degrees and sides are similar length.
    
    Args:
        corner_points: Four corner points [top_left, top_right, bottom_right, bottom_left]
    
    Returns:
        bool: True if square-like, False otherwise
    """
    point_A = corner_points[0]
    point_B = corner_points[1]
    point_C = corner_points[2]
    point_D = corner_points[3]
    
    # Calculate edge vectors
    vector_AB = point_B - point_A
    vector_AD = point_D - point_A
    vector_BC = point_C - point_B
    vector_DC = point_C - point_D
    
    # Calculate angles between vectors
    angle_AB_AD = calculate_angle_between_vectors(vector_AB, vector_AD)
    angle_AB_BC = calculate_angle_between_vectors(vector_AB, vector_BC)
    angle_BC_DC = calculate_angle_between_vectors(vector_BC, vector_DC)
    angle_AD_DC = calculate_angle_between_vectors(vector_AD, vector_DC)
    
    # Check if all corners are approximately 90 degrees
    if not (is_angle_approximately_90_degrees(angle_AB_AD, 15) and is_angle_approximately_90_degrees(angle_AB_BC, 15) and 
            is_angle_approximately_90_degrees(angle_BC_DC, 15) and is_angle_approximately_90_degrees(angle_AD_DC, 15)):
        return False
    
    # Check if sides have similar length
    if check_if_quadrilateral_sides_are_similar(point_A, point_B, point_C, point_D, 1.2):
        return False
    
    return True

def create_empty_sudoku_grid():
    """
    Create an empty 9x9 Sudoku grid filled with zeros.
    
    Returns:
        list: 9x9 grid initialized with zeros
    """
    grid = []
    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            row.append(0)
        grid.append(row)
    return grid
        
def overlay_solution_on_warped_grid(image, solved_grid, original_grid):
    """
    Draw the Sudoku solution on the warped grid image.
    
    Only draws digits that were originally empty (zeros in original_grid).
    
    Args:
        image: Warped grid image to draw on
        solved_grid: Complete solved Sudoku grid
        original_grid: Original grid with clues (to identify empty cells)
    
    Returns:
        None (modifies image in place)
    """
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if original_grid[i][j] != 0:
                continue
            else:
                digit_to_draw = solved_grid[i][j]
                cell_height = image.shape[0]/9
                cell_width = image.shape[1]/9
                digit_x_position = cell_height*i + cell_height/2
                digit_y_position = cell_width*j + cell_width/2
                cv2.putText(image, str(digit_to_draw), (int(digit_y_position)-5,int(digit_x_position)+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) 
    
def create_visualization_grid_from_digits(digit_images):
    """
    Create a visualization of all 81 digits in a 9x9 grid format.
    
    Args:
        digit_images: List of 81 digit images (28x28 each)
    
    Returns:
        np.ndarray: Combined 252x252 image showing all digits
    """
    grid_rows = []
    for i in range(9):
        row_of_digits = np.concatenate(digit_images[i * 9:((i + 1) * 9)], axis=0)
        grid_rows.append(row_of_digits)
    combined_grid = np.concatenate(grid_rows, axis=1)
    return combined_grid

def extract_and_solve_sudoku(frame, model, cached_solved_grid):
    """
    Main pipeline to detect, extract, solve, and overlay Sudoku puzzle on video frame.
    
    Processing steps:
    1. Convert to grayscale and apply thresholding
    2. Find largest contour (likely the Sudoku grid)
    3. Find and order 4 corners
    4. Validate it's square-shaped
    5. Apply perspective transform
    6. Extract and recognize digits
    7. Solve the puzzle
    8. Overlay solution back on original frame
    
    Args:
        frame: Input video frame
        model: Trained digit recognition model
        cached_solved_grid: Previously solved grid (to reuse if puzzle unchanged)
    
    Returns:
        tuple: (processed_frame, cached_solved_grid)
    """
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (9,9), 0)
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    puzzle_area, largest_contour = find_largest_contour(contours)
    
    if largest_contour is not None:
        corner_points = approximate_contour_to_polygon(largest_contour)
        if corner_points is not None:
            corner_points = corner_points.reshape(4,2)
            ordered_corners = order_corner_points_clockwise(corner_points)
            if validate_quadrilateral_is_square_like(ordered_corners):
                (top_left, top_right, bottom_right, bottom_left) = ordered_corners
                bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
                top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
                right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
                left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
                
                max_width = max(int(bottom_width), int(top_width))
                max_height = max(int(right_height), int(left_height))
                ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
                destination_points = np.array([
            		[0, 0],
            		[max_width - 1, 0],
            		[max_width - 1, max_height - 1],
            		[0, max_height - 1]], dtype = "float32")
                
                perspective_transform = cv2.getPerspectiveTransform(ordered_corners, destination_points)
                warped_grid = cv2.warpPerspective(frame, perspective_transform, (max_width, max_height))
                warped_grid_copy = warped_grid.copy()
                cell_rectangles = calculate_cell_boundaries(warped_grid_copy)
                digit_images = extract_all_digits_from_grid(warped_grid_copy, cell_rectangles, 28)
                digit_grid_visualization = create_visualization_grid_from_digits(digit_images)
                recognized_grid = convert_digit_images_to_sudoku_grid(digit_grid_visualization, model)
                original_clues_grid = [row.copy() for row in recognized_grid]
                
                # Process Sudoku puzzle
                if(puzzle_area > 65000):
                    if(np.count_nonzero(original_clues_grid) >= 17):
                        if (cached_solved_grid is not None):
                            if(Solver.is_solved(cached_solved_grid)):
                                overlay_solution_on_warped_grid(warped_grid_copy, cached_solved_grid, original_clues_grid)
                        else:
                            _, solved_grid = Solver.solve_sudoku(recognized_grid)
                            if(Solver.is_solved(solved_grid)):
                                overlay_solution_on_warped_grid(warped_grid_copy, solved_grid, original_clues_grid)
                                cached_solved_grid = [row.copy() for row in solved_grid]
                    else:
                        return frame, None
                
                    warped_solution = cv2.warpPerspective(warped_grid_copy, perspective_transform, (frame.shape[1], frame.shape[0])
                                            , flags=cv2.WARP_INVERSE_MAP)
                    result_frame = np.where(warped_solution.sum(axis=-1,keepdims=True)!=0, warped_solution, frame)
                    
                    # Draw corner markers
                    cv2.circle(result_frame, (int(ordered_corners[0][0]), int(ordered_corners[0][1])), 5, (0,0,255), 5)
                    cv2.circle(result_frame, (int(ordered_corners[1][0]), int(ordered_corners[1][1])), 5, (0,0,255), 5)
                    cv2.circle(result_frame, (int(ordered_corners[2][0]), int(ordered_corners[2][1])), 5, (0,0,255), 5)
                    cv2.circle(result_frame, (int(ordered_corners[3][0]), int(ordered_corners[3][1])), 5, (0,0,255), 5)
                    
                    # Draw contour outline
                    cv2.drawContours(result_frame, [largest_contour], 0,  (0,255,0), 3)
                
                    return result_frame, cached_solved_grid
                else:
                    return frame, None
            else:
                return frame, None
        else:
            return frame, None
    else:
        return frame, None