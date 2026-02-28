"""
Real-time Sudoku Solver using Computer Vision and Machine Learning.

This module captures video from a webcam, detects Sudoku puzzles in the frame,
extracts the grid, recognizes digits using a trained CNN model, solves the puzzle
using a backtracking algorithm, and overlays the solution back onto the video feed.
"""

import cv2
import keras
import SudokuExtractor

# Camera configuration constants
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Display window constants
WINDOW_NAME = "Sudoku Solver - Real-time"
QUIT_KEY = 'q'

# Model file path
MODEL_PATH = 'digit_model.h5'


def initialize_camera(camera_index=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """
    Initialize and configure the camera for video capture.

    Args:
        camera_index (int): Index of the camera device to use (default: 0 for primary camera).
        width (int): Desired frame width in pixels.
        height (int): Desired frame height in pixels.

    Returns:
        cv2.VideoCapture: Configured camera object.

    Raises:
        RuntimeError: If camera cannot be opened.
    """
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        raise RuntimeError(f"Failed to open camera at index {camera_index}")
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    return camera


def load_digit_recognition_model(model_path=MODEL_PATH):
    """
    Load the pre-trained digit recognition model.

    Args:
        model_path (str): Path to the saved Keras model file.

    Returns:
        keras.Model: Loaded digit recognition model.

    Raises:
        IOError: If model file cannot be loaded.
    """
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise IOError(f"Failed to load model from {model_path}: {str(e)}")


def process_video_stream(camera, model):
    """
    Process video stream to detect, solve, and display Sudoku puzzles in real-time.

    Args:
        camera (cv2.VideoCapture): Initialized camera object.
        model (keras.Model): Pre-trained digit recognition model.

    Returns:
        None
    """
    cached_solution = None  # Store the solution to avoid redundant computations
    
    while True:
        frame_captured, frame = camera.read()
        
        if not frame_captured:
            print("Warning: Failed to capture frame from camera")
            break
        
        # Extract sudoku grid, solve it, and overlay solution on frame
        processed_frame, cached_solution = SudokuExtractor.extract_and_solve_sudoku(
            frame, model, cached_solution
        )
        
        # Display the processed frame
        cv2.imshow(WINDOW_NAME, processed_frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord(QUIT_KEY):
            print("User requested exit")
            break


def cleanup_resources(camera):
    """
    Release camera resources and close all OpenCV windows.

    Args:
        camera (cv2.VideoCapture): Camera object to release.

    Returns:
        None
    """
    camera.release()
    cv2.destroyAllWindows()


def main():
    """
    Main function to run the real-time Sudoku solver application.

    Initializes camera, loads model, processes video stream, and handles cleanup.

    Returns:
        None
    """
    try:
        print("Initializing Sudoku Solver...")
        camera = initialize_camera()
        print(f"Camera initialized with resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
        
        print(f"Loading digit recognition model from {MODEL_PATH}...")
        model = load_digit_recognition_model()
        print("Model loaded successfully")
        
        print(f"Starting video stream... Press '{QUIT_KEY}' to quit")
        process_video_stream(camera, model)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        if 'camera' in locals():
            cleanup_resources(camera)
            print("Resources cleaned up")


if __name__ == "__main__":
    main()