"""
Digit Recognition Model Training using Convolutional Neural Network.

This module trains a CNN model to recognize handwritten digits (1-9) from Sudoku puzzles.
It loads digit images from a dataset, preprocesses them (centering by center of mass),
splits into train/test sets, and trains a CNN model using Keras.

The trained model is saved and can be used for real-time digit recognition in Sudoku grids.
"""

import os
import cv2
import numpy as np
import random 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from scipy import ndimage

# Dataset configuration
DATASET_PATH = "dataset"
DIGIT_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Model architecture constants
IMAGE_SIZE = 28
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
NUM_CLASSES = 9

# Training configuration
TRAIN_TEST_SPLIT_RATIO = 0.8
VALIDATION_SPLIT_RATIO = 0.1
BATCH_SIZE = 128
TRAINING_EPOCHS = 35

# Model output
MODEL_OUTPUT_PATH = 'digit_model_v2.h5'

def calculate_centering_shift(image):
    """
    Calculate the shift needed to center an image based on its center of mass.

    Args:
        image (np.ndarray): 2D grayscale image array.

    Returns:
        tuple: (shift_x, shift_y) - pixel shifts needed to center the image.
    """
    center_y, center_x = ndimage.measurements.center_of_mass(image)

    rows, cols = image.shape
    shift_x = np.round(cols / 2.0 - center_x).astype(int)
    shift_y = np.round(rows / 2.0 - center_y).astype(int)

    return shift_x, shift_y

def apply_shift(image, shift_x, shift_y):
    """
    Apply a translation shift to an image.

    Args:
        image (np.ndarray): 2D grayscale image array.
        shift_x (int): Horizontal shift in pixels.
        shift_y (int): Vertical shift in pixels.

    Returns:
        np.ndarray: Shifted image.
    """
    rows, cols = image.shape
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return shifted_image

def center_image_by_mass(image):
    """
    Center an image based on its center of mass.

    This function inverts the image (to make digit pixels have high values),
    calculates the center of mass, shifts the image to center it, and inverts back.

    Args:
        image (np.ndarray): 2D grayscale image array.

    Returns:
        np.ndarray: Centered image.
    """
    inverted_image = cv2.bitwise_not(image)

    # Calculate shift needed to center the digit
    shift_x, shift_y = calculate_centering_shift(inverted_image)
    centered_image = apply_shift(inverted_image, shift_x, shift_y)

    # Invert back to original representation
    result_image = cv2.bitwise_not(centered_image)
    return result_image

def load_dataset():
    """
    Load and preprocess digit images from the dataset directory.

    Loads images from subdirectories named "1" through "9", resizes them to 28x28 pixels,
    centers them by center of mass, and labels them with their corresponding digit class.

    Returns:
        list: List of [image, label] pairs where image is 28x28 numpy array and label is 0-8.
    """
    dataset = []
    
    for label in DIGIT_LABELS:
        digit_folder_path = os.path.join(DATASET_PATH, label)
        class_index = DIGIT_LABELS.index(label)
        
        for image_filename in os.listdir(digit_folder_path):
            # Load image in grayscale
            image_path = os.path.join(digit_folder_path, image_filename)
            image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Resize to standard size
            resized_image = cv2.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE))
            
            # Center the digit within the image
            centered_image = center_image_by_mass(resized_image)
            
            dataset.append([centered_image, class_index])
            
    return dataset

def split_dataset(dataset):
    """
    Split dataset into training and testing sets.

    Randomly shuffles the dataset and splits it according to TRAIN_TEST_SPLIT_RATIO.
    Default is 80% training, 20% testing.

    Args:
        dataset (list): List of [image, label] pairs.

    Returns:
        tuple: (train_images, train_labels, test_images, test_labels)
    """
    random.shuffle(dataset)
    
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    split_index = int(len(dataset) * TRAIN_TEST_SPLIT_RATIO)
    
    # Create training set
    for i in range(split_index):
        train_images.append(dataset[i][0])
        train_labels.append(dataset[i][1])

    # Create testing set
    for i in range(split_index, len(dataset)):
        test_images.append(dataset[i][0])
        test_labels.append(dataset[i][1])
    
    return train_images, train_labels, test_images, test_labels

def reshape_images(train_images, test_images):
    """
    Reshape image arrays to include channel dimension for CNN input.

    Converts list of 2D images to 4D numpy arrays with shape (samples, height, width, channels).

    Args:
        train_images (list): List of training images.
        test_images (list): List of testing images.

    Returns:
        tuple: (reshaped_train_images, reshaped_test_images) as 4D numpy arrays.
    """
    train_images_array = np.array(train_images)
    train_images_array = train_images_array.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    
    test_images_array = np.array(test_images)
    test_images_array = test_images_array.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    
    return train_images_array, test_images_array

def preprocess_data(train_images, train_labels, test_images, test_labels):
    """
    Preprocess training and testing data for neural network training.

    Reshapes images to include channel dimension, converts to float32,
    normalizes pixel values to [0, 1], and converts labels to categorical format.

    Args:
        train_images (list): Training images.
        train_labels (list): Training labels (0-8).
        test_images (list): Testing images.
        test_labels (list): Testing labels (0-8).

    Returns:
        tuple: (train_images, train_labels, test_images, test_labels) - all preprocessed.
    """
    # Reshape to add channel dimension
    train_images, test_images = reshape_images(train_images, test_images)
    
    # Convert to float32
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    
    # Normalize pixel values to [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # Convert labels to categorical (one-hot encoding)
    train_labels = keras.utils.to_categorical(train_labels, NUM_CLASSES)
    test_labels = keras.utils.to_categorical(test_labels, NUM_CLASSES)
    
    return train_images, train_labels, test_images, test_labels
    
def build_and_train_model(train_images, train_labels, test_images, test_labels):
    """
    Build, compile, train, and save a CNN model for digit recognition.

    Architecture:
    - 2 Convolutional layers (32 and 64 filters, 3x3 kernel, ReLU activation)
    - MaxPooling layer (2x2)
    - Dropout layer (25%)
    - Flatten layer
    - Dense layer (128 units, ReLU activation)
    - Dropout layer (50%)
    - Output layer (9 units, softmax activation)

    Args:
        train_images (np.ndarray): Preprocessed training images.
        train_labels (np.ndarray): One-hot encoded training labels.
        test_images (np.ndarray): Preprocessed testing images.
        test_labels (np.ndarray): One-hot encoded testing labels.

    Returns:
        None. Saves trained model to disk.
    """
    # Build model architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=INPUT_SHAPE))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    # Compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    # Train model
    print(f"Training model for {TRAINING_EPOCHS} epochs...")
    model.fit(train_images, train_labels,
              batch_size=BATCH_SIZE,
              epochs=TRAINING_EPOCHS,
              verbose=1,
              validation_split=VALIDATION_SPLIT_RATIO)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model
    model.save(MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")

def main():
    """
    Main function to train the digit recognition model.

    Loads dataset, splits into train/test sets, preprocesses data,
    builds and trains the CNN model, and saves it to disk.
    """
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} images")
    
    print("\nSplitting dataset into train/test sets...")
    train_images, train_labels, test_images, test_labels = split_dataset(dataset)
    print(f"Training samples: {len(train_images)}")
    print(f"Testing samples: {len(test_images)}")
    
    print("\nPreprocessing data...")
    train_images, train_labels, test_images, test_labels = preprocess_data(
        train_images, train_labels, test_images, test_labels
    )
    
    print("\nBuilding and training model...")
    build_and_train_model(train_images, train_labels, test_images, test_labels)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
    