# 🧩 Real-Time Sudoku Solver

A real-time Sudoku solver that uses your webcam to detect a Sudoku puzzle, recognize its digits with a Convolutional Neural Network, solve it using a backtracking algorithm, and overlay the solution directly onto the live video feed.

---

## 📋 Table of Contents

- [Demo Overview](#demo-overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation & Setup](#installation--setup)
- [Running the Program](#running-the-program)
- [Training the Model](#training-the-model)
- [References](#references)

---

## 🎯 Demo Overview

Point your camera at a Sudoku puzzle and the program will:

1. Detect the puzzle grid in real time
2. Extract and recognize each digit
3. Solve the puzzle
4. Overlay the solution in red on the empty cells — directly on the video feed

---

## ⚙️ How It Works

The program is a full computer vision + machine learning pipeline. Here is each stage explained:

### 1. 🔍 Puzzle Detection

**File:** `SudokuExtractor.py` — `extract_and_solve_sudoku()`

Each video frame is processed as follows:

- Convert to **grayscale**
- Apply **Gaussian blur** to reduce noise
- Apply **adaptive thresholding** (inverted) to create a binary image highlighting edges
- Find all **contours** in the image using `cv2.findContours`
- Select the **largest contour** by area — this is assumed to be the Sudoku grid

The largest contour is then approximated to a **4-corner polygon** (using iterative `approxPolyDP` + `convexHull`). The grid is only accepted if the shape is **approximately square** — all 4 angles must be close to 90° and all 4 sides must have similar lengths.

### 2. 📐 Perspective Correction

**File:** `SudokuExtractor.py` — `extract_and_solve_sudoku()`

Once the 4 corners are identified and ordered (top-left, top-right, bottom-right, bottom-left), a **perspective transform** is applied using `cv2.getPerspectiveTransform` + `cv2.warpPerspective`.

This produces a flat, top-down, distortion-free image of the Sudoku grid regardless of the camera angle.

### 3. ✂️ Cell Extraction

**File:** `SudokuExtractor.py` — `calculate_cell_boundaries()`, `extract_all_digits_from_grid()`

The warped grid image is divided into a **9×9 grid of 81 cells**. For each cell:

- The image is converted to grayscale and preprocessed with adaptive thresholding
- The **largest connected white region** in the center area of each cell is located using flood fill — this isolates the digit from grid lines and noise
- The digit region is scaled and centered into a clean **28×28 pixel image**

Cells with no significant content are classified as empty (value = 0).

### 4. 🤖 Digit Recognition

**File:** `SudokuExtractor.py` — `convert_digit_images_to_sudoku_grid()`  
**Model:** `digit_model.h5`

All non-empty cells are collected into a batch and fed to a **pre-trained CNN** (Convolutional Neural Network). Before prediction, each digit image is:

- Binarized with thresholding
- Centered using **center-of-mass shift** (from `scipy.ndimage`)
- Inverted (black digit on white background)
- Normalized to `[0, 1]`

The model outputs a probability distribution over digits 1–9 and `argmax` selects the predicted digit.

#### CNN Architecture (`DigitRecognition.py`):

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3×3, ReLU |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Dropout | 25% |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dropout | 50% |
| Dense (output) | 9 units, Softmax |

> The model was trained on the **Chars74k dataset** (digits 1–9 only). See [References](#references).

### 5. 🧠 Solving the Puzzle

**File:** `Solver.py` — `solve_sudoku()`

The recognized 9×9 grid is solved using a **recursive backtracking algorithm**:

1. Find the next empty cell
2. Try digits 1–9, checking Sudoku validity (row, column, 3×3 box)
3. Recurse if valid; backtrack if no digit works
4. Return the solved grid when all cells are filled

> A valid Sudoku puzzle requires a minimum of **17 clues** to guarantee a unique solution — the program enforces this check before attempting to solve.

Once solved, the result is **cached** so the solver doesn't run on every frame — it reuses the solution as long as the same puzzle is detected.

### 6. 🖥️ Overlaying the Solution

**File:** `SudokuExtractor.py` — `overlay_solution_on_warped_grid()`

The solution digits are drawn in **red** onto the warped grid image, only on cells that were originally empty. The solved image is then **inverse perspective-warped** back to the original camera view and blended with the original frame using `np.where`.

Additionally, the 4 detected corners are marked with red circles and the puzzle contour is drawn in green.

---

## 📁 Project Structure

```
SudokuSolver/
│
├── main.py                 # Entry point — camera loop and orchestration
├── SudokuExtractor.py      # Computer vision pipeline (detection, extraction, overlay)
├── Solver.py               # Backtracking Sudoku solver algorithm
├── DigitRecognition.py     # CNN model definition and training script
│
├── digit_model.h5          # Pre-trained digit recognition model
├── requirements.txt        # Full conda environment specification
│
└── dataset/                # Training images (Chars74k, digits 1–9)
    ├── 1/
    ├── 2/
    ...
    └── 9/
```

---

## 📦 Dependencies

| Library | Version |
|---|---|
| Python | 3.8.20 |
| OpenCV | 4.11.0 |
| Keras | 2.13.1 |
| TensorFlow | 2.13.0 |
| NumPy | 1.24.3 |
| SciPy | 1.10.1 |

---

## 🛠️ Installation & Setup

### Option A — Conda (recommended)

```bash
conda create --name sudoku --file requirements.txt
conda activate sudoku
```

### Option B — pip

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python==4.11.0 keras==2.13.1 tensorflow==2.13.0 numpy==1.24.3 scipy==1.10.1
```

---

## ▶️ Running the Program

Make sure your webcam is connected, then run:

```bash
python main.py
```

- A window named **"Sudoku Solver - Real-time"** will open with the live camera feed
- Point the camera at a printed or displayed Sudoku puzzle
- The program will detect the grid, solve it, and overlay the solution in **red**
- Press **`q`** to quit

> **Tip:** For best results, ensure good lighting and hold the puzzle flat and steady until the first solution is detected. After that, the solution is cached and stays locked even if you move the camera.

---

## 🏋️ Training the Model

If you want to retrain the CNN on your own dataset:

1. Place digit images (grayscale) in `dataset/1/`, `dataset/2/`, ..., `dataset/9/`
2. Run:

```bash
python DigitRecognition.py
```

The script will load the dataset, split it 80/20 into train/test sets, train for 35 epochs, evaluate accuracy, and save the model as `digit_model_v2.h5`.

---

## 🔗 References

- **Dataset used to train CNN:** [Chars74k — University of Surrey](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)
- **Sudoku backtracking algorithm:** [GeeksForGeeks — Sudoku Backtracking](https://www.geeksforgeeks.org/sudoku-backtracking-7/)
- **Perspective transform:** [PyImageSearch — 4-Point getPerspectiveTransform](https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/)
- **Sudoku grid extraction:** [Nesh Patel — Solving Sudoku Part II](https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2)