# CS 5330 Assignment 3: Real-time 2D Object Recognition

**Student:** Rohil Kulshreshtha  
**Course:** CS 5330 - Pattern Recognition & Computer Vision  
**Semester:** Spring 2026  
**Submission Date:** February 22, 2026

---

## Project Overview

This project implements a complete real-time 2D object recognition system with two classification approaches:
1. **Hand-crafted geometric features** with K-NN classifier
2. **CNN embeddings** using pre-trained ResNet18

The system recognizes 10 object categories across translation, rotation, and scale variations.

---

## Video Demonstrations

### Demo 1: File-Based Object Detection
**URL:** https://northeastern.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=71e7eb5f-b452-449a-9f42-b3fa0054ed07

Demonstrates single-image processing showing all pipeline stages (threshold, cleanup, segmentation, features, classification).

### Demo 2: Real-Time Camera Detection  
**URL:** https://northeastern.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=2207ff54-7b43-45b9-90df-b3fa0054ed3a

Shows live camera feed with real-time object recognition and labeling.

---

## Development Environment

**Operating System:** Windows 11  
**IDE:** Visual Studio Code with CMake Tools extension  
**Compiler:** MSVC 19.45 (Visual Studio 2022 Build Tools)  
**Build System:** CMake 3.28  
**Libraries:**
- OpenCV 4.12.0 (with DNN module for ONNX support)
- C++17 standard library

**Python Environment (for visualizations):**
- Python 3.9
- NumPy, Pandas, Matplotlib, Scikit-learn

---

## Project Structure

```
Assignment3/
├── src/                          # Source files (.cpp)
│   ├── objectRecognition.cpp     # Main program (camera + image modes)
│   ├── train.cpp                 # Build hand-crafted feature database
│   ├── evaluate.cpp              # Evaluate hand-crafted classifier
│   ├── trainEmbedding.cpp        # Build CNN embedding database
│   ├── evaluateEmbedding.cpp     # Evaluate CNN classifier
│   ├── threshold.cpp             # Thresholding algorithms
│   ├── filter.cpp                # Gaussian blur
│   ├── kmeans.cpp                # K-means ISODATA
│   ├── morphology.cpp            # Erosion, dilation, opening, closing
│   ├── segmentation.cpp          # Two-pass connected components
│   ├── features.cpp              # Moment-based feature extraction
│   ├── database.cpp              # CSV database management
│   ├── classifier.cpp            # K-NN classifier
│   └── embedding.cpp             # ResNet18 embedding extraction
│
├── include/                      # Header files (.h)
│   └── [corresponding .h files for each .cpp]
│
├── data/                         # Datasets and databases
│   ├── training_images/          # 86 training images
│   ├── test_images/              # 30 test images
│   ├── object_database.csv       # Hand-crafted feature database
│   └── embedding_database.csv    # CNN embedding database
│
├── visualization/                # Python visualization scripts
│   ├── visualize_embedding.py    # 2D PCA plot for embeddings
│   ├── visualize_handcrafted.py  # 2D PCA plot for hand-crafted features
│   └── run_visualizations.py     # Run both visualizations
│
├── bin/                          # Compiled executables (after build)
│   ├── resnet18-v2-7.onnx        # Pre-trained ResNet18 model
│   └── [executables after build]
│
├── CMakeLists.txt                # CMake build configuration
├── README.md                     # This file
└── Report document               # Project report PDF
```

---

## Build Instructions

### Prerequisites

1. **Install OpenCV 4.5+** with DNN module enabled
   - Windows: Download pre-built binaries from opencv.org
   - Linux: `sudo apt-get install libopencv-dev`
   - macOS: `brew install opencv`

2. **Install CMake 3.15+**
   - Download from cmake.org or use package manager

3. **C++17 compatible compiler**
   - Windows: Visual Studio 2019+ or MSVC Build Tools
   - Linux: GCC 7+ or Clang 5+
   - macOS: Xcode Command Line Tools

### Build Steps

```bash
# Navigate to project directory
cd Assignment3

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build all executables
cmake --build . --config Release

# Executables will be in bin/ directory
```

---

## Running the Executables

### 1. Object Recognition (Main Program)

**Camera Mode (Real-time):**
```bash
./bin/objectRecognition
```
- Opens webcam and displays 3 windows: Original, Regions, Features
- Shows real-time object recognition with labels
- Press 'q' or ESC to quit

**Image Mode (Single image):**
```bash
./bin/objectRecognition path/to/image.jpg
```
- Processes single image and displays results
- Shows all pipeline stages
- Press any key to close windows

**Example:**
```bash
./bin/objectRecognition data/test_images/scissors_01.jpg
```

---

### 2. Training Programs

**Build Hand-Crafted Feature Database:**
```bash
./bin/train <training_directory> <output_csv>
```

**Example:**
```bash
./bin/train data/training_images data/object_database.csv
```
- Processes all .jpg/.png images in training directory
- Extracts labels from filenames (objectname_##.jpg)
- Saves 4D feature vectors to CSV
- Progress printed every 10 images

**Build CNN Embedding Database:**
```bash
./bin/trainEmbedding <training_directory> <model_path> <output_csv>
```

**Example:**
```bash
./bin/trainEmbedding data/training_images bin/resnet18-v2-7.onnx data/embedding_database.csv
```
- Processes all training images
- Extracts 512D embeddings using ResNet18
- Saves to CSV
- Requires resnet18-v2-7.onnx model in bin/ directory

---

### 3. Evaluation Programs

**Evaluate Hand-Crafted Features:**
```bash
./bin/evaluate <test_directory> <database_csv>
```

**Example:**
```bash
./bin/evaluate data/test_images data/object_database.csv
```
- Tests classifier on all images in test directory
- Outputs confusion matrix
- Displays per-image classification results
- Saves confusion matrix to `confusion_matrix_handcrafted.csv`

**Evaluate CNN Embeddings:**
```bash
./bin/evaluateEmbedding <test_directory> <model_path> <embedding_csv>
```

**Example:**
```bash
./bin/evaluateEmbedding data/test_images bin/resnet18-v2-7.onnx data/embedding_database.csv
```
- Tests embedding-based classifier
- Outputs confusion matrix
- Saves confusion matrix to `confusion_matrix_embeddings.csv`

---

## Extension Testing

### Extension 1: From-Scratch Implementation
**Verification:**
The following algorithms are implemented entirely from scratch (no high-level OpenCV functions):
1. HSV thresholding (threshold.cpp)
2. Morphological operations (morphology.cpp)
3. Connected components (segmentation.cpp)  
4. Feature extraction (features.cpp)

**To verify:** Review source code - all operations use manual pixel loops.

---

### Extension 2: Ten Object Categories
**Verification:**
```bash
# Check training database
# Should show 10 unique labels:
# book, earbud_case, glasses, marker, scissors, spork, stress_ball, usb, wallet, watch
```

---

### Extension 3: Feature Comparison
**Verification:**
```bash
# Run both evaluations
./bin/evaluate data/test_images data/object_database.csv
./bin/evaluateEmbedding data/test_images bin/resnet18-v2-7.onnx data/embedding_database.csv

# Compare accuracy outputs:
# Hand-crafted: 90.00%
# Embeddings: 96.67%
```

---

### Extension 4: 2D Embedding Visualization
**Requirements:**
```bash
pip install numpy pandas matplotlib scikit-learn opencv-python
```

**Run visualizations:**
```bash
cd visualization
python run_visualizations.py
```

**Output:**
- `visualization/results/embedding_visualization.png` - 2D PCA plot of CNN embeddings
- `visualization/results/handcrafted_visualization.png` - 2D PCA plot of hand-crafted features
- Console output shows cluster statistics and variance explained

**Individual scripts:**
```bash
python visualize_embedding.py      # CNN embeddings only
python visualize_handcrafted.py    # Hand-crafted features only
```

---

## Dataset Preparation

### Training Images
- **Format:** JPG or PNG
- **Naming:** `objectname_##.jpg` (e.g., `scissors_01.jpg`, `wallet_12.jpg`)
- **Background:** White paper for consistent thresholding
- **Quantity:** 8-10 images per object at different rotations
- **Setup:** Top-down view, good lighting, ~30° rotation between shots

### Test Images
- Same format and background as training
- Different poses/rotations from training set
- 3 images per object category minimum

---

## Troubleshooting

### Common Issues

**1. "Database not found"**
- Run training programs first to generate databases
- Check paths are relative to executable location

**2. "No regions detected"**
- Ensure white background (thresholds tuned for white)
- Check lighting (bright, even illumination)
- Verify object is darker than background

**3. OpenCV windows don't appear**
- Check OpenCV installation includes highgui module
- The number of operations running on a frame might be high so as to cause the window to pop up and close automatically

**4. Slow performance**
- Images automatically resized to max 800px
- Reduce processEveryN in code if needed
- Use Release build (not Debug) for better performance

---

## Known Limitations

1. **Background Sensitivity:** System tuned for white paper background; performance degrades on colored backgrounds
2. **Reflective Objects:** Metal/glass objects may fragment due to specular reflections  
3. **Small Objects:** Minimum area threshold (500-1000 pixels) filters very small objects
4. **Lighting:** Requires even, bright lighting for consistent thresholding
5. **Similar Objects:** Hand-crafted features confuse USB ↔ watch (resolved with embeddings)

---
