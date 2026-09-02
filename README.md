# Sign Language Detection

A computer vision and machine learning project for detecting
hand signs representing English letters (A-Z) and numbers (0-9).

## Project Status

Current progress: **Phase 1-4 Completed**

- [x] Phase 1 - Webcam Setup
- [x] Phase 2 - Hand Detection and Feature Extraction
- [x] Phase 3 - Dataset Collection
- [x] Phase 4 - Data Preprocessing
- [x] Phase 5 - Machine Learning Model
- [ ] Phase 6 - Real-Time Sign Prediction
- [ ] Phase 7 - Final Integration

---

## Project Overview

The system uses a webcam to capture a user's hand and MediaPipe
Hand Landmarker to detect 21 hand landmarks.

The landmark coordinates are converted into numerical features
which are then used to build a machine learning dataset.

The final goal is to recognize:

- English letters: A-Z
- Numbers: 0-9

Total classes: **36**

---

## Phase 1 - Webcam Setup

Configured the webcam using OpenCV.

The DirectShow backend was used on Windows to provide reliable
webcam access.

---

## Phase 2 - Hand Detection and Feature Extraction

Implemented hand detection using the MediaPipe Tasks API.

The system detects 21 hand landmarks.

Each landmark contains:

- X coordinate
- Y coordinate
- Z coordinate

This produces:

21 × 3 = **63 features**

The coordinates are normalized relative to the wrist landmark
to make the features less dependent on hand position.

---

## Phase 3 - Dataset Collection

Collected a balanced dataset containing:

- A-Z: 26 classes
- 0-9: 10 classes
- Total: 36 classes
- Samples per class: 100
- Total samples: 3,600

Dataset shape:

```text
3600 rows × 64 columns

The 64 columns consist of:

63 landmark features
1 label column

Dataset validation confirmed:

No missing values
All features are numeric
100 samples per class
No duplicate samples
Phase 4 - Data Preprocessing

The preprocessing pipeline performs:

Dataset loading
Feature/label separation
Constant feature detection
Constant feature removal
Label encoding
Stratified train/test splitting
Constant Features

The original dataset contained 63 features.

The following constant features were detected:

f0
f1
f2

These correspond to the wrist coordinates after wrist-relative
normalization, which are always zero.

Therefore:

63 features → 60 features
Label Encoding

The 36 classes were encoded into numerical labels:

0-9 → 0-9
A-Z → 10-35
Train/Test Split

An 80/20 stratified split was used.

Training dataset:

2880 samples × 60 features

Testing dataset:

720 samples × 60 features

Each class contains:

80 training samples
20 testing samples

A reusable preprocessing function is implemented in:

src/preprocess_dataset.py
Dataset Structure
data/
└── sign_landmarks.csv
Project Structure
SignLanguageDetection/
│
├── data/
│   └── sign_landmarks.csv
│
├── models/
│
├── src/
│   ├── __init__.py
│   ├── analyze_dataset.py
│   ├── collect_dataset.py
│   ├── dataset_utils.py
│   ├── feature_extraction.py
│   ├── hand_detection.py
│   ├── preprocess_dataset.py
│   ├── test_constant_features.py
│   ├── test_dataset.py
│   ├── test_features.py
│   └── verify_dataset.py
│
├── tests/
│
├── .gitignore
├── requirements.txt
└── README.md
Installation

Clone the repository:

git clone <repository-url>

Create a virtual environment:

python -m venv venv

Activate it on Windows PowerShell:

.\venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
Dataset

The project currently contains 3,600 landmark samples
representing 36 classes.

Each sample contains 60 usable normalized landmark features
after preprocessing.

Technologies Used
Python
OpenCV
MediaPipe
NumPy
Pandas
Scikit-learn
Matplotlib
Future Work

The next major stage is Phase 5:

Machine Learning Model Development

This will include:

Model selection
Model training
Validation
Hyperparameter tuning
Model evaluation
Saving the trained model

After that, the trained model will be integrated with the
real-time webcam pipeline.
