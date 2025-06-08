# Drowsiness Detection System

A real-time drowsiness detection system using computer vision and facial landmarks detection. This system monitors a person's eyes to detect signs of drowsiness and alerts when drowsy behavior is detected.

## Features

- Real-time face detection
- Eye tracking and blink detection
- Drowsiness alert system
- Accuracy monitoring
- Frame capture and storage

## Prerequisites

- Python 3.x
- OpenCV
- dlib
- imutils
- scipy

## Required Files

- `shape_predictor_68_face_landmarks.dat`: Facial landmark predictor file from dlib
  - You can download this file from dlib's official website or model repository

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install opencv-python dlib imutils scipy
```
3. Place the `shape_predictor_68_face_landmarks.dat` file in the project root directory

## Usage

Run the program using:
```bash
python drowsiness.py
```

- Press 'q' to quit the application
- The system will display a live feed with eye tracking
- Drowsiness alerts will appear when prolonged eye closure is detected
- Frames are saved in the 'frames' directory
- Real-time accuracy is displayed on the screen

## Configuration

You can adjust the following parameters in `drowsiness.py`:
- `EAR_THRESH`: Eye Aspect Ratio threshold (default: 0.3)
- `EAR_CONSEC_FRAMES`: Number of consecutive frames for drowsiness detection (default: 30)

## License

This project is open source and available under the MIT License.
