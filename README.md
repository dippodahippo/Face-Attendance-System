# Face Recognition Attendance System

This Python program uses OpenCV for face detection and recognition to create a simple attendance system. It allows users to recognize a person, capture and store a new face, and update the recognition model.

## Requirements

- Python 3
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- MediaPipe (`pip install mediapipe`)

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/face-attendance-system.git
   cd face-attendance-system
   
Run the program:

bash
Copy code
python facedetection.py
Follow the on-screen instructions:

Press 'r' to recognize a person.
Press 's' to capture and store a new face.
Press 'q' to quit the program.

# Features
Face recognition using the LBPH algorithm.

Automatic face capturing and updating of the recognition model.

Real-time recognition with webcam feed.

# File Structure
facedetection.py: The main Python script.

facedata.yml: YAML file to store trained face recognition data.

capture_count.txt: Text file to keep track of the number of faces stored.

Credits

OpenCV for providing computer vision libraries.
NumPy for numerical computing.


