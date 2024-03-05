import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def load_recognizer():
    if os.path.exists("facedata.yml") and os.path.getsize("facedata.yml") > 0:
        recognizer.read("facedata.yml")
        print("Face recognition model loaded successfully.")
    else:
        print("No pre-trained data found. Capture and store faces first.")

def save_recognizer():
    try:
        recognizer.save("facedata.yml")
        print("Face recognition model saved successfully.")
    except cv2.error as e:
        print(f"Error saving facedata.yml: {e}")

def capture_and_store_face():
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("No face detected.")
        return

    x, y, w, h = faces[0]
    roi_gray = gray[y:y+h, x:x+w]

    for i in range(30):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(roi_gray, (100, 100))
        cv2.imwrite("captured_face.jpg", roi_gray)

    print("Face captured and stored successfully.")

    cap.release()

while True:
    key = input("Press 'r' to recognize, 's' to capture and store, or 'q' to quit: ")

    if key == 'q':
        break

    elif key == 'r':
        load_recognizer()

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (100, 100))

                label, confidence = recognizer.predict(roi_gray)

                if confidence < 70:
                    print("Person recognized. Label:", label)
                else:
                    print("Recognition failed. Unknown person.")
        else:
            print("No face detected.")

        cap.release()

    elif key == 's':
        capture_and_store_face()

        images, labels = [], []
        captured_face = cv2.imread("captured_face.jpg", cv2.IMREAD_GRAYSCALE)
        images.append(captured_face)
        labels.append(1)  

        recognizer.update(np.asarray(images), np.asarray(labels))
        save_recognizer()

    else:
        print("Invalid input. Please try again.")
