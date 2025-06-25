import cv2 as cv 
import numpy as np
import dlib
import playsound as ps
from threading import Thread as T
from scipy.spatial import distance as dist
import face_recognition as fr

# Constants for eye aspect ratio (EAR) and related parameters
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 20

# Sound alarm file
ALARM_FILE = "alarm.wav"

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_file = 'haarcascade_frontalface_default.xml'
cascade = cv.CascadeClassifier(face_file)


# Function to play alarm sound
def sound_alarm(source):
    ps.playsound(source)

# Function to calculate EAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Main function to detect drowsiness
def detect_drowsiness():
    counter = 0
    alarm_triggered = False

    v_c = cv.VideoCapture(0)
    while True:
        ret, frame = v_c.read()
        if not ret:
            break

        # Convert frame to grayscale for better processing
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = detector(gray, 0)
        
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = []
            right_eye = []

            # Extract left and right eye coordinates from facial landmarks
            for i in range(36, 42):
                left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
            for i in range(42, 48):
                right_eye.append((landmarks.part(i).x, landmarks.part(i).y))

            # Calculate EAR for left and right eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # Average EAR of both eyes
            ear = (left_ear + right_ear) / 2

            # Check if EAR is below the threshold
            if ear < EAR_THRESHOLD:
                counter += 1
                if counter >= CONSECUTIVE_FRAMES:
                    if not alarm_triggered:
                        alarm_triggered = True
                        t = T(target=sound_alarm, args=("alarm.wav",))
                        t.daemon = True
                        t.start()
                cv.putText(frame, "Alert! You're feeling sleepy", (5, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                counter = 0
                alarm_triggered = False

            # Display EAR on the frame
            cv.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame
        face = cascade.detectMultiScale(frame)
        for (x,y,w,h) in face:
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        cv.imshow("Drowsiness Detection", frame)
        if cv.waitKey(1) == ord('q'):
            break

    v_c.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    detect_drowsiness()

