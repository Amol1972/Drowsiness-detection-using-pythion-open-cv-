This project detects driver drowsiness using computer vision techniques. It leverages eye aspect ratio (EAR) to determine if a user is sleepy and triggers an alarm when drowsiness is detected.

🧠 How It Works
The system uses dlib’s facial landmark detector to locate the eyes.

It calculates the Eye Aspect Ratio (EAR) to monitor eye closure.

If the eyes remain closed for a defined number of frames, the system plays an alert sound.

📦 Dependencies
Ensure the following Python packages are installed:

bash
Copy
Edit
pip install opencv-python dlib playsound face_recognition scipy numpy
Also, make sure to download:

shape_predictor_68_face_landmarks.dat: Pre-trained facial landmark model from dlib.

alarm.wav: An alarm sound file to alert the user.

haarcascade_frontalface_default.xml: Haar Cascade for face detection.

🔧 Files Included
main.py: Main script to run the detection.

alarm.wav: Alarm audio file (You can use any .wav sound).

shape_predictor_68_face_landmarks.dat: Dlib's face landmark detector.

haarcascade_frontalface_default.xml: OpenCV Haar cascade face detector.

▶️ How to Run
bash
Copy
Edit
python advance_driving.py
The system starts your webcam and displays a live feed.

If you close your eyes for 20 consecutive frames, an alarm will play.

Press q to quit the application.

🛠️ Parameters
EAR_THRESHOLD: Default is 0.25. Lower it if the alarm is too sensitive.

CONSECUTIVE_FRAMES: Default is 20. Increase to allow longer eye closures.

📸 Example Output
EAR value is displayed on screen.

A red rectangle highlights the detected face.

Warning text appears if drowsiness is detected.

🧠 Applications
Driver fatigue monitoring

Workplace attention tracking

Sleep research

