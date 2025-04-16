import sys
import os
sys.path.append('/home/stevek/bulb-test/bulbtest/lib/python3.11/site-packages')
import subprocess
from picamera2 import Picamera2
from flask import Flask, Response, render_template, request
from functools import wraps
import threading
from buzzer import activate_siren, deactivate_siren, setup_gpio
import asyncio
import cv2
import requests

# Initialize Haar Cascade
haar_cascade_path = "/usr/lib/python3/dist-packages/data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)
max_val_global = 0

# Initialize the camera
picam2 = Picamera2()
width, height = 1280, 720
picam2.configure(picam2.create_preview_configuration(main={"size": (width, height)}))
picam2.start()
app = Flask(__name__)
frame_counter = 0
detected_faces = []
lock = threading.Lock()
bulb_activated = False


def capture_frame():
    """Captures a frame from the camera and detects faces using Haar cascade."""
    global frame_counter, detected_faces, max_val_global, bulb_activated

    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.flip(frame_bgr, 0)

    if frame_counter % 5 == 0:  # Process every 5th frame for efficiency
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for Haar cascade
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        detected_faces = faces  # Store detected faces

        if len(faces) > 0:  # If faces are detected
            activate_siren()

            if not bulb_activated:
                print("Face detected. Turning on the bulb...")
                # Use Popen to execute the bulb activation command asynchronously
                bulb_activated = True  # Mark the bulb as activated
                subprocess.Popen(["python3", "project/Stevek/BulbControl.py"])
                max_val_global = 1
        else:  # No faces detected
            deactivate_siren()
            
            if bulb_activated:
                bulb_activated = False  # Mark the bulb as deactivated
                max_val_global = 0

    frame_counter += 1
    return frame_bgr



def draw_faces(frame_bgr):
    """Draws rectangles around detected faces on the frame."""
    global detected_faces
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 255, 0), 2)


def generate_frames():
    """Generates frames for the video stream."""
    while True:
        frame_bgr = capture_frame()
        draw_faces(frame_bgr)
        ret, buffer = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Authentication decorator
def check_auth(username, password):
    return username == 'stevek' and password == '9286'


def authenticate():
    return Response(
        'Could not verify your login.', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated


@app.route('/')
@requires_auth
def index():
    return render_template('index.html', max_val=max_val_global)


@app.route('/video_feed')
@requires_auth
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/max_val')
def max_val():
    return {"max_val":max_val_global}

if __name__ == '__main__':
    setup_gpio()  # Call the setup function to initialize GPIO
    try:
        app.run(host='0.0.0.0', port=8085)
    finally:
        GPIO.cleanup()  # Ensure GPIO cleanup on exit
