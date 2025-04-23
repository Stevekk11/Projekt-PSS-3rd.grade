import sys
import os
sys.path.append('/home/stevek/bulb-test/bulbtest/lib/python3.11/site-packages')
import cv2
import numpy as np
from picamera2 import Picamera2
from flask import Flask, Response, render_template, request
from functools import wraps
import threading
from buzzer import activate_siren, deactivate_siren, setup_gpio
import subprocess

# --- CONFIGURATION ---
KNOWN_FACES_DIR = "known_faces/Stevek"
HAAR_CASCADE_PATH = "/usr/lib/python3/dist-packages/data/haarcascade_frontalface_default.xml"
RECOGNITION_THRESHOLD = 120
max_val_global = 0

# --- LOAD FACE DETECTOR ---
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# --- LOAD KNOWN FACES AND TRAIN RECOGNIZER ---
def load_known_faces(known_faces_dir):
    faces = []
    labels = []
    label_map = {}
    label_id = 0
    
    if not os.path.exists(known_faces_dir):
        print(f"Warning: Directory {known_faces_dir} does not exist!")
        return faces, labels, label_map
        
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            label = filename.split('_')[0]
            if label not in label_map:
                label_map[label] = label_id
                label_id += 1
            
            img_path = os.path.join(known_faces_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            # Detect face in the training image
            faces_in_img = face_cascade.detectMultiScale(img, 1.1, 4, minSize=(25, 25))
            if len(faces_in_img) != 1:
                print(f"Warning: Expected 1 face in {filename}, found {len(faces_in_img)}. Skipping.")
                continue
                
            # Extract and normalize the face
            x, y, w, h = faces_in_img[0]
            face_img = img[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))  # Consistent size
            face_img = cv2.equalizeHist(face_img)  # Normalize lighting
            
            faces.append(face_img)
            labels.append(label_map[label])
            print(f"Added {filename} as {label} (ID: {label_map[label]})")
            
    return faces, labels, {v: k for k, v in label_map.items()}

print("Loading known faces...")
faces, labels, id_label_map = load_known_faces(KNOWN_FACES_DIR)

# Initialize and train the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
if len(faces) > 0 and len(labels) > 0:
    recognizer.train(faces, np.array(labels))
    print(f"Trained recognizer with {len(faces)} faces from {len(id_label_map)} people")
else:
    print("ERROR: No valid training faces found! Recognition will not work.")

# --- CAMERA & FLASK SETUP ---
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
    global frame_counter, detected_faces, max_val_global, bulb_activated

    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.flip(frame_bgr, 0)
    recognized_faces = []

    if frame_counter % 5 == 0:  # Process every 5th frame
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Clear previous detections
        detected_faces = []
        
        for (x, y, w, h) in faces_rects:
            face_roi = gray[y:y+h, x:x+w]
            
            # Skip if face is too small
            if w < 30 or h < 30:
                continue
                
            # Preprocess face same way as training data
            face_roi = cv2.resize(face_roi, (100, 100))
            face_roi = cv2.equalizeHist(face_roi)
            
            # Only try recognition if we have trained faces
            if len(faces) > 0:
                label_id, confidence = recognizer.predict(face_roi)
                label = id_label_map.get(label_id, "Unknown")
                
                # For LBPH: Lower confidence = better match
                is_recognized = confidence > RECOGNITION_THRESHOLD
                
                if is_recognized:
                    recognized_faces.append((label, confidence))
                    print(f"Recognized {label} with confidence {confidence:.2f}")
                else:
                    label = "Unknown"
                    print(f"Unknown face detected (confidence {confidence:.2f})")
            else:
                label = "No training data"
                confidence = 999
                
            detected_faces.append((x, y, w, h, label, confidence))
        
        # Activate based on recognition, not just detection
        if recognized_faces:
            # Get the best match if multiple faces recognized
            best_match = min(recognized_faces, key=lambda x: x[1])
            max_val_global = 1  # Recognized face
            
            activate_siren()
            if not bulb_activated:
                print(f"Recognized {best_match[0]}. Turning on bulb...")
                bulb_activated = True
                subprocess.Popen(["python3", "project/Stevek/BulbControl.py"])
        else:
            max_val_global = 0  # No recognized face
            deactivate_siren()
            if bulb_activated:
                bulb_activated = False

    frame_counter += 1
    return frame_bgr

def draw_faces(frame_bgr):
    for (x, y, w, h, label, confidence) in detected_faces:
        # Green for recognized, red for unknown
        is_recognized = label != "Unknown" and confidence > RECOGNITION_THRESHOLD
        color = (0, 255, 0) if is_recognized else (0, 0, 255)
        
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        
        # Add label and confidence
        text = f"{label}: {confidence:.1f}" if label != "Unknown" else "Unknown"
        cv2.putText(frame_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def generate_frames():
    while True:
        frame_bgr = capture_frame()
        draw_faces(frame_bgr)
        ret, buffer = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Rest of the Flask app code remains the same
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
    return {"max_val": max_val_global}

if __name__ == '__main__':
    setup_gpio()
    try:
        app.run(host='0.0.0.0', port=8086)
    finally:
        import RPi.GPIO as GPIO
        GPIO.cleanup()


