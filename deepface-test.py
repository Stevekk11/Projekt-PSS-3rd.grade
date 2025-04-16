import sys
sys.path.append('/home/stevek/bulb-test/bulbtest/lib/python3.11/site-packages')
from picamera2 import Picamera2, Preview
from deepface import DeepFace
import cv2
import numpy as np
import time

# Initialize Picamera2
picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

# Configuration for DeepFace
DB_PATH = "known_faces/"  # Path to folder with known faces
DETECTOR_BACKEND = "retinaface"  # Detection backend
RECOGNITION_MODEL = "ArcFace"  # Recognition model
MIN_CONFIDENCE = 0.4  # Confidence threshold

# Warm-up DeepFace model cache
DeepFace.find(img_path=np.zeros((100, 100, 3), dtype=np.uint8), db_path=DB_PATH, enforce_detection=False)

print("[INFO] Starting live face recognition...")
time.sleep(2)  # Allow camera to warm up

try:
    while True:
        # Capture frame from Picamera2
        frame = picam2.capture_array()

        # Detect faces in the frame using DeepFace
        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )
        except Exception as e:
            print(f"[ERROR] Face detection failed: {str(e)}")
            continue

        # Process detected faces
        for face in faces:
            if not face.get("facial_area"):
                continue

            # Extract facial area coordinates
            x = face["facial_area"]["x"]
            y = face["facial_area"]["y"]
            w = face["facial_area"]["w"]
            h = face["facial_area"]["h"]

            # Draw bounding box around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Perform face recognition
            try:
                cropped_face = frame[y:y+h, x:x+w]
                dfs = DeepFace.find(
                    img_path=cropped_face,
                    db_path=DB_PATH,
                    model_name=RECOGNITION_MODEL,
                    enforce_detection=False,
                    silent=True
                )

                if not dfs[0].empty and dfs[0].iloc[0]["distance"] < MIN_CONFIDENCE:
                    identity = dfs[0].iloc[0]["identity"].split("/")[-2]
                    cv2.putText(frame, identity, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            except Exception as e:
                print(f"[ERROR] Recognition failed: {str(e)}")

        # Display the frame with bounding boxes and names
        cv2.imshow("Live Face Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("[INFO] Exiting...")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
