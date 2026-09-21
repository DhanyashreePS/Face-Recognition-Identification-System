
from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import pickle
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -----------------------------
# Configuration
# -----------------------------

DATABASE_FILE = "web_embeddings.pkl"
THRESHOLD = 0.45

# -----------------------------
# Load models ONCE
# -----------------------------

print("Loading face detector...")
detector = MTCNN()

print("Loading FaceNet...")
embedder = FaceNet()

print("Models loaded successfully.")

# -----------------------------
# Load saved embeddings
# -----------------------------

if os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, "rb") as f:
        database = pickle.load(f)
else:
    database = {}

print("Enrolled people:", list(database.keys()))

# -----------------------------
# Webcam
# -----------------------------

camera = None


def get_embedding(face_image):
    """Generate FaceNet embedding."""

    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    face_image = cv2.resize(face_image, (160, 160))

    embedding = embedder.embeddings([face_image])[0]

    return embedding


def detect_faces(frame):
    """Detect faces using MTCNN."""

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections = detector.detect_faces(rgb)

    return detections


def recognize_face(face_image):
    """Compare face with enrolled embeddings."""

    if not database:
        return "No enrolled person", 0.0

    embedding = get_embedding(face_image)

    best_name = "Unknown"
    best_score = 0.0

    for name, saved_embedding in database.items():

        score = cosine_similarity(
            [embedding],
            [saved_embedding]
        )[0][0]

        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= THRESHOLD:
        return best_name, best_score

    return "Unknown", best_score


# -----------------------------
# Generate webcam frames
# -----------------------------

def generate_frames():

    global camera

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam")
        return

    while True:

        success, frame = camera.read()

        if not success:
            break

        detections = detect_faces(frame)

        for detection in detections:

            x, y, w, h = detection["box"]

            # Prevent negative coordinates
            x = max(0, x)
            y = max(0, y)

            face = frame[
                y:y + h,
                x:x + w
            ]

            if face.size == 0:
                continue

            name, score = recognize_face(face)

            # Bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Label
            label = f"{name} ({score:.2f})"

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Convert frame to JPEG
        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

    if camera:
        camera.release()
        camera = None


# -----------------------------
# Home page
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Webcam stream
# -----------------------------

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# -----------------------------
# Enroll person
# -----------------------------

@app.route("/enroll", methods=["POST"])
def enroll():

    global database

    data = request.get_json()

    if not data or "name" not in data:
        return jsonify({
            "success": False,
            "message": "Name is required"
        })

    name = data["name"].strip()

    if not name:
        return jsonify({
            "success": False,
            "message": "Please enter a name"
        })

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return jsonify({
            "success": False,
            "message": "Could not access webcam"
        })

    print(f"Capturing face for {name}...")

    embedding = None

    # Capture several frames
    for _ in range(30):

        success, frame = cap.read()

        if not success:
            continue

        detections = detect_faces(frame)

        # Only enroll when exactly one face is visible
        if len(detections) == 1:

            x, y, w, h = detections[0]["box"]

            x = max(0, x)
            y = max(0, y)

            face = frame[
                y:y + h,
                x:x + w
            ]

            if face.size != 0:

                embedding = get_embedding(face)

                break

    cap.release()

    if embedding is None:

        return jsonify({
            "success": False,
            "message": "No clear single face detected. Please try again."
        })

    # Save embedding
    database[name] = embedding

    with open(DATABASE_FILE, "wb") as f:
        pickle.dump(database, f)

    print(f"Successfully enrolled: {name}")

    return jsonify({
        "success": True,
        "message": f"{name} enrolled successfully"
    })


# -----------------------------
# Run application
# -----------------------------

if __name__ == "__main__":

    print("\n===================================")
    print("Face Recognition Web Application")
    print("===================================")
    print("Open: http://127.0.0.1:5000")
    print("===================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
