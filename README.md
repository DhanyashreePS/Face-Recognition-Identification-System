# Face Recognition Identification System

A real-time face recognition web application that detects faces through a webcam, generates face embeddings using FaceNet, compares them with enrolled faces using cosine similarity, and rejects unknown faces using a similarity threshold.

## Features

- Real-time webcam face detection
- Face detection using **MTCNN**
- Face embeddings using **FaceNet**
- Cosine similarity-based face matching
- Unknown-face rejection using a configurable threshold
- Web-based interface using **Flask**
- Simple enrollment of new people through the webcam
- Visual bounding box around detected faces
- Displays identity and similarity score

## System Flow

```text
Webcam
   ↓
Face Detection
   ↓
MTCNN
   ↓
Face Crop
   ↓
FaceNet Embedding
   ↓
Cosine Similarity
   ↓
Compare with Enrolled Faces
   ↓
Similarity ≥ 0.45 ?
   ↓
Known Person / Unknown
```

## Technologies Used

- Python
- Flask
- OpenCV
- MTCNN
- FaceNet
- NumPy
- Scikit-learn
- HTML / CSS / JavaScript

## How It Works

### 1. Face Detection

The webcam continuously captures frames. MTCNN detects faces in each frame and creates a bounding box around every detected face.

### 2. Face Embedding

The detected face is resized and passed to FaceNet. FaceNet converts the face into a numerical embedding that represents the facial features.

### 3. Face Matching

The new embedding is compared with the embeddings of enrolled people using **cosine similarity**.

### 4. Unknown Face Rejection

The system uses a similarity threshold of **0.45**.

```text
Similarity ≥ 0.45  →  Recognized person
Similarity < 0.45  →  Unknown
```

For example:

```text
Dhanya (0.94)
```

means the best matching enrolled identity is Dhanya with a similarity score of 0.94.

```text
Unknown (0.24)
```

means the best similarity score is below the threshold, so the person is rejected as unknown.

## Enrollment

To enroll a person:

1. Enter the person's name.
2. Start the enrollment process.
3. The webcam captures the person's face.
4. MTCNN detects the face.
5. FaceNet generates the face embedding.
6. The embedding is stored for future identification.

The enrolled embeddings are stored locally in:

```text
web_embeddings.pkl
```

This file is generated automatically and is not included in the GitHub repository.

## Screenshots

### 1. Initiation

The main interface provides the live webcam feed and enrollment options.

<img width="929" height="780" alt="Screenshot 2026-09-21 150305" src="https://github.com/user-attachments/assets/ad4cdd39-65ec-487e-89e5-487177c810e1" />


### 2. Person Enrollment

A person can be enrolled by entering their name and capturing their face through the webcam.

<img width="890" height="813" alt="Screenshot 2026-09-21 150322" src="https://github.com/user-attachments/assets/3cf2204c-304e-4007-ab05-4c12e36c2726" />


### 3. Known Person Recognition

After enrollment, the system recognizes the person and displays the identity with the similarity score.

<img width="958" height="788" alt="Screenshot 2026-09-21 150408" src="https://github.com/user-attachments/assets/2b9e9446-0805-4263-8603-d7a0fca3c49c" />


### 4. Unknown Person Rejection

When a person who is not enrolled is detected, the system displays **Unknown** because the similarity score is below the threshold.

<img width="854" height="807" alt="Screenshot 2026-09-21 150511" src="https://github.com/user-attachments/assets/4174c6cf-1d58-43f2-9720-6e26a692cc94" />


<img width="876" height="777" alt="Screenshot 2026-09-21 150436" src="https://github.com/user-attachments/assets/e6535f9c-1720-44b3-8bb4-dca1b05e7409" />



## Project Structure

```text
Face-Recognition-Identification-System/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── templates/
    └── index.html
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DhanyashreePS/Face-Recognition-Identification-System.git
cd Face-Recognition-Identification-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

Allow webcam access when requested.

## Testing

The system can be tested using:

### Known Person

1. Enroll a person.
2. Stand in front of the webcam again.
3. The system should display the enrolled person's name and similarity score.

### Unknown Person

1. Do not enroll the second person.
2. Let the second person stand in front of the webcam.
3. The system should display **Unknown** when the similarity score is below the threshold.

### Different Conditions

The system can also be tested under:

- Different lighting conditions
- Different face positions
- Slight changes in facial appearance
- Multiple faces in the frame
- Blurry or partially visible faces

## Failure Cases

The recognition result can be affected by:

- Poor lighting
- Blurry images
- Partial face visibility
- Extreme face angles
- Very small faces
- Multiple faces in the frame
- Significant changes in appearance

## Possible Improvements

- Use multiple images during enrollment
- Add face quality checking
- Improve handling of different lighting conditions
- Add liveness detection
- Add face tracking for smoother recognition
- Calibrate the similarity threshold using a larger test dataset
- Add formal evaluation metrics such as precision, recall, F1-score, FAR and FRR

## Conclusion

This project demonstrates a complete real-time face recognition pipeline:

**Face Detection → Face Embedding → Similarity Matching → Unknown Rejection**

The system provides a simple web interface for enrollment and real-time identification using a webcam.
