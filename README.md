# Face-Recognition
The Face Recognition Application is a Python-based program that utilizes the Face Recognition library to detect and recognize faces in real-time using a webcam or camera feed. This application can compare detected faces with a set of known faces and provide confidence levels for the matches. Below, you'll find information on how to set up and use the application effectively.

# Features
- Real-time face detection and recognition using a webcam or camera feed.
- Comparing detected faces with known faces stored in the "faces" folder.
- Displaying bounding boxes around detected faces and labeling them with confidence levels.
- Different box colors indicating whether the face is known, unknown, or falls within a certain confidence range.

# Setup
Before using the Face Recognition Application, ensure you have the following:
- Python 3.x installed on your system.
- Required Python packages installed. You can install them using the following command:
pip install numpy opencv-python-headless face-recognition

# Usage
Follow these steps to set up and use the Face Recognition Application:

1. Clone the repository
git clone [repository_url]

2. Prepare Known Faces:
Add images of known faces to the "faces" folder. Each image should contain a single face and should be named appropriately to identify the person (e.g., "john.jpg" or "mary.png").

3. Run the Application:
python face_recognition_app.py

# Usage
- The application will open your webcam or camera feed.
- It will automatically detect and recognize faces in real-time.
- Detected faces will be enclosed in bounding boxes, and labels will display confidence levels or "Unknown" if the face is not recognized.
- The box color indicates whether the face is known (green), falls within a certain confidence range (yellow), or is unknown (red).
- Press "Q" to quit the application.

# Dependencies
- Python 3.x
- OpenCV (cv2)
- NumPy
- Face Recognition Library

# Acknowledgments
- This application utilizes the Face Recognition library, which is built on the dlib library.
- The project may require additional setup for GPU support and face recognition model availability.

# Author
Michael Park
