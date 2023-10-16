import sys
import os
import cv2
import numpy as np
import math
import face_recognition


def compute_face_confidence(distance, threshold=0.6):
    range_value = (1.0 - threshold)
    linear_val = (1.0 - distance) / (range_value * 2.0)

    if distance > threshold:
        return linear_val * 100
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return round(value, 2)


class FaceIdentifier:
    detected_face_locations = []
    detected_face_encodings = []
    detected_face_labels = []
    stored_face_encodings = []
    stored_face_labels = []
    process_toggle = True

    def __init__(self):
        self.encode_known_faces()

    def encode_known_faces(self):
        for image_file in os.listdir('faces'):
            image = face_recognition.load_image_file(f'faces/{image_file}')
            encoding = face_recognition.face_encodings(image)
            if encoding:
                self.stored_face_encodings.append(encoding[0])
                self.stored_face_labels.append(os.path.splitext(image_file)[0])  # Use filename without extension

        print(self.stored_face_labels)

    def start_recognition(self):
        video_stream = cv2.VideoCapture(0)

        if not video_stream.isOpened():
            sys.exit("Video source not found...")

        while True:
            success, frame = video_stream.read()

            if self.process_toggle:
                downscaled_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_frame = downscaled_frame[:, :, ::-1]
                self.detected_face_locations = face_recognition.face_locations(rgb_frame)
                self.detected_face_encodings = face_recognition.face_encodings(rgb_frame, self.detected_face_locations)

                self.detected_face_labels = []
                for encoding in self.detected_face_encodings:
                    matches = face_recognition.compare_faces(self.stored_face_encodings, encoding)
                    label = 'Unknown'

                    face_distances = face_recognition.face_distance(self.stored_face_encodings, encoding)
                    closest_match_idx = np.argmin(face_distances)

                    if matches[closest_match_idx]:
                        confidence = compute_face_confidence(face_distances[closest_match_idx])
                        label = f"{self.stored_face_labels[closest_match_idx]} ({confidence}%)"

                    self.detected_face_labels.append(label)

            self.process_toggle = not self.process_toggle

            # Annotate frame
            for (top, right, bottom, left), label in zip(self.detected_face_locations, self.detected_face_labels):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                box_color = self._choose_box_color(label)
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, -1)
                cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            key = cv2.waitKey(1)
            if key in [ord('q'), ord('Q')]:
                break

        video_stream.release()
        cv2.destroyAllWindows()

    @staticmethod
    def _choose_box_color(label):
        if "Unknown" in label:
            return (0, 0, 255)
        confidence_val = float(label.split('(')[-1].replace('%)', '').strip())
        if confidence_val > 90:
            return (0, 255, 0)
        elif 50 <= confidence_val <= 90:
            return (0, 255, 255)
        else:
            return (0, 0, 255)


if __name__ == '__main__':
    face_id = FaceIdentifier()
    face_id.start_recognition()



