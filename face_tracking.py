import cv2
from filterpy.kalman import KalmanFilter
import numpy as np
from deepface import DeepFace

class FaceTracking:
    def __init__(self):
        self.kalman_filters = []
        self.tracks = []
        self.face_ids = {}
        self.face_embeddings = {}
        self.next_id = 0
        self.max_frames_no_detection = 2

    @staticmethod
    def initialize_kalman():
        kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.0  # time interval

        kf.F = np.array([[1, 0, 0, 0, dt, 0, 0],
                         [0, 1, 0, 0, 0, dt, 0],
                         [0, 0, 1, 0, 0, 0, dt],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 1]])

        kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]])

        kf.P *= 1000.0
        kf.P[4:, 4:] *= 1000.0

        kf.Q *= 0.1
        kf.R = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

        return kf

    @staticmethod
    def detect_faces(frame):
        faces = DeepFace.extract_faces(frame, detector_backend='retinaface', enforce_detection=False)
        return faces

    @staticmethod
    def draw_bounding_boxes(frame, faces, ids, emotions):
        for (x, y, w, h), face_id, emotion in zip(faces, ids, emotions):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {face_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    @staticmethod
    def euclidean_distance(embedding1, embedding2):
        return np.linalg.norm(np.array(embedding1) - np.array(embedding2))

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    @staticmethod
    def update_embedding(old_embedding, new_embedding, alpha=0.5):
        return alpha * np.array(old_embedding) + (1 - alpha) * np.array(new_embedding)

    @staticmethod
    def read_emotions_from_file(file_path):
        with open(file_path, 'r') as file:
            emotions = [line.strip() for line in file]
        return emotions

    def process_video(self, video_path, output_path, emotions_file_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        emotions_list = self.read_emotions_from_file(emotions_file_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            detected_faces = self.detect_faces(frame)

            if frame_count <= len(emotions_list):
                emotion = emotions_list[frame_count - 1]
            else:
                emotion = 'unknown'

            if detected_faces:
                new_kalman_filters = []
                new_tracks = []
                new_ids = []

                for face in detected_faces:
                    face_area = face['facial_area']
                    x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
                    face_crop = face['face']
                    embedding = DeepFace.represent(face_crop, model_name='Facenet', align=True, enforce_detection=False)[0]["embedding"]

                    new_face = True
                    for kf, track in zip(self.kalman_filters, self.tracks):
                        track_x, track_y, track_w, track_h, _, _, _ = kf.x.flatten()
                        if np.linalg.norm([x - track_x, y - track_y]) < max(w, h):
                            kf.update([x, y, w, h])
                            track['hits'] += 1
                            track['no_detection'] = 0

                            if track['id'] in self.face_ids:
                                if DeepFace.verify(face_crop, self.face_ids[track['id']], model_name='Facenet', enforce_detection=False)["verified"]:
                                    self.face_ids[track['id']] = face_crop
                                else:
                                    self.next_id += 1
                                    self.face_ids[self.next_id] = face_crop
                                    track['id'] = self.next_id

                            new_ids.append(track['id'])
                            new_face = False
                            break

                    if new_face:
                        assigned_id = None
                        min_dist = float("inf")

                        for face_id, saved_embedding in self.face_embeddings.items():
                            dist = self.cosine_similarity(embedding, saved_embedding)
                            if dist > min_dist and dist >= 0.4:
                                min_dist = dist
                                assigned_id = face_id

                        if assigned_id is not None:
                            self.face_ids[assigned_id] = face_crop
                            self.face_embeddings[assigned_id] = self.update_embedding(self.face_embeddings[assigned_id], embedding)
                            new_ids.append(assigned_id)
                        else:
                            assigned_id = self.next_id
                            self.face_ids[self.next_id] = face_crop
                            self.face_embeddings[self.next_id] = embedding
                            new_ids.append(self.next_id)
                            self.next_id += 1

                        kf = self.initialize_kalman()
                        kf.x = np.array([x, y, w, h, 0, 0, 0]).reshape(-1, 1)
                        new_kalman_filters.append(kf)
                        new_tracks.append({'hits': 1, 'no_detection': 0, 'id': assigned_id})

                self.kalman_filters = self.kalman_filters + new_kalman_filters
                self.tracks = self.tracks + new_tracks

            for kf, track in zip(self.kalman_filters, self.tracks):
                kf.predict()
                track['no_detection'] += 1

            self.kalman_filters = [kf for kf, track in zip(self.kalman_filters, self.tracks) if track['no_detection'] < self.max_frames_no_detection]
            self.tracks = [track for track in self.tracks if track['no_detection'] < self.max_frames_no_detection]

            predicted_faces = []
            ids = []
            for kf, track in zip(self.kalman_filters, self.tracks):
                x, y, w, h, _, _, _ = kf.x.flatten()
                predicted_faces.append((int(x), int(y), int(w), int(h)))
                ids.append(track['id'])

            self.draw_bounding_boxes(frame, predicted_faces, ids, [emotion] * len(predicted_faces))

            print(f"Number of faces detected: {len(predicted_faces)}")

            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        for face_id, face_img in self.face_ids.items():
            img_path = f'face_id_{face_id}.jpg'
            cv2.imwrite(img_path, face_img)
            print(f"Saved face ID {face_id} to {img_path}")
