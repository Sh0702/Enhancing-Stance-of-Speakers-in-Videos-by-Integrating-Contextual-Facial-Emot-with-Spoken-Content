import cv2
from filterpy.kalman import KalmanFilter
import numpy as np
from deepface import DeepFace

# Function to initialize a Kalman filter for tracking
def initialize_kalman():
    kf = KalmanFilter(dim_x=7, dim_z=4)
    dt = 1.0  # time interval

    # State transition matrix
    kf.F = np.array([[1, 0, 0, 0, dt, 0, 0],
                     [0, 1, 0, 0, 0, dt, 0],
                     [0, 0, 1, 0, 0, 0, dt],
                     [0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 1]])

    # Measurement function
    kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0]])

    # Covariance matrix
    kf.P *= 1000.0
    kf.P[4:, 4:] *= 1000.0

    # Process noise
    kf.Q *= 0.1

    # Measurement noise
    kf.R = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    return kf

# Function to detect faces in a frame using DeepFace with RetinaFace as the backend
def detect_faces(frame):
    faces = DeepFace.extract_faces(frame, detector_backend='retinaface', enforce_detection=False)
    return faces

def draw_bounding_boxes(frame, faces, ids, emotions):
    for (x, y, w, h), face_id, emotion in zip(faces, ids, emotions):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {face_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

# Function to calculate the Euclidean distance between two embeddings
def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(np.array(embedding1) - np.array(embedding2))

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Function to update the embedding by weighted average
def update_embedding(old_embedding, new_embedding, alpha=0.5):
    return alpha * np.array(old_embedding) + (1 - alpha) * np.array(new_embedding)

# Function to read emotions from a text file
def read_emotions_from_file(file_path):
    with open(file_path, 'r') as file:
        emotions = [line.strip() for line in file]
    return emotions

# Function to process video frames for face detection, tracking, and recognition
def process_video(video_path, output_path, emotions_file_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read emotions from the text file
    emotions_list = read_emotions_from_file(emotions_file_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    kalman_filters = []
    tracks = []
    face_ids = {}
    face_embeddings = {}
    next_id = 0

    frame_count = 0
    max_frames_no_detection = 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detected_faces = detect_faces(frame)

        # Ensure we have a corresponding emotion for the current frame
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
                face_crop = face['face']  # Cropped face image
                embedding = DeepFace.represent(face_crop, model_name='Facenet', align=True, enforce_detection=False)[0]["embedding"]

                new_face = True
                for kf, track in zip(kalman_filters, tracks):
                    track_x, track_y, track_w, track_h, _, _, _ = kf.x.flatten()
                    if np.linalg.norm([x - track_x, y - track_y]) < max(w, h):
                        kf.update([x, y, w, h])
                        track['hits'] += 1
                        track['no_detection'] = 0

                        # Update face ID
                        if track['id'] in face_ids:
                            if DeepFace.verify(face_crop, face_ids[track['id']], model_name='Facenet', enforce_detection=False)["verified"]:
                                face_ids[track['id']] = face_crop
                            else:
                                next_id += 1
                                face_ids[next_id] = face_crop
                                track['id'] = next_id

                        new_ids.append(track['id'])
                        new_face = False
                        break

                if new_face:
                    # Check if the new face matches any previous face embeddings
                    assigned_id = None
                    min_dist = float("inf")

                    for face_id, saved_embedding in face_embeddings.items():
                        dist = cosine_similarity(embedding, saved_embedding)
                        if dist > min_dist and dist >= 0.4:  # Set a suitable threshold for matching
                            min_dist = dist
                            assigned_id = face_id

                    if assigned_id is not None:
                        # Update the existing face with the new embedding
                        face_ids[assigned_id] = face_crop
                        face_embeddings[assigned_id] = update_embedding(face_embeddings[assigned_id], embedding)
                        new_ids.append(assigned_id)
                    else:
                        # Assign a new ID if no match is found
                        assigned_id = next_id
                        face_ids[next_id] = face_crop
                        face_embeddings[next_id] = embedding
                        new_ids.append(next_id)
                        next_id += 1

                    kf = initialize_kalman()
                    kf.x = np.array([x, y, w, h, 0, 0, 0]).reshape(-1, 1)
                    new_kalman_filters.append(kf)
                    new_tracks.append({'hits': 1, 'no_detection': 0, 'id': assigned_id})

            kalman_filters = kalman_filters + new_kalman_filters
            tracks = tracks + new_tracks

        # Predict new states for all Kalman filters
        for kf, track in zip(kalman_filters, tracks):
            kf.predict()
            track['no_detection'] += 1

        # Remove tracks that have not been detected for too long
        kalman_filters = [kf for kf, track in zip(kalman_filters, tracks) if track['no_detection'] < max_frames_no_detection]
        tracks = [track for track in tracks if track['no_detection'] < max_frames_no_detection]

        # Get the current states and corresponding IDs
        predicted_faces = []
        ids = []
        for kf, track in zip(kalman_filters, tracks):
            x, y, w, h, _, _, _ = kf.x.flatten()
            predicted_faces.append((int(x), int(y), int(w), int(h)))
            ids.append(track['id'])

        # Draw bounding boxes and emotions in the current frame
        draw_bounding_boxes(frame, predicted_faces, ids, [emotion] * len(predicted_faces))

        # Print the number of faces detected in the current frame
        print(f"Number of faces detected: {len(predicted_faces)}")

        # Write the frame into the file
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save the unique IDs and corresponding face images
    for face_id, face_img in face_ids.items():
        img_path = f'face_id_{face_id}.jpg'
        cv2.imwrite(img_path, face_img)
        print(f"Saved face ID {face_id} to {img_path}")

# Path to your video file
video_path = '/content/angelina2.mp4'

# Path to the output video file
output_path = '/content/output.mp4'

# Path to the emotions file
emotions_file_path = '/content/frame_emotions.txt'

# Process the video
process_video(video_path, output_path, emotions_file_path)
