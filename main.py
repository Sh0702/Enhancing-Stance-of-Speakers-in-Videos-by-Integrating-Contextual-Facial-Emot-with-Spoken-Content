from face_tracking import FaceTracking

# Path to your video file
video_path = '/path/to/video'

# Path to the output video file
output_path = '/path/to/output'

# Path to the emotions file
emotions_file_path = '/path/to/csv_file'

# Create an instance of FaceTrackingFactory
face_tracker = FaceTrackingFactory()

# Process the video
face_tracker.process_video(video_path, output_path, emotions_file_path)
