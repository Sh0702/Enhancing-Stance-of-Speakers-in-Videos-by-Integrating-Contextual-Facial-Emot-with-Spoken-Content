import os
import pandas as pd
import matplotlib.pyplot as plt

# Define paths
reports_dir = "/content/drive/MyDrive/sentiment/csv/reports"
platforms = ["instagram", "tiktok", "youtube"]
categories = ["alm", "blm"]

# Initialize storage for video durations
duration_data = []

# Process files
for platform in platforms:
    for category in categories:
        folder_path = os.path.join(reports_dir, platform, category)

        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Ensure 'frame' column exists
                if 'frame' not in df.columns:
                    print(f"Missing 'frame' column in {file}. Skipping.")
                    continue

                # Calculate video duration (assuming 30 FPS)
                video_duration = df['frame'].max() / 30  # Replace 30 with actual FPS if different
                duration_data.append({
                    'platform': platform,
                    'category': category,
                    'file': file,
                    'duration': video_duration
                })

# Convert durations to a DataFrame
duration_df = pd.DataFrame(duration_data)

# Print summary statistics
summary = duration_df.groupby(['platform', 'category'])['duration'].agg(['mean', 'max', 'min'])
print("Summary by Platform and Category:")
print(summary)

# Overall statistics
overall_summary = duration_df['duration'].agg(['mean', 'max', 'min'])
print("\nOverall Summary:")
print(overall_summary)

# Box-and-Whisker Plot
plt.figure(figsize=(10, 6))
duration_df.boxplot(column='duration', by=['platform', 'category'], grid=False)
plt.title("Box-and-Whisker Plot of Video Durations")
plt.suptitle("")
plt.xlabel("Platform and Category")
plt.ylabel("Duration (seconds)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
