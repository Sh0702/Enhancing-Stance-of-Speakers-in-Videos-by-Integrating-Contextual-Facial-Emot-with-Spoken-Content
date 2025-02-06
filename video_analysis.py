import os
import pandas as pd
import matplotlib.pyplot as plt

# Define paths
reports_dir = "./reports"
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

# Summary Statistics
summary_by_category = duration_df.groupby(['platform', 'category'])['duration'].agg(['mean', 'max', 'min'])
summary_by_platform = duration_df.groupby(['platform'])['duration'].agg(['mean', 'max', 'min'])
overall_summary = duration_df['duration'].agg(['mean', 'max', 'min'])

# Print summaries
print("Summary by Platform and Category:")
print(summary_by_category)
print("\nSummary by Platform:")
print(summary_by_platform)
print("\nOverall Summary:")
print(overall_summary)

# Box-and-Whisker Plot: Category and Platform
plt.figure(figsize=(10, 6))
duration_df.boxplot(column='duration', by=['platform', 'category'], grid=False)
plt.title("Box-and-Whisker Plot of Video Durations (By Category and Platform)")
plt.suptitle("")
plt.xlabel("Platform and Category")
plt.ylabel("Duration (seconds)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Box-and-Whisker Plot: Platform
plt.figure(figsize=(8, 6))
duration_df.boxplot(column='duration', by=['platform'], grid=False)
plt.title("Box-and-Whisker Plot of Video Durations (By Platform)")
plt.suptitle("")
plt.xlabel("Platform")
plt.ylabel("Duration (seconds)")
plt.tight_layout()
plt.show()

# Box-and-Whisker Plot: Overall Dataset
plt.figure(figsize=(6, 6))
duration_df.boxplot(column='duration', grid=False)
plt.title("Box-and-Whisker Plot of Video Durations (Overall Dataset)")
plt.ylabel("Duration (seconds)")
plt.tight_layout()
plt.show()