import os
import pandas as pd
import ast


def safe_eval(expression):
    """Safely evaluate a string as a dictionary."""
    try:
        result = ast.literal_eval(expression)
        return result if isinstance(result, dict) else {}
    except (ValueError, SyntaxError, TypeError):
        return {}


def process_and_combine(subtitles_path, reports_path, output_path):
    """
    Combine files of the same name from subtitles and reports folders,
    normalize emotion column, and save combined output in lowercase.
    """
    platforms = ["instagram", "tiktok", "youtube"]
    categories = ["alm", "blm"]

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    for platform in platforms:
        for category in categories:
            subtitles_folder = os.path.join(subtitles_path, platform, category)
            reports_folder = os.path.join(reports_path, platform, category)
            output_folder = os.path.join(output_path, platform, category)

            # Create the output subdirectory for platform/category
            os.makedirs(output_folder, exist_ok=True)

            if not os.path.exists(subtitles_folder):
                print(f"Subtitles folder missing for {platform}/{category}. Skipping.")
                continue

            if not os.path.exists(reports_folder):
                print(f"Reports folder missing for {platform}/{category}. Skipping.")
                continue

            for file in os.listdir(subtitles_folder):
                if file.endswith(".xlsx"):
                    subtitles_file = os.path.join(subtitles_folder, file)
                    reports_file = os.path.join(reports_folder, file.replace(".xlsx", ".csv"))

                    if not os.path.exists(reports_file):
                        print(f"Report file missing for {file}. Skipping.")
                        continue

                    try:
                        # Read subtitles and reports files
                        subtitles_df = pd.read_excel(subtitles_file, engine="openpyxl")
                        reports_df = pd.read_csv(reports_file)

                        # Normalize the emotion column (second_category)
                        reports_df['second_category'] = reports_df['second_category'].str.lower()
                        reports_df['second_category'] = reports_df['second_category'].replace('neutral', 'positive')

                        # Extract relevant columns
                        subtitles_df['ground_truth_text'] = subtitles_df['ground_truth'].apply(safe_eval)

                        # Initialize match column
                        subtitles_df['match'] = False

                        # Compare text and emotion for matches
                        for idx, row in subtitles_df.iterrows():
                            frame = row['Frame']
                            ground_truth_text = row['ground_truth_text']

                            # Find the matching row in reports by frame
                            report_row = reports_df[reports_df['frame'] == frame]
                            if report_row.empty:
                                continue

                            emotion = report_row.iloc[0]['second_category']

                            # Check if any word from ground_truth_text matches emotion
                            match = any(
                                word_emotion == emotion for word, word_emotion in ground_truth_text.items()
                            )
                            subtitles_df.at[idx, 'match'] = match

                        # Combine the dataframes
                        combined_df = pd.concat(
                            [
                                subtitles_df[['Frame', 'Subtitle', 'ground_truth_text', 'match']],
                                reports_df[['frame', 'second_category']],
                            ],
                            axis=1,
                        )

                        # Save the combined dataframe
                        output_file = os.path.join(output_folder, file.replace(".xlsx", ".csv"))
                        combined_df.to_csv(output_file, index=False)
                        print(f"Combined and processed file saved: {output_file}")

                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue


# Paths to folders
subtitles_path = "./subtitles"
reports_path = "./reports"
output_path = "./output_combination"

process_and_combine(subtitles_path, reports_path, output_path)