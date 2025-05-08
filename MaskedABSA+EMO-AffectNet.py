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

def match_emotion_text(row, reports_df):
    """
    Given a subtitle row and the corresponding reports dataframe,
    check if the EMO-AffectNet emotion aligns with MaskedABSA prediction.
    """
    frame = row['Frame']
    ground_truth_text = row['ground_truth_text']

    report_row = reports_df[reports_df['frame'] == frame]
    if report_row.empty:
        return False

    emotion = report_row.iloc[0]['second_category'].lower()
    emotion = 'positive' if emotion == 'neutral' else emotion  # Normalize

    return any(word_emotion == emotion for word, word_emotion in ground_truth_text.items())

def process_and_combine(subtitles_path, reports_path, output_path):
    """
    For each platform/category/video file:
    - Load subtitle (MaskedABSA) and emotion report (EMO-AffectNet) files
    - Compare if the ABSA text sentiment aligns with the facial emotion
    - Save combined results to output path
    """
    platforms = ["instagram", "tiktok", "youtube"]
    categories = ["alm", "blm"]

    os.makedirs(output_path, exist_ok=True)

    for platform in platforms:
        for category in categories:
            sub_path = os.path.join(subtitles_path, platform, category)
            rep_path = os.path.join(reports_path, platform, category)
            out_path = os.path.join(output_path, platform, category)
            os.makedirs(out_path, exist_ok=True)

            if not os.path.exists(sub_path) or not os.path.exists(rep_path):
                print(f"Missing folder(s) for {platform}/{category}. Skipping.")
                continue

            for file in os.listdir(sub_path):
                if file.endswith(".xlsx"):
                    subtitle_file = os.path.join(sub_path, file)
                    report_file = os.path.join(rep_path, file.replace(".xlsx", ".csv"))

                    if not os.path.exists(report_file):
                        print(f"No matching report for {file}. Skipping.")
                        continue

                    try:
                        subtitles_df = pd.read_excel(subtitle_file, engine="openpyxl")
                        reports_df = pd.read_csv(report_file)

                        # Normalize and evaluate emotion column
                        reports_df['second_category'] = reports_df['second_category'].str.lower().replace('neutral', 'positive')
                        subtitles_df['ground_truth_text'] = subtitles_df['ground_truth'].apply(safe_eval)

                        # Compute match between facial emotion and ABSA sentiment
                        subtitles_df['match'] = subtitles_df.apply(lambda row: match_emotion_text(row, reports_df), axis=1)

                        # Combine subset for analysis
                        combined_df = pd.concat(
                            [
                                subtitles_df[['Frame', 'Subtitle', 'ground_truth_text', 'match']],
                                reports_df[['frame', 'second_category']],
                            ],
                            axis=1,
                        )

                        # Save result
                        output_file = os.path.join(out_path, file.replace(".xlsx", ".csv"))
                        combined_df.to_csv(output_file, index=False)
                        print(f"Saved combined result: {output_file}")

                    except Exception as e:
                        print(f"Error with {file}: {e}")
                        continue

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine MaskedABSA + EMO-AffectNet results and evaluate alignment.")
    parser.add_argument("--subtitles_path", type=str, required=True, help="Path to folder containing ABSA outputs (.xlsx)")
    parser.add_argument("--reports_path", type=str, required=True, help="Path to folder containing EMO-AffectNet outputs (.csv)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save combined output CSVs")

    args = parser.parse_args()
    process_and_combine(args.subtitles_path, args.reports_path, args.output_path)
