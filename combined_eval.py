import os
import pandas as pd


def calculate_metrics(output_path):
    """
    Calculate accuracy, precision, recall, and F1-score for the files in the output_combination folder.
    """
    platforms = ["instagram", "tiktok", "youtube"]
    categories = ["alm", "blm"]

    # Initialize counters
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for platform in platforms:
        for category in categories:
            platform_folder = os.path.join(output_path, platform, category)

            if not os.path.exists(platform_folder):
                print(f"Folder missing for {platform}/{category}. Skipping.")
                continue

            for file in os.listdir(platform_folder):
                if file.endswith(".csv"):
                    file_path = os.path.join(platform_folder, file)

                    try:
                        # Read the file
                        df = pd.read_csv(file_path)

                        # Iterate through the rows to calculate metrics
                        for _, row in df.iterrows():
                            second_category = row['second_category']
                            match = row['match']
                            ground_truth_text = eval(row['ground_truth_text'])

                            # Convert neutral to positive if necessary
                            if second_category == "neutral":
                                second_category = "positive"

                            if match:
                                # Match exists
                                if second_category == "positive":
                                    true_positive += 1
                                elif second_category == "negative":
                                    true_negative += 1
                            else:
                                # No match
                                if second_category == "positive":
                                    false_positive += 1
                                elif second_category == "negative":
                                    false_negative += 1

                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue

    # Calculate metrics
    total = true_positive + true_negative + false_positive + false_negative
    accuracy = (true_positive + true_negative) / total if total > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print results
    print("Metrics:")
    print(f"True Positive: {true_positive}")
    print(f"True Negative: {true_negative}")
    print(f"False Positive: {false_positive}")
    print(f"False Negative: {false_negative}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")


# Path to the output_combination folder
output_path = "./output_combination"
calculate_metrics(output_path)