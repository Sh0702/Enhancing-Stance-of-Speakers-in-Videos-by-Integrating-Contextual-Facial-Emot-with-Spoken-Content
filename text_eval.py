import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_metrics(ground_truth_dict, prediction_dict):
    """
    Compare ground_truth and masked_absa_prediction word by word and calculate metrics.
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total_words = 0

    for word, emotion in ground_truth_dict.items():
        total_words += 1
        pred_emotion = prediction_dict.get(word, None)
        if pred_emotion == emotion:
            true_positive += 1
        elif pred_emotion is not None:
            false_positive += 1
        else:
            false_negative += 1

    accuracy = true_positive / total_words if total_words > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1

def process_files_in_folder(folder_path):
    """
    Process all Excel files in a single folder path and calculate average metrics.
    """
    folder_metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}

    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file)

            # Read the Excel file
            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

            # Ensure required columns exist
            required_columns = ['Subtitle', 'ground_truth', 'masked_absa_prediction']
            if not all(column in df.columns for column in required_columns):
                print(f"Skipping {file} due to missing required columns.")
                continue

            # Drop duplicates and keep the first frame for each subtitle
            df.drop_duplicates(subset=['Subtitle'], keep='first', inplace=True)

            # Process each unique subtitle
            for _, row in df.iterrows():
                try:
                    # Safely evaluate the ground_truth and masked_absa_prediction strings to dictionaries
                    ground_truth = eval(row['ground_truth']) if isinstance(row['ground_truth'], str) else {}
                    masked_prediction = eval(row['masked_absa_prediction']) if isinstance(row['masked_absa_prediction'], str) else {}

                    # Calculate metrics
                    accuracy, precision, recall, f1 = calculate_metrics(ground_truth, masked_prediction)

                    # Store metrics for this subtitle
                    folder_metrics["accuracy"].append(accuracy)
                    folder_metrics["precision"].append(precision)
                    folder_metrics["recall"].append(recall)
                    folder_metrics["f1_score"].append(f1)
                except Exception as e:
                    print(f"Error processing row in {file}: {e}")
                    continue

    # Calculate average metrics for the folder
    avg_accuracy = sum(folder_metrics["accuracy"]) / len(folder_metrics["accuracy"]) if folder_metrics["accuracy"] else 0
    avg_precision = sum(folder_metrics["precision"]) / len(folder_metrics["precision"]) if folder_metrics["precision"] else 0
    avg_recall = sum(folder_metrics["recall"]) / len(folder_metrics["recall"]) if folder_metrics["recall"] else 0
    avg_f1 = sum(folder_metrics["f1_score"]) / len(folder_metrics["f1_score"]) if folder_metrics["f1_score"] else 0

    return avg_accuracy, avg_precision, avg_recall, avg_f1

def process_all_folders(base_folder_path):
    """
    Process all subfolders in the given base folder path and calculate overall metrics.
    """
    overall_metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}

    platforms = ["instagram", "tiktok", "youtube"]
    categories = ["alm", "blm"]

    for platform in platforms:
        for category in categories:
            folder_path = os.path.join(base_folder_path, platform, category)
            if os.path.exists(folder_path):
                print(f"Processing folder: {folder_path}")
                accuracy, precision, recall, f1 = process_files_in_folder(folder_path)
                print(f"Metrics for {platform}/{category}:")
                print(f"  Accuracy: {accuracy:.2f}")
                print(f"  Precision: {precision:.2f}")
                print(f"  Recall: {recall:.2f}")
                print(f"  F1-Score: {f1:.2f}\n")

                # Append folder metrics to overall metrics
                overall_metrics["accuracy"].append(accuracy)
                overall_metrics["precision"].append(precision)
                overall_metrics["recall"].append(recall)
                overall_metrics["f1_score"].append(f1)
            else:
                print(f"Folder {folder_path} does not exist. Skipping.")

    # Calculate overall metrics across all folders
    overall_accuracy = sum(overall_metrics["accuracy"]) / len(overall_metrics["accuracy"]) if overall_metrics["accuracy"] else 0
    overall_precision = sum(overall_metrics["precision"]) / len(overall_metrics["precision"]) if overall_metrics["precision"] else 0
    overall_recall = sum(overall_metrics["recall"]) / len(overall_metrics["recall"]) if overall_metrics["recall"] else 0
    overall_f1 = sum(overall_metrics["f1_score"]) / len(overall_metrics["f1_score"]) if overall_metrics["f1_score"] else 0

    print("Overall Metrics Across All Folders:")
    print(f"  Accuracy: {overall_accuracy:.2f}")
    print(f"  Precision: {overall_precision:.2f}")
    print(f"  Recall: {overall_recall:.2f}")
    print(f"  F1-Score: {overall_f1:.2f}")

# Base folder containing the structured subtitles
base_folder_path = "/content/drive/MyDrive/sentiment/csv/subtitles"
process_all_folders(base_folder_path)
