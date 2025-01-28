import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define paths
reports_dir = "/content/drive/MyDrive/sentiment/csv/reports"  # Path to the reports folder
platforms = ["instagram", "tiktok", "youtube"]  # Platforms to process
categories = ["alm", "blm"]  # Subcategories

# Initialize overall metrics
overall_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
platform_metrics = {}

# Iterate through each platform and category
for platform in platforms:
    platform_metrics[platform] = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
    for category in categories:
        folder_path = os.path.join(reports_dir, platform, category)
        category_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Filter out rows where ground_truth is NULL
                df = df.dropna(subset=['ground_truth'])

                # Skip files with no valid rows
                if df.empty:
                    print(f"No valid rows for {platform}/{category}/{file}. Skipping.")
                    continue

                # Extract ground truth and predictions
                ground_truth = df['ground_truth']  # Replace with actual ground truth column name
                predictions = df['third_category']  # Replace with actual predictions column name

                # Calculate metrics
                accuracy = accuracy_score(ground_truth, predictions)
                precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
                recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
                f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)

                # Store metrics for the video
                print(f"Metrics for {platform}/{category}/{file}:")
                print(f"  Accuracy: {accuracy:.2f}")
                print(f"  Precision: {precision:.2f}")
                print(f"  Recall: {recall:.2f}")
                print(f"  F1-Score: {f1:.2f}\n")

                # Add metrics to category-level lists
                category_metrics['accuracy'].append(accuracy)
                category_metrics['precision'].append(precision)
                category_metrics['recall'].append(recall)
                category_metrics['f1_score'].append(f1)

        # Aggregate category-level metrics and add to platform-level lists
        for metric in category_metrics:
            if category_metrics[metric]:  # Avoid division by zero
                platform_metrics[platform][metric].extend(category_metrics[metric])
                overall_metrics[metric].extend(category_metrics[metric])

        # Print category-level metrics
        print(f"Metrics for {platform}/{category}:")
        print(f"  Accuracy: {sum(category_metrics['accuracy']) / len(category_metrics['accuracy']):.2f}" if category_metrics['accuracy'] else "  Accuracy: N/A")
        print(f"  Precision: {sum(category_metrics['precision']) / len(category_metrics['precision']):.2f}" if category_metrics['precision'] else "  Precision: N/A")
        print(f"  Recall: {sum(category_metrics['recall']) / len(category_metrics['recall']):.2f}" if category_metrics['recall'] else "  Recall: N/A")
        print(f"  F1-Score: {sum(category_metrics['f1_score']) / len(category_metrics['f1_score']):.2f}" if category_metrics['f1_score'] else "  F1-Score: N/A")
        print()

# Aggregate platform-level metrics
for platform, metrics in platform_metrics.items():
    print(f"Metrics for {platform}:")
    print(f"  Accuracy: {sum(metrics['accuracy']) / len(metrics['accuracy']):.2f}" if metrics['accuracy'] else "  Accuracy: N/A")
    print(f"  Precision: {sum(metrics['precision']) / len(metrics['precision']):.2f}" if metrics['precision'] else "  Precision: N/A")
    print(f"  Recall: {sum(metrics['recall']) / len(metrics['recall']):.2f}" if metrics['recall'] else "  Recall: N/A")
    print(f"  F1-Score: {sum(metrics['f1_score']) / len(metrics['f1_score']):.2f}" if metrics['f1_score'] else "  F1-Score: N/A")
    print()

# Aggregate overall dataset metrics
print("Overall Metrics for All Videos:")
print(f"  Accuracy: {sum(overall_metrics['accuracy']) / len(overall_metrics['accuracy']):.2f}" if overall_metrics['accuracy'] else "  Accuracy: N/A")
print(f"  Precision: {sum(overall_metrics['precision']) / len(overall_metrics['precision']):.2f}" if overall_metrics['precision'] else "  Precision: N/A")
print(f"  Recall: {sum(overall_metrics['recall']) / len(overall_metrics['recall']):.2f}" if overall_metrics['recall'] else "  Recall: N/A")
print(f"  F1-Score: {sum(overall_metrics['f1_score']) / len(overall_metrics['f1_score']):.2f}" if overall_metrics['f1_score'] else "  F1-Score: N/A")
