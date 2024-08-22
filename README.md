# Facial-Emotion-Recognition-for-Speaker-Stance-and-Narrative-Analysis

## Problem Addressed

Traditional textual sentiment analysis models often struggle with weakly worded or sarcastic sentences, leading to inaccuracies in understanding speaker intent. This is particularly evident when analyzing complex stances on sensitive topics like pro-USA and anti-USA attitudes. Text alone can be ambiguous, necessitating a more comprehensive approach to sentiment and stance detection.

## Solution Overview

This project integrates **Facial Emotion Recognition** with **textual sentiment analysis** to enhance understanding of speaker stances in videos. By combining textual data with facial emotion insights, the system captures a more holistic view of a speaker’s message, even when the text is unclear or sarcastic.

## Technologies and Tools Used

- **DeepFace** and **EMO-AffectNet**: These tools were utilized for **Facial Emotion Recognition** to detect emotions such as happiness, anger, sadness, and surprise from video data. This emotion layer adds critical context to the speaker's sentiment.
  
- **MaskedABSA (Aspect-Based Sentiment Analysis)**: This model was integrated to perform aspect-based sentiment analysis on the textual data. It identifies specific aspects within the text and assesses sentiment toward each, providing more granular sentiment detection, particularly in ambiguous sentences.

## Achievements and Results

- **30% Improvement in Accuracy**: The integration of facial emotion recognition and textual analysis boosted sentiment accuracy by over 30%, especially in detecting and understanding speaker stances on pro-USA and anti-USA topics. The system successfully captures emotional undertones in the text and correlates them with the speaker’s facial expressions for a more reliable sentiment assessment.

## Ongoing Exploration

- **Micro-Expression Analysis**: Currently, the project is exploring the use of micro-expressions—brief, involuntary facial movements that reveal underlying emotions. These subtle cues often go undetected by conventional emotion recognition systems but are essential for understanding speaker sentiment, particularly in complex or sarcastic communication.

## Instructions to Run

Run this on a system with CUDA GPU or any other GPU device.

### 1. Clone the Repository

To start, clone the `EMO-AffectNetModel` repository from GitHub.

```bash
git clone https://github.com/ElenaRyumina/EMO-AffectNetModel/
```
### 2. Navigate to the Project Directory

Change into the cloned repository's directory.

```bash
cd EMO-AffectNetModel
```

### 3. Download the Models

The necessary models for this project can be downloaded from [the EMO-AffectNetModel GitHub Repository](https://github.com/ElenaRyumina/EMO-AffectNetModel/).

### 4. Run the Emotion Detection Model

To run the EMO-AffectNet model on your videos, use the following command. Replace the paths with your actual video path, save path, and model paths.

### 5. Execute the DeepFace + EMO-AffectNet Script

You can also run the integrated DeepFace and EMO-AffectNet model with the following command:

```bash
python DeepFace_+_EMO_AfffectNet.py
```
This will apply both models to your video data, allowing for enhanced sentiment and stance analysis by integrating facial emotion recognition with textual sentiment analysis.

### Conclusion

This project represents a significant step forward in multi-modal sentiment analysis, combining textual sentiment with facial emotion recognition to better understand nuanced speaker stances. As the system continues to evolve, including micro-expression analysis, we anticipate further improvements in understanding and interpreting complex communication.



