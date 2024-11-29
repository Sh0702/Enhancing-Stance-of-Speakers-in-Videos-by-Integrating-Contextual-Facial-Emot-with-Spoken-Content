# Facial-Emotion-Recognition-for-Speaker-Stance-and-Narrative-Analysis

## Problem Addressed

Traditional textual sentiment analysis models often struggle with weakly worded or sarcastic sentences, leading to inaccuracies in understanding speaker intent. This is particularly evident when analyzing complex stances on sensitive topics like BLM and ALM. Text alone can be ambiguous, necessitating a more comprehensive approach to sentiment and stance detection.

## Solution Overview

This project integrates contextual **Facial Emotion Recognition** with **Textual Sentiment Analysis** to enhance understanding of speaker stances in videos. By combining textual data with facial emotion insights, the system captures a more holistic view of a speaker’s message, even when the text is unclear or sarcastic.

## Technologies and Tools Used

- **EMO-AffectNet**: These tools were utilized for **Facial Emotion Recognition** to detect emotions such as happiness, anger, sadness, and surprise from video data. This emotion layer adds critical context to the speaker's sentiment. The emotions will be clubbed into Positive(Happiness), Negative(Sadness, Fear, Disgust, Anger) and Neutral(Neutral, Suprise). 
  
- **MaskedABSA (Aspect-Based Sentiment Analysis)**: This model was integrated to perform aspect-based sentiment analysis on the textual data. It identifies specific aspects within the text and assesses sentiment toward each, providing more granular sentiment detection, particularly in ambiguous sentences.

## Data

- We have scrapped our own data from social media platforms such as TikTok, Instagram and YouTube.
- We possess 20 videos(10 from each camp) from each platform each ranging from 45 - 90s. Each of these videos consists of a single speaker speaking either pro BLM or pro ALM. 

## Ongoing Exploration

- Currently, we are annotating our scrapped data with facial emotions and marking spoken content as Positive and Negative.
- The facial emotions are annotated using a custom built API which allows the annotater to view the video frame by frame and annotate. The annotator views the video once and also is provided spoken content of the video in order to incorporate contextual annotation.
- Similarly the spoken content is also annotated by the annotator after watching the video.

## Upcoming work

- Once the annotation is completed, we will be focussing on validating the performance of our model using the following metrics.
    a. Accuracy = TP + TN / TP + TN + FP + FN
    b. Unweighted Average Recall = (1/N) * ∑ (TP for that instance) / (Total instances)
    c. F1-Score = (Precision * Recall) / (Precision + Recall)
    d. Area Under Curve = ∫ TPR(x)dx where TPR(x) - True Positive Rate for instance x - TPR = TP / (TP + FN)
  TP - True Positive TN - True Negative FP - False Positive FN - False Negative

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
python main.py
```
This will apply both models to your video data, allowing for enhanced sentiment and stance analysis by integrating facial emotion recognition with textual sentiment analysis.

### Conclusion

This project represents a significant step forward in multi-modal sentiment analysis, combining textual sentiment with facial emotion recognition to better understand nuanced speaker stances. As the system continues to evolve, including micro-expression analysis, we anticipate further improvements in understanding and interpreting complex communication.

### References

1. https://github.com/serengil/deepface
2. https://github.com/ElenaRyumina/EMO-AffectNetModel
3. https://github.com/tweetpie/masked-absa



