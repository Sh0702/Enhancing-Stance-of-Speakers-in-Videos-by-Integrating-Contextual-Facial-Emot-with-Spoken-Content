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

## Conclusion

This project marks a significant advancement in multi-modal sentiment analysis. By combining textual sentiment with facial emotion recognition, the system is better equipped to understand nuanced speaker stances. Future developments in micro-expression analysis are expected to further improve the accuracy and depth of sentiment interpretation.

## Contributing

Feel free to check out the project, suggest improvements, or contribute! Together, we can push the boundaries of multi-modal sentiment and stance analysis.

