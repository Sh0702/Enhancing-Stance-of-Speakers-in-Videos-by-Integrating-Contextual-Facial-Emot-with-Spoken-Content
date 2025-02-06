# Enhancing the Stance of Speakers in Videos by Integrating Contextual Facial Emotions with Spoken Content

## Problem Addressed

Traditional stance detection methods rely primarily on analyzing spoken content. However, these methods struggle with weakly worded, sarcastic, or ambiguous sentences, leading to incorrect stance classification. Sentiment analysis techniques perform well with structured datasets but fail in real-world, dynamic settings like social media. This project enhances stance detection by integrating **Facial Emotion Recognition (FER)** with **Spoken Content Analysis**, providing a more accurate interpretation of speaker intent.

## Research Objective

This research aims to enhance stance detection by incorporating facial emotions into spoken content analysis. By leveraging diverse video data from social media platforms, the project seeks to overcome limitations in existing datasets that predominantly focus on extreme emotions while neglecting subtle emotional cues. The ultimate goal is to build a more robust, multimodal stance detection system.

## Dataset

A **custom dataset** has been curated from three major social media platforms: **TikTok, Instagram, and YouTube**.

### **Dataset Characteristics:**

- **60 videos** (20 from each platform), equally split between **Black Lives Matter (BLM) and All Lives Matter (ALM)** stances.
- **Diverse formats**, including reels, podcasts, and stage discussions.
- **Average duration**: 47.39 seconds (ranging from 14.54 - 95.37 seconds).
- **Single-speaker focus**: Only videos featuring a single speaker are included to eliminate external influences on stance detection.
- **Annotation**:
  - Subtitles were extracted using **Whisper AI** and manually verified.
  - Facial emotions were annotated using a custom-built API that allows frame-by-frame annotation while incorporating contextual speech content.
  - Spoken content was manually labeled for stance classification.
- **Focus on both extreme and subtle emotions**, providing a realistic testbed for stance detection.

## Methodology

### **Multimodal Approach**
The project integrates **Facial Emotion Recognition (FER)** and **Aspect-Based Sentiment Analysis (ABSA)** to improve stance detection accuracy:

1. **Facial Emotion Recognition (FER):**
   - Uses **EMO-AffectNet**, a two-part model with:
     - **Static backbone** for extreme emotion detection and feature extraction.
     - **Dynamic temporal model** for contextual emotion shifts across video frames.
   - Categorizes emotions into **Positive, Negative, and Neutral**.

2. **Textual Sentiment Analysis:**
   - Uses **MaskedABSA**, a model trained to analyze sentiment polarity by masking context-specific terms.
   - Capable of identifying **subtle sentiment nuances**, crucial for sarcasm and ambiguous speech.

3. **Fusion of Modalities:**
   - EMO-AffectNet provides **emotional context**, which enhances sentiment detection by MaskedABSA.
   - The combined approach enables **more accurate and nuanced stance classification**.

## Experimental Evaluation

### **Performance Metrics**
The model’s effectiveness is evaluated using the following metrics:

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **Unweighted Average Recall (UAR)** = (1/N) * ∑ (TP for each class / Total instances per class)
- **F1-Score** = (2 * Precision * Recall) / (Precision + Recall)
- **Area Under Curve (AUC)** = ∫ TPR(x)dx, where **TPR = TP / (TP + FN)**

### **Preliminary Results**

| Model          | Accuracy | Precision | Recall | F1-Score |
| -------------- | -------- | --------- | ------ | -------- |
| EMO-AffectNet  | 0.36     | 0.67      | 0.36   | 0.41     |
| MaskedABSA     | 0.51     | 0.51      | 0.59   | 0.54     |
| Combined Model | 0.49     | 0.49      | 0.64   | 0.56     |

### **Findings**
- **MaskedABSA** captures a wide range of sentiments but struggles with precision.
- **EMO-AffectNet** detects facial emotions with high precision but has lower recall.
- **The Combined Model** improves F1-Score, balancing recall and precision, making it **more effective for stance detection in real-world scenarios**.

## Future Work

- **Fine-Tuning**: Optimize hyperparameters for better performance on sarcastic and subtle speech.
- **Scalability**: Extend the dataset to include multilingual and cross-cultural videos.
- **Integration with Recommendation Systems**: Test how improved stance detection influences content recommendations on social media platforms.
- **Micro-Expression Analysis**: Explore the role of rapid facial muscle movements in stance classification.

## Conclusion

This research presents a **multimodal stance detection system** that integrates facial emotion recognition with spoken content analysis, addressing limitations in traditional sentiment-based methods. The system’s capability to detect **subtle emotional shifts** makes it a promising tool for applications such as **social media content categorization, targeted advertising, and misinformation detection**. Ongoing work focuses on refining annotation quality, improving model robustness, and scaling up dataset size to enhance real-world applicability.

## References

1. Ryumina, E., Dresvyanskiy, D., & Karpov, A. (2022). In search of a robust facial expressions recognition model: A large-scale visual cross-corpus study. *Neurocomputing, 514*, 435-450.
2. Lee, Y., Çetinkaya, Y., Külah, E., Toroslu, İ., & Davulcu, H. Masking the Bias: From Echo Chambers to Large Scale Aspect-Based Sentiment Analysis.
3. Lee, J., Kim, S., Kim, S., Park, J., & Sohn, K. (2019). Context-aware emotion recognition networks. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 10143-10152.
4. Tomar, P.S., Mathur, K., & Suman, U. (2024). Fusing facial and speech cues for enhanced multimodal emotion recognition. *International Journal of Information Technology, 16*(3), 1397-1405.
