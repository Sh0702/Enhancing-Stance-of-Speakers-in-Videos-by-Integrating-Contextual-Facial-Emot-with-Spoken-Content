# Towards Robust Speaker Stance Classification in Social Media Videos  
**A Multimodal Framework Integrating Facial Emotion Recognition with Aspect-Based Sentiment Analysis**

## 🧠 Problem Statement  
Traditional stance detection systems primarily rely on analyzing spoken content. However, they often fail to classify stances correctly in weakly labeled, sarcastic, or emotionally ambiguous content—common in real-world platforms like YouTube, TikTok, and Instagram. Existing sentiment analysis models excel on well-structured datasets but struggle to generalize in dynamic, context-rich settings such as social media.

## 🎯 Research Objective  
This research aims to build a robust, multimodal stance detection framework that integrates **Facial Emotion Recognition (FER)** with **Aspect-Based Sentiment Analysis (ABSA)**. The goal is to accurately classify speaker stance, especially when dealing with subtle or ambiguous expressions that challenge unimodal methods.

## 📦 Dataset  
A custom weakly-labeled dataset is being curated, consisting of video content focused on socio-political discourse.

### 🔍 Dataset Characteristics
- **Size**: 600 videos (300 Black Lives Matter, 300 All Lives Matter)
- **Sources**: TikTok, Instagram, YouTube  
- **Average Duration**: ~47.39 seconds (range: 14.54 – 95.37 seconds)
- **Content Types**: Reels, podcasts, stage discussions  
- **Speaker**: Single-speaker focused to minimize external bias  
- **Subtitles**: Generated using [Whisper AI](https://github.com/openai/whisper), manually verified  
- **Emotion Annotation**: Performed frame-by-frame using a custom-built annotation tool with contextual speech view

## 🔧 Methodology  

### 🧩 Models Used  
| Model | Description |
|-------|-------------|
| **MaskedABSA** | Text-only baseline using masked aspect-based sentiment analysis |
| **VLM Embeddings + Classifier** | Text + Vision baseline using vision-language model embeddings |
| **MaskedABSA + EMO-AffectNet** | Proposed multimodal approach integrating facial emotions with spoken content |
| **MaskedABSA + EMO-AffectNet + GAT** | Proposed improvement using Graph Attention Networks to model temporal-emotional dependencies |

### 🧠 Emotion Recognition (EMO-AffectNet)
- Combines static and dynamic modeling:
  - **Static Backbone**: For frame-wise emotion detection
  - **Temporal Model**: For tracking emotional shifts across video frames
- Classifies each frame into **Positive**, **Negative**, or **Neutral**

### 🗣️ Textual Sentiment (MaskedABSA)
- Uses masked attention around context-specific targets
- Designed to handle ambiguous, sarcastic, or weakly expressed sentiments

### 🔗 Fusion Strategies
- Emotion embeddings from EMO-AffectNet are integrated with sentiment scores from MaskedABSA
- A multimodal classifier is trained to leverage these fused features
- GAT layers are added to learn temporal and contextual dependencies across frames

## 📊 Experimental Evaluation  

### 📈 Metrics
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)  
- **Unweighted Average Recall (UAR)**  
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)  
- **AUC** (Area Under Curve)

### 🧪 Preliminary Results  
| Model                          | Accuracy | Precision | Recall | F1-Score |
|-------------------------------|----------|-----------|--------|----------|
| EMO-AffectNet (FER only)      | 0.36     | 0.67      | 0.36   | 0.41     |
| MaskedABSA (Text only)        | 0.51     | 0.51      | 0.59   | 0.54     |
| Combined (MaskedABSA + FER)   | 0.49     | 0.49      | 0.64   | 0.56     |
| + Graph Attention Networks    | TBD      | TBD       | TBD    | TBD      |

## 🔍 Key Findings  
- **MaskedABSA** captures sentiment but lacks precision in real-world, noisy inputs  
- **EMO-AffectNet** offers high emotional precision but suffers in temporal recall  
- **Multimodal Integration** provides better balance and interpretability in stance detection  
- **GAT Layers** (ongoing work) expected to enhance sequence-level understanding

## 🚧 Future Work  
- 📈 Fine-tune hyperparameters for sarcasm and ambiguity handling  
- 🔍 Scale up dataset size and diversity  
- 🧠 Train a feedforward neural network on fused embeddings  
- 📊 Introduce GAT-enhanced context modeling for emotion-sentiment propagation

## ✅ Applications  
- Social Media Content Categorization  
- Targeted Advertising and Audience Profiling  
- Misinformation and Bias Detection  

## 👨‍🔬 Thesis Advisor  
Dr. Hasan Davulcu  
Professor, School of Computing and Augmented Intelligence  
Arizona State University  
