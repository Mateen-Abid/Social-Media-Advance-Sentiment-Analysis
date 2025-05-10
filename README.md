# Emotion Detection from Text using BERT

This project is a Final Year University Project that detects emotions from short texts like tweets, reviews, or user comments. It uses a deep learning model called **BERT (Bidirectional Encoder Representations from Transformers)**, which has been fine-tuned on a labeled dataset called **GoEmotions** developed by Google.

The system can predict one of 28 different emotions like joy, sadness, anger, fear, love, curiosity, surprise, and more — along with a confidence score. Charts are also included to visualize top predictions.

## 📘 About the Project

The goal of this project is to use NLP and machine learning to automatically detect emotions in text, which has many practical uses — such as monitoring social media sentiment, analyzing customer reviews, or building emotionally aware chatbots.

This project includes:
- Preprocessing and tokenization of input text
- Fine-tuning of a pre-trained BERT model
- Prediction of emotions with confidence scores
- Visualization (bar/pie charts)
- Compatibility with Google Colab

## 📁 Project Structure

emotion-detection/
│
├── train_model.py # Trains the model on GoEmotions dataset
├── emotion_predictor.py # Predicts emotion from input text
├── evaluate_model.py # Evaluates model performance
├── visualize.py # Creates emotion charts
├── notebook.ipynb # Full project code (Colab/Jupyter)
├── requirements.txt # Python libraries required
└── README.md # Project documentation


## 🧠 Emotions Covered (28 Classes)

admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral


## 🚀 How to Run

### 1. Clone the Project
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
2. Install Required Libraries
bash
Copy
Edit
pip install -r requirements.txt
3. Predict an Emotion
from emotion_predictor import predict_emotion

text = "I love the new AI tool!"
emotion, confidence = predict_emotion(text)
print(f"Predicted Emotion: {emotion} ({confidence * 100:.1f}%)")
📊 Example Output
Text: "Loving the new AI art tool!"
→ Predicted Emotion: love (69.3%)
Visualization: Bar and pie charts display top 5 emotions.

📈 Model Training Summary
Model Used: BERT-base (uncased)

Dataset: GoEmotions (from Hugging Face)

Training Method: Hugging Face Trainer API

Evaluation Loss: ~1.36

Accuracy: ~88%

Training Time: ~47 minutes on Google Colab

✅ Future Work
Enable multi-label emotion detection

Connect to Twitter or Reddit APIs for real-time text input

Add a web dashboard using Streamlit or Flask

Build a mobile app with the model backend

