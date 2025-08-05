#Twitter Sentiment Analysis with LSTM

This project performs **sentiment analysis** on tweets using a **Bidirectional LSTM** model to classify text into **Positive**, **Neutral**, or **Negative** sentiment.

### 🚀 [Live Web App](https://twittersentimentanalysis-byraghavsharma.streamlit.app/)

---

## Project Highlights

- ✅ Trained deep learning model using Bidirectional LSTM
- ✅ Text vectorization using `Tokenizer` and padding
- ✅ Classifies tweets into Positive, Negative, or Neutral
- ✅ Real-time predictions via a Streamlit web app
- ✅ Cleaned and preprocessed Twitter dataset

---

## File Structure

twitter_sentiment_analysis/
├── app.py # Streamlit app for sentiment prediction
├── best_model.h5 # Trained Bidirectional LSTM model
├── tokenizer.pkl # Tokenizer for text preprocessing
├── eda(1).py # Data cleaning, preprocessing & training
├── requirements.txt # Python package dependencies
└── README.md # Project documentation

## Model Overview

- **Embedding Layer** for word vector representation  
- **Bidirectional LSTM** to capture context from both directions  
- **Dense Layers** for classification  
- **Softmax Output** for multiclass prediction
