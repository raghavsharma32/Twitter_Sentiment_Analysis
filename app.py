import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# --- Load model and tokenizer ---
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model('best_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# --- Prediction function ---
def predict_sentiment(text, model, tokenizer):
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len = 50
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)
    pred_class = sentiment_classes[pred.argmax()]
    return pred_class

# --- Streamlit UI ---
st.title("Twitter Sentiment Analysis App 🐦")
st.write("Enter a tweet or any text below to predict the sentiment:")

user_input = st.text_area("Input Text")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        prediction = predict_sentiment(user_input, model, tokenizer)
        st.success(f"**Predicted Sentiment:** {prediction}")
