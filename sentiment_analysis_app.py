import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("punkt")

# Function for text preprocessing
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    return text

# Function for stemming
def stem_text(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()])

# Function for removing stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

# Streamlit app
st.title("Sentiment Analysis App")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file with 'review' and 'sentiment' columns", type=["csv"])

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Display dataset preview
    st.write("Dataset Preview:")
    st.write(data.head())

    # Check for required columns
    if "review" in data.columns and "sentiment" in data.columns:
        st.success("Dataset loaded successfully!")

        # Preprocessing options
        st.write("**Preprocessing Options**")
        preprocess_steps = st.multiselect(
            "Select preprocessing steps to apply:",
            ["Remove HTML Tags", "Remove Special Characters", "Convert to Lowercase", "Remove Stopwords", "Stemming"]
        )

        if st.button("Apply Preprocessing"):
            if "Remove HTML Tags" in preprocess_steps:
                data["review"] = data["review"].apply(lambda x: re.sub(r"<.*?>", "", str(x)))
            if "Remove Special Characters" in preprocess_steps:
                data["review"] = data["review"].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", str(x)))
            if "Convert to Lowercase" in preprocess_steps:
                data["review"] = data["review"].apply(lambda x: str(x).lower())
            if "Remove Stopwords" in preprocess_steps:
                data["review"] = data["review"].apply(remove_stopwords)
            if "Stemming" in preprocess_steps:
                data["review"] = data["review"].apply(stem_text)

            st.write("Preprocessed Data:")
            st.write(data.head())

        # Feature extraction
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data["review"])
        y = data["sentiment"].apply(lambda x: 1 if x.lower() == "positive" else 0)

        # Train-Test Split
        st.write("**Model Training**")
        train_ratio = st.slider("Select training data ratio:", 0.5, 0.9, 0.8)
        split_index = int(len(data) * train_ratio)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Train model
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)

        # Evaluate model
        predictions = logistic_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"**Model Accuracy:** {accuracy:.2f}")

        # Make predictions on custom input
        st.write("**Test Custom Review**")
        user_input = st.text_area("Enter a review to predict sentiment:")
        if user_input:
            preprocessed_input = user_input
            if "Remove HTML Tags" in preprocess_steps:
                preprocessed_input = re.sub(r"<.*?>", "", preprocessed_input)
            if "Remove Special Characters" in preprocess_steps:
                preprocessed_input = re.sub(r"[^a-zA-Z\s]", "", preprocessed_input)
            if "Convert to Lowercase" in preprocess_steps:
                preprocessed_input = preprocessed_input.lower()
            if "Remove Stopwords" in preprocess_steps:
                preprocessed_input = remove_stopwords(preprocessed_input)
            if "Stemming" in preprocess_steps:
                preprocessed_input = stem_text(preprocessed_input)

            input_vector = vectorizer.transform([preprocessed_input])
            prediction = logistic_model.predict(input_vector)
            sentiment = "Positive" if prediction[0] == 1 else "Negative"
            st.write(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.error("The dataset must have 'review' and 'sentiment' columns.")
