import pickle
import re
import streamlit as st
from joblib import load
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download

# Download necessary NLTK resources
download('punkt_tab')
download('stopwords')

# Streamlit App Title
st.title("NLP Query Preprocessing and Classification App")

# Emojis and Emoticons Replacement
def replace_emojis_emoticons(text):
    emojis = {
        "🙂":" يبتسم ", "😂":" يضحك ", "💔":" قلب حزين ", "❤️":" حب ", "❤":" حب ", "😍":" حب ",
        "😭":" يبكي ", "😢":" حزن ", "😔":" حزن ", "♥":" حب ", "💜":" حب ", "😅":" يضحك ",
        "🙁":" حزين ", "💕":" حب ", "💙":" حب ", "😞":" حزين ", "😊":" سعادة ", "👏":" يصفق ",
        "👌":" احسنت ", "😴":" ينام ", "😀":" يضحك ", "😌":" حزين ", "🌹":" وردة ", "🙈":" حب "
        # Add more as needed
    }
    emoticons = {
        ":)" : "🙂", ":(" : "🙁", "xD" : "😆", ":=\(": "😭", ":'(": "😢", ":'‑(": "😢", "XD" : "😂", ":D" : "🙂"
        # Add more as needed
    }
    # Replace emojis with text
    for emoji, word in emojis.items():
        text = text.replace(emoji, word + " ")

    # Replace emoticons
    for emoticon, emoji in emoticons.items():
        text = re.sub(re.escape(emoticon), emoji + " ", text)

    return text

# Arabic Text Normalization
def normalize_text(text):
    text = re.sub(r'[\u064B-\u065F]', '', text)  # Remove diacritics
    text = re.sub(r'[إأآا]', 'ا', text)  # Normalize alef
    text = re.sub(r'ة', 'ه', text)  # Normalize taa marbuta
    text = re.sub(r'ى', 'ي', text)  # Normalize yaa
    text = re.sub(r'ؤ', 'و', text)  # Normalize waw
    text = re.sub(r'[ءأ]', 'ا', text)  # Normalize hamza
    text = re.sub(r'(.)\1+', r'\1', text)  # Remove duplicate letters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[!.,;\'\[\]\(\)\{\}\?:\-\u2013\u2014_]', '', text)  # Remove punctuation
    text = re.sub(r'[^ء-ي ]', '', text)  # Keep only Arabic letters
    return text

def remove_stopwords(text):
    if isinstance(text, list):
        return [word for word in text if word not in ['لا', 'ليس', 'ما', 'لم', 'لن', 'بالتأكيد', 'حقًا', 'إلا', 'غير', 'فقط', 'مجرد', 'رغم', 'لكن', 'بالرغم', 'مع ذلك', 'ربما', 'قد', 'حتى', 'سوى']]
    return text

# Text Preprocessing Function
stop_words = set(stopwords.words('arabic'))

def preprocess_text(text):
    text = replace_emojis_emoticons(text)  # Replace emojis and emoticons
    text = normalize_text(text)  # Normalize Arabic text
    tokens = word_tokenize(text)  # Tokenize text
    tokens = remove_stopwords(tokens)
  
    return ' '.join(tokens)

# Model and Vectorizer File Paths
model_file_path = r'C:\Users\Pro\Downloads\log_model.pkl'
vectorizer_file_path = r'C:\Users\Pro\Downloads\VecModel.pkl'

# Load the model using Pickle or Joblib
try:
    model = load(model_file_path)  # Try loading with Joblib
except Exception:
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)  # Fallback to Pickle

# Load the vectorizer using Pickle or Joblib
try:
    vectorizer = load(vectorizer_file_path)  # Try loading with Joblib
except Exception:
    with open(vectorizer_file_path, 'rb') as file:
        vectorizer = pickle.load(file)  # Fallback to Pickle


# Define label mapping
label_mapping = {
    -1: "Negative",
    1: "Positive"
}

# Input Query for Prediction
query = st.text_input("Enter your query for classification:")
if query:
    processed_query = preprocess_text(query)  # Preprocess the query
    query_vectorized = vectorizer.transform([processed_query])  # Vectorize the query
    prediction = model.predict(query_vectorized)  # Predict the class
    prediction_prob = model.predict_proba(query_vectorized)  # Get prediction probabilities
    
    # Get the numerical predicted class
    predicted_class = int(prediction[0])  # Ensure it is an integer
    
    # Get the probability of the predicted class
    probability = prediction_prob[0][model.classes_ == predicted_class][0]
    
    # Map the numerical prediction to a text label
    predicted_label = label_mapping.get(predicted_class, "Unknown")  # Use "Unknown" if not in mapping
    
    # Display the results
    st.write("### Prediction")
    st.write(f"The query is classified as: {predicted_class}")  # Print the numerical class (1 or -1)
    st.write(f"Prediction Confidence: {probability * 100:.2f}%")