from joblib import load
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk 

# --- CRITICAL FIX: The custom tokenizer MUST be defined here for joblib to load the model ---

# Initialize lemmatizer and English stop words (must match the training script)
lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))

def custom_tokenizer(text):
    """
    Defines the exact tokenizer function used in the training script.
    This function's presence is required for the pipeline to load.
    """
    if not isinstance(text, str):
        text = str(text) 
        
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in english_stopwords]
    return tokens

# --- Configuration ---
MODEL_FILE = 'fake_news_detector_pipeline.joblib'

def new_detector(news_text: str):
    """
    Predicts if a news article is Fake (0) or True (1) using the saved model pipeline.
    """
    try:
        # Load the saved model pipeline (now it can find custom_tokenizer)
        pipeline = load(MODEL_FILE)
    except FileNotFoundError:
        return "Error: Model file 'fake_news_detector_pipeline.joblib' not found. Run training script first."
    except Exception as e:
        # Catch and return the specific error
        return f"CRITICAL LOAD ERROR: {e}. Check NLTK imports/installation."
    
    # Check if the result from 'load' is a string error message
    if isinstance(pipeline, str):
        return pipeline 
    
    # 1. Prepare the input
    input_data = [news_text]
    
    # 2. Get the prediction and probability
    try:
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
    except Exception as e:
        return f"Prediction Error: Failed to predict on new data. {e}"

    
    # 3. Return the structured result
    label_map = {0: 'FAKE', 1: 'TRUE'}
    
    return {
        'prediction': label_map.get(prediction, 'Unknown'),
        'confidence_true': f"{probabilities[1]*100:.2f}%",
        'confidence_fake': f"{probabilities[0]*100:.2f}%",
    }

# --- Example Usage ---

# Example 1: True news (the one that previously failed)
new_article = "The local city council approved a new measure today to increase public transportation funding by 15 percent, with the funds coming from a recently established infrastructure grant."

# Example 2: Highly sensational text
suspicious_article = "Doctors discover a single, simple pill that completely reverses aging in just 48 hours; Big Pharma is trying to silence the discovery!"

print("\n--- NEW DETECTOR TEST ---")

result_new = new_detector(new_article)
if isinstance(result_new, str):
    print(result_new)
else:
    print(f"Article 1: '{new_article[:65]}...'")
    print(f"Prediction: {result_new['prediction']} (Confidence: {result_new['confidence_true']})\n")

result_suspicious = new_detector(suspicious_article)
if isinstance(result_suspicious, str):
    print(result_suspicious)
else:
    print(f"Article 2: '{suspicious_article[:65]}...'")
    print(f"Prediction: {result_suspicious['prediction']} (Confidence: {result_suspicious['confidence_fake']})")