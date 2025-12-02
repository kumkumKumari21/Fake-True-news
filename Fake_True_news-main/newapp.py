from flask import Flask, render_template, request
import joblib
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# --- NLTK Setup (CRITICAL: Required for joblib to load the custom_tokenizer) ---
try:
    lemmatizer = WordNetLemmatizer()
    english_stopwords = set(stopwords.words('english'))
except LookupError:
    # Handle case where NLTK data is not present in the environment
    print("WARNING: NLTK data (wordnet/stopwords) not found. Attempting download...")
    nltk.download('wordnet')
    nltk.download('stopwords')
    lemmatizer = WordNetLemmatizer()
    english_stopwords = set(stopwords.words('english'))

def custom_tokenizer(text):
    """
    The exact tokenizer function defined in train_detector.py. 
    This MUST be present in the namespace before loading the pipeline.
    """
    if not isinstance(text, str):
        text = str(text) 
        
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in english_stopwords]
    return tokens


# --- Load the model ---
MODEL_FILE = 'fake_news_detector_pipeline.joblib' # Corrected file name
model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE)

try:
    # Load the model pipeline, which now relies on custom_tokenizer
    model = joblib.load(model_path)
    print(f"Model '{MODEL_FILE}' loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model from {model_path}. Ensure the file exists and NLTK data is present.")
    print(f"Error details: {e}")
    model = None # Set model to None to prevent crashes

app = Flask(__name__)

@app.route('/')
def home():
    """Renders the main input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if model is None:
         return render_template('index.html', prediction_text="❌ Model not loaded. Check server logs for errors.")

    try:
        news_text = request.form.get('text')
        
        if not news_text or news_text.strip() == "":
            return render_template('index.html', prediction_text="⚠️ Please enter some text.", result_class="bg-yellow-100 border-yellow-500")

        # Get prediction and probabilities
        prediction = model.predict([news_text])[0]
        probabilities = model.predict_proba([news_text])[0]
        
        # Determine result and confidence
        if prediction == 1:
            # True News
            result_label = "✅ REAL NEWS"
            confidence = probabilities[1] * 100
            result_class = "bg-green-100 border-green-500 text-green-800"
        else:
            # Fake News
            result_label = "❌ FAKE NEWS"
            confidence = probabilities[0] * 100
            result_class = "bg-red-100 border-red-500 text-red-800"

        # Format final prediction message
        prediction_text = f"{result_label} (Confidence: {confidence:.2f}%)"
        
        return render_template('index.html', 
                               prediction_text=prediction_text, 
                               result_class=result_class,
                               input_text=news_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error during prediction: {str(e)}", result_class="bg-gray-100 border-gray-500")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)