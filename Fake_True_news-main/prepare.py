import pandas as pd
import csv 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump, load
import numpy as np
import re
import io
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
# Run these once if you haven't installed them. Uncomment and run if needed.
# nltk.download('wordnet')
# nltk.download('stopwords')


# --- Configuration ---
FAKE_FILE = 'Fake.csv'
TRUE_FILE = 'True.csv'
TEXT_COLUMN = 'text' 
TITLE_COLUMN = 'title'
MODEL_FILE = 'fake_news_detector_pipeline.joblib'
RANDOM_STATE = 42

# --- Custom Tokenizer for Advanced Preprocessing ---
lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))

def custom_tokenizer(text):
    """Tokenizes, removes non-alphabetic, removes stop words, and lemmatizes text."""
    if not isinstance(text, str):
        text = str(text) # Handle any non-string input just in case
        
    # 1. Remove non-alphabetic characters (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    
    # 2. Tokenize and convert to lower case
    tokens = text.lower().split()
    
    # 3. Lemmatize (reduce word to its root form) and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in english_stopwords]
    return tokens

# --- Data Loading and Cleaning ---

def load_data_robust(file_path, label):
    """
    Loads CSV after aggressively pre-cleaning to remove problem characters.
    """
    print(f"Attempting to load and clean {file_path}...")
    try:
        # 1. Read the entire file as one large string
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_data = f.read()

        # 2. Aggressively clean up internal quotes and special characters
        # Replace backslashes that might be escaping quotes
        cleaned_data = raw_data.replace('\\"', '"').replace('\\', ' ')
        # Replace non-standard newlines and tabs
        cleaned_data = cleaned_data.replace('\r\n', '\n').replace('\t', ' ')
        
        # Use StringIO to treat the cleaned string data as an in-memory file
        data_io = io.StringIO(cleaned_data)

        # 3. Use pandas to read the cleaned in-memory file
        # The Python engine is still used, but the cleaning makes it work
        df = pd.read_csv(
            data_io,
            engine='python',
            encoding='utf-8',
            on_bad_lines='warn', # This should now skip very few lines
            quoting=csv.QUOTE_MINIMAL 
        )
        df['label'] = label
        return df
    
    except Exception as e:
        print(f"CRITICAL ERROR loading {file_path}: {e}")
        return pd.DataFrame() 

print("Loading and combining data...")
fake_df = load_data_robust(FAKE_FILE, 0)
true_df = load_data_robust(TRUE_FILE, 1)

if fake_df.empty or true_df.empty:
    print("FATAL ERROR: Could not load one or both data files.")
    exit()

df = pd.concat([fake_df, true_df], ignore_index=True)

# 🎯 FIX 1: Fill NaN values in 'text' and 'title' columns with an empty string ('')
df[TEXT_COLUMN].fillna('', inplace=True)
df[TITLE_COLUMN].fillna('', inplace=True)

# Combine 'title' and 'text' for richer features
df['full_text'] = df[TITLE_COLUMN] + ' ' + df[TEXT_COLUMN]
FINAL_FEATURE = 'full_text'

df.drop(columns=['subject', 'date', TEXT_COLUMN, TITLE_COLUMN], errors='ignore', inplace=True)
df.drop_duplicates(subset=[FINAL_FEATURE], inplace=True)
print(f"Total unique news articles after cleaning: {len(df)}")


# --- Model Training Pipeline ---
print("Splitting data and training model...")

X_train, X_test, y_train, y_test = train_test_split(
    df[FINAL_FEATURE], df['label'], test_size=0.2, random_state=RANDOM_STATE
)

# Create a pipeline: Vectorizer (feature extraction) -> Classifier (prediction)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        tokenizer=custom_tokenizer, # Use the advanced tokenizer
        max_features=25000,   
        ngram_range=(1, 2),   
        dtype=np.float32 
    )),
    ('clf', LogisticRegression(
        C=0.1,                # 🎯 FIX 2: STRONGER Regularization to prevent overfitting
        solver='liblinear',
        random_state=RANDOM_STATE
    ))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
accuracy = pipeline.score(X_test, y_test)
print(f"\nModel training complete. (Total trained data: {len(X_train)})")
print(f"Test Accuracy: {accuracy:.4f}")

# Save the entire pipeline (vectorizer and model)
dump(pipeline, MODEL_FILE)
print(f"✅ Model saved as '{MODEL_FILE}'. Ready for prediction.")