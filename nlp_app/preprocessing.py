import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def preprocess_text(text):
    # 1. Lowercasing: Standardize text 
    text = text.lower()
    
    # 2. Cleaning: Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenization: Split text into words
    tokens = word_tokenize(text)
    
    # 4. Stopword Removal: Remove words like 'the', 'is', 'in'
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    # 5. Lemmatization: Reduce words to their root form (e.g., 'running' -> 'run') 
    lemmatizer = WordNetLemmatizer()
    lemmatized_output = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    
    # Join back into a single string for Vectorization
    return " ".join(lemmatized_output)