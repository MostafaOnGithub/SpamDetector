
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# Download necessary NLTK data packages
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

from nlp_app.preprocessing import preprocess_text


# 1. Load the dataset
# Assuming your file is named 'spam.csv'
# Many spam datasets use 'latin-1' encoding
df = pd.read_csv('spam.csv', encoding='latin-1')

# 2. Basic Cleaning of the CSV structure
# Drop unnecessary columns if they exist and rename columns for clarity
df = df[['v1', 'v2']] 
df.columns = ['label', 'text']

# 3. Map labels to numbers 
# ham = 0, spam = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 4. Apply the NLTK preprocessing function
df['clean_text'] = df['text'].apply(preprocess_text)

# 5. Split the data (The "Corpus" you were missing)
X_train_corpus, X_test_corpus, y_train, y_test = train_test_split(
    df['clean_text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42
)

#--------------------------------------------------------------------Vectorization------------------------------------------

# 1. Initialize the Vectorizer
# We can pass our NLTK-based preprocess_text function here if needed
vectorizer = TfidfVectorizer(max_features=3000) 

# 2. Fit and Transform the data
# 'X_train' would be your preprocessed SMS messages
X_train_tfidf = vectorizer.fit_transform(X_train_corpus)

# 3. Transform the test data
X_test_tfidf = vectorizer.transform(X_test_corpus)

#-------------------------------------------------------------------Model Training----------------------------------------------

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

#------------------------------------------------------------------Evaluation---------------------------------------------------

# 3. Evaluation (Requirement 7: Present results using appropriate metrics)
predictions = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print(classification_report(y_test, predictions))

#-----------------------------------------------------------------Saving The Model-------------------------------------------------


import joblib

# 1. Save the trained Naive Bayes model
joblib.dump(model, 'spam_model.pkl')

# 2. Save the TF-IDF vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model and Vectorizer saved successfully!")