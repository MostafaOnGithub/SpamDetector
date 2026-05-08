import joblib
import os
from django.conf import settings
from .preprocessing import preprocess_text
from django.shortcuts import render


MODEL_PATH = os.path.join(settings.BASE_DIR, 'saved_models', 'spam_model.pkl')
VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'saved_models', 'tfidf_vectorizer.pkl')


if os.path.exists(VECTORIZER_PATH):
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    raise FileNotFoundError(f"Missing vectorizer at: {VECTORIZER_PATH}")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Missing model at: {MODEL_PATH}")

def index(request):
    result = None
    original_text = ""
    
    if request.method == "POST":
        original_text = request.POST.get('message', '')
        
        # 1. Preprocess
        cleaned_msg = preprocess_text(original_text)
        
        # 2. Vectorize
        vectorized_msg = vectorizer.transform([cleaned_msg])
        
        # 3. Predict
        prediction = model.predict(vectorized_msg)
        result = "SPAM" if prediction[0] == 1 else "HAM (Legitimate)"

    return render(request, 'index.html', {
        'prediction': result,
        'original_text': original_text
    })