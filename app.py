"""
Flask Web Application for Spam Detection
This app provides a simple web interface to predict if a message is spam or not.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Global variables for model and vectorizer
model = None
vectorizer = None


def load_model():
    """
    Load the trained model and vectorizer from saved files.
    """
    global model, vectorizer
    
    try:
        # Load metadata to get paths
        metadata_path = 'models/metadata.pkl'
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Model metadata not found. Please train the model first.")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Load model
        model_path = metadata['model_path']
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load vectorizer
        vectorizer_path = metadata['vectorizer_path']
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        print("Model and vectorizer loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def preprocess_text(text):
    """
    Preprocess input text (same as training).
    
    Args:
        text (str): Input text message
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and stem
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    # Join tokens back
    return ' '.join(processed_tokens)


@app.route('/')
def index():
    """
    Render the main page.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if a message is spam or not.
    
    Returns:
        JSON response with prediction and probability
    """
    try:
        # Get message from request
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({
                'error': 'Please provide a message',
                'prediction': None,
                'probability': None
            }), 400
        
        # Check if model is loaded
        if model is None or vectorizer is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure model files exist.',
                'prediction': None,
                'probability': None
            }), 500
        
        # Preprocess message
        processed_message = preprocess_text(message)
        
        # Transform message using vectorizer
        message_vector = vectorizer.transform([processed_message])
        
        # Predict
        prediction = model.predict(message_vector)[0]
        
        # Get prediction probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(message_vector)[0]
            probability = float(proba[1])  # Probability of being spam
        
        # Convert prediction to label
        result = "Spam" if prediction == 1 else "Not Spam"
        
        return jsonify({
            'prediction': result,
            'probability': probability,
            'message': message
        })
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'prediction': None,
            'probability': None
        }), 500


@app.route('/health')
def health():
    """
    Health check endpoint.
    """
    model_loaded = model is not None and vectorizer is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })


if __name__ == '__main__':
    # Load model on startup
    print("Loading model...")
    if load_model():
        print("Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please train the model first using train.py")
