"""
Spam Detection Model Training Script
This script handles data preprocessing, model training, evaluation, and model saving.
"""

import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize stemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def load_data(file_path):
    """
    Load the spam dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        # Try different possible column names
        df = pd.read_csv(file_path, encoding='latin-1')
        
        # Handle different CSV formats
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df.rename(columns={'v1': 'label', 'v2': 'message'})
        elif 'Category' in df.columns and 'Message' in df.columns:
            df = df.rename(columns={'Category': 'label', 'Message': 'message'})
        
        # Keep only label and message columns
        df = df[['label', 'message']].copy()
        
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def preprocess_text(text):
    """
    Preprocess a single text message.
    
    Steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Tokenize
    4. Remove stopwords
    5. Stem words
    
    Args:
        text (str): Input text message
        
    Returns:
        str: Preprocessed text
    """
    if pd.isna(text):
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


def preprocess_data(df):
    """
    Preprocess the entire dataset.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    print("Preprocessing data...")
    
    # Handle missing values
    print(f"Missing values before: {df.isnull().sum().sum()}")
    df = df.dropna()
    print(f"Missing values after: {df.isnull().sum().sum()}")
    
    # Convert labels (spam=1, ham=0)
    df['label'] = df['label'].map({'spam': 1, 'ham': 0, 'Spam': 1, 'Ham': 1})
    
    # Handle any remaining non-numeric labels
    df = df[df['label'].isin([0, 1])]
    
    # Preprocess messages
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    print(f"Data preprocessing completed. Final shape: {df.shape}")
    return df


def extract_features(X_train, X_test, method='tfidf'):
    """
    Extract features using CountVectorizer or TF-IDF.
    
    Args:
        X_train (pd.Series): Training messages
        X_test (pd.Series): Test messages
        method (str): 'count' for CountVectorizer or 'tfidf' for TF-IDF
        
    Returns:
        tuple: (train_features, test_features, vectorizer)
    """
    print(f"Extracting features using {method.upper()}...")
    
    if method == 'count':
        vectorizer = CountVectorizer(max_features=5000, min_df=2, max_df=0.95)
    else:  # tfidf
        vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95)
    
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    print(f"Feature extraction completed. Feature matrix shape: {X_train_features.shape}")
    return X_train_features, X_test_features, vectorizer


def train_models(X_train, X_test, y_train, y_test, vectorizer):
    """
    Train multiple models and compare their performance.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        vectorizer: Fitted vectorizer
        
    Returns:
        dict: Dictionary containing trained models and their metrics
    """
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("Training Models")
    print("="*60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predict_proba': y_pred_proba is not None
        }
        
        print(f"{name} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    return results


def plot_confusion_matrices(results, y_test):
    """
    Plot confusion matrices for all models.
    
    Args:
        results (dict): Dictionary containing model results
        y_test: Test labels
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('models/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrices saved to models/confusion_matrices.png")
    plt.close()


def compare_models(results):
    """
    Compare models and return the best one.
    
    Args:
        results (dict): Dictionary containing model results
        
    Returns:
        tuple: (best_model_name, best_model_object)
    """
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    # Create comparison dataframe
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n", comparison_df.to_string(index=False))
    
    # Find best model based on F1-score (balanced metric)
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
    
    return best_model_name, best_model


def save_model(model, vectorizer, model_name, method):
    """
    Save the trained model and vectorizer.
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        model_name (str): Name of the model
        method (str): Feature extraction method used
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = f'models/{model_name.lower().replace(" ", "_")}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")
    
    # Save vectorizer
    vectorizer_path = f'models/{method}_vectorizer.pkl'
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {vectorizer_path}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'method': method,
        'model_path': model_path,
        'vectorizer_path': vectorizer_path
    }
    
    metadata_path = 'models/metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to {metadata_path}")


def main():
    """
    Main function to run the complete ML pipeline.
    """
    print("="*60)
    print("Spam Detection Model Training")
    print("="*60)
    
    # Load data
    data_path = 'data/spam.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure spam.csv is in the data/ directory.")
        return
    
    df = load_data(data_path)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Split data
    X = df['processed_message']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Extract features using TF-IDF (better performance typically)
    X_train_features, X_test_features, vectorizer = extract_features(
        X_train, X_test, method='tfidf'
    )
    
    # Train models
    results = train_models(X_train_features, X_test_features, y_train, y_test, vectorizer)
    
    # Plot confusion matrices
    plot_confusion_matrices(results, y_test)
    
    # Compare models and get best one
    best_model_name, best_model = compare_models(results)
    
    # Save best model
    save_model(best_model, vectorizer, best_model_name, 'tfidf')
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
