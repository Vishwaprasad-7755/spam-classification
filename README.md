# Spam Detection using Machine Learning

A complete end-to-end Machine Learning project for detecting spam messages using various classification algorithms. This project includes data preprocessing, model training, evaluation, and a Flask web application for real-time predictions.

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [API Endpoints](#api-endpoints)
- [Screenshots](#screenshots)

## ✨ Features

- **Multiple ML Models**: Naive Bayes, Logistic Regression, and Support Vector Machine
- **Text Preprocessing**: Lowercasing, punctuation removal, stopword removal, stemming
- **Feature Extraction**: CountVectorizer and TF-IDF comparison
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- **Web Interface**: Beautiful and responsive Flask web application
- **Prediction Probability**: Shows confidence score for predictions
- **Model Persistence**: Save and load trained models using pickle

## 🛠️ Tech Stack

- **Python 3.8+**
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **NLTK**: Natural language processing
- **Matplotlib/Seaborn**: Data visualization
- **Flask**: Web framework
- **HTML/CSS**: Frontend interface

## 📁 Project Structure

```
spam-detection-ml/
│
├── data/
│   └── spam.csv                 # Dataset file
│
├── notebooks/                    # Jupyter notebooks (optional)
│
├── models/                      # Saved models and vectorizers
│   ├── *_model.pkl             # Trained models
│   ├── *_vectorizer.pkl        # Feature vectorizers
│   ├── metadata.pkl            # Model metadata
│   └── confusion_matrices.png  # Visualization
│
├── templates/
│   └── index.html              # Web app template
│
├── static/
│   └── style.css              # Web app styling
│
├── train.py                    # Model training script
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🚀 Installation

### 1. Clone or Download the Project

```bash
git clone <repository-url>
cd spam-detection-ml
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

The script will automatically download required NLTK data on first run. Alternatively, you can download manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 📊 Dataset

This project uses the **SMS Spam Collection Dataset**. The dataset should be placed in the `data/` directory as `spam.csv`.

### Dataset Format

The CSV file should contain two columns:
- `label`: Message label (spam/ham)
- `message`: Text content of the message

### Download Dataset

You can download the dataset from:
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

**Note**: If your dataset has different column names (like `v1`/`v2` or `Category`/`Message`), the script will automatically handle the conversion.

## 💻 Usage

### Step 1: Prepare Dataset

Place your `spam.csv` file in the `data/` directory.

### Step 2: Train Models

Run the training script to preprocess data, train models, and save the best model:

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Split data into train/test sets (80/20)
- Extract features using TF-IDF
- Train three models (Naive Bayes, Logistic Regression, SVM)
- Evaluate and compare models
- Save the best model and vectorizer
- Generate confusion matrix visualization

### Step 3: Run Web Application

Start the Flask web server:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

Open your browser and navigate to the URL to use the spam detection interface.

## 📈 Model Performance

The training script evaluates all models and displays:

- **Accuracy**: Overall correctness
- **Precision**: Spam detection accuracy
- **Recall**: Ability to find all spam messages
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

The best model (based on F1-Score) is automatically saved for use in the web application.

### Example Output

```
Model Comparison
============================================================
Model                Accuracy  Precision  Recall  F1-Score
Naive Bayes          0.9785    0.9234     0.9123  0.9178
Logistic Regression  0.9812    0.9456     0.9234  0.9344
SVM                  0.9834    0.9567     0.9345  0.9454

Best Model: SVM
F1-Score: 0.9454
```

## 🌐 API Endpoints

### POST /predict

Predict if a message is spam.

**Request:**
```json
{
    "message": "Your message text here"
}
```

**Response:**
```json
{
    "prediction": "Spam",
    "probability": 0.9234,
    "message": "Your message text here"
}
```

### GET /health

Check application health and model status.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

## 🎨 Screenshots

The web application features:
- Clean, modern UI with gradient design
- Real-time prediction with loading states
- Visual probability indicators
- Responsive design for mobile and desktop
- Error handling and user feedback

## 🔧 Customization

### Change Feature Extraction Method

In `train.py`, modify the `extract_features` call:

```python
# Use CountVectorizer instead of TF-IDF
X_train_features, X_test_features, vectorizer = extract_features(
    X_train, X_test, method='count'
)
```

### Adjust Model Parameters

Edit model initialization in `train.py`:

```python
models = {
    'Naive Bayes': MultinomialNB(alpha=1.0),
    'Logistic Regression': LogisticRegression(max_iter=2000, C=1.0),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale')
}
```

### Modify Preprocessing

Update the `preprocess_text` function in both `train.py` and `app.py` to add custom preprocessing steps.

## 🐛 Troubleshooting

### Model Not Found Error

If you see "Model not loaded" error:
1. Ensure you've run `train.py` first
2. Check that `models/` directory contains the saved files
3. Verify `models/metadata.pkl` exists

### NLTK Data Error

If NLTK data is missing:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Port Already in Use

If port 5000 is busy, change it in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 👤 Author

Created as part of a Machine Learning project demonstrating end-to-end ML pipeline implementation.

---

**Note**: This is a demonstration project. For production use, consider:
- Adding authentication
- Implementing rate limiting
- Using a production WSGI server (e.g., Gunicorn)
- Adding logging and monitoring
- Implementing model versioning
- Adding unit tests
