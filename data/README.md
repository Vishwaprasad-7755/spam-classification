# Data Directory

Place your `spam.csv` file in this directory.

## Dataset Information

The SMS Spam Collection Dataset should contain two columns:
- `label`: Message label (spam/ham)
- `message`: Text content of the message

## Download Links

You can download the dataset from:

1. **UCI Machine Learning Repository**
   - URL: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
   - Direct download: http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip

2. **Kaggle**
   - URL: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Dataset Format

The CSV file may have different column names. The training script (`train.py`) automatically handles:
- `v1`/`v2` → `label`/`message`
- `Category`/`Message` → `label`/`message`
- Standard `label`/`message` format

## Expected File Structure

```
data/
└── spam.csv
```

After placing the file, run `python train.py` to start training.
