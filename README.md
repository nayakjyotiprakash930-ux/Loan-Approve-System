# Loan-Approve-System

## Creditwise Loan System

This project is a machine learning-based loan approval prediction system built around the provided loan dataset. It includes data loading, preprocessing, model training, evaluation, and a simple CLI entry point.

## Project Structure

- `main.py` - main application entry point
- `data/` - raw and processed datasets
- `models/` - trained model artifacts
- `notebooks/` - exploratory notebooks
- `tests/` - project tests
- `requirements.txt` - Python dependencies

## Features

- Data loading from CSV
- Missing-value handling
- Feature engineering and encoding
- Train/test split
- Logistic regression model training
- Accuracy, precision, recall, and F1 scoring
- Prediction pipeline and summary report

## Dataset

The project uses `loan_approval_data.csv`, which contains fields such as income, credit score, DTI ratio, savings, collateral value, loan purpose, and final loan approval status.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

## Example Usage

```python
from main import load_dataset, prepare_features, train_model

df = load_dataset('data/loan_approval_data.csv')
X, y = prepare_features(df)
model = train_model(X, y)
print(model)
```

## Notes

This is a starter version intended for academic and portfolio use. It can be extended with dashboards, explainability, and deployment APIs.
