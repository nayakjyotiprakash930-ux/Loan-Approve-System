import os
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = os.getenv('DATA_PATH', 'data/loan_approval_data.csv')


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the loan application dataset from CSV."""
    df = pd.read_csv(path)
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target, and clean missing values for model input."""
    target_col = 'Loan_Approved'
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in the dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols),
        ],
        remainder='drop'
    )

    X_processed = preprocessor.fit_transform(X)
    X_df = pd.DataFrame(X_processed)
    y_clean = y.fillna('Unknown').astype(str)
    return X_df, y_clean


def train_model(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """Train a logistic regression model on the prepared dataset."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('Precision:', precision_score(y_test, predictions, zero_division=0))
    print('Recall:', recall_score(y_test, predictions, zero_division=0))
    print('F1 Score:', f1_score(y_test, predictions, zero_division=0))
    print(classification_report(y_test, predictions, zero_division=0))

    return model


def main() -> None:
    """Run the full pipeline for the Creditwise Loan System."""
    df = load_dataset(DATA_PATH)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    X, y = prepare_features(df)
    model = train_model(X, y)
    print('Training complete.')
    print(f'Model type: {type(model).__name__}')


if __name__ == '__main__':
    main()
