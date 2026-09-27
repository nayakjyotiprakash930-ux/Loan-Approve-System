import pandas as pd

from main import load_dataset, prepare_features


def test_load_dataset_returns_dataframe():
    df = load_dataset('data/loan_approval_data.csv')
    assert isinstance(df, pd.DataFrame)
    assert 'Loan_Approved' in df.columns


def test_prepare_features_returns_expected_columns():
    df = load_dataset('data/loan_approval_data.csv')
    X, y = prepare_features(df)
    assert X.shape[0] == y.shape[0]
    assert y.name == 'Loan_Approved'
