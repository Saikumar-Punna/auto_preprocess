import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def typecast(df: pd.DataFrame, col_types: dict) -> pd.DataFrame:
    """Typecast columns according to the provided dictionary."""
    return df.astype(col_types)

def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""
    return df.drop_duplicates()

def handle_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Handle outliers by capping them at 1.5*IQR."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    return df

def create_dummies(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Create dummy variables for categorical columns."""
    return pd.get_dummies(df, columns=columns)

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', fill_value: float = None) -> pd.DataFrame:
    """Handle missing values in the DataFrame."""
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'constant':
        return df.fillna(fill_value)
    else:
        raise ValueError("Invalid strategy. Use 'mean', 'median', 'mode', or 'constant'.")


class AutoPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def typecast(self, col_types: dict):
        self.df = typecast(self.df, col_types)
        return self

    def handle_duplicates(self):
        self.df = handle_duplicates(self.df)
        return self

    def handle_outliers(self, columns: list):
        self.df = handle_outliers(self.df, columns)
        return self

    def create_dummies(self, columns: list):
        self.df = create_dummies(self.df, columns)
        return self

    def handle_missing_values(self, strategy: str = 'mean', fill_value: float = None):
        self.df = handle_missing_values(self.df, strategy, fill_value)
        return self

    def get_data(self):
        return self.df

from setuptools import setup, find_packages

setup(
    name='auto_preprocess',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn'
    ],
    author='Punna Saikumar',
    author_email='sk8627928@mail.com',
    description='A library for automatic data preprocessing tasks.',
    url='https://github.com/yourusername/auto_preprocess',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
