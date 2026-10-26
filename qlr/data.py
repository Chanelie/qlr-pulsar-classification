# qlr/data.py
# ------------------------------------------------------------
# Dataset loader, standardization, and seed-stratified subset helper
# ------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler




# HTRU2 Loading

def load_htru2(csv_path: str | Path = "HTRU2.csv"):
    """Load HTRU2 dataset from CSV file with robust header handling."""
    path_obj = Path(csv_path)
    df = pd.read_csv(path_obj)
    # Add standard column names if missing
    if not isinstance(df.columns[0], str) or df.columns[0].startswith("Unnamed"):
        df.columns = ['mean_ip', 'std_ip', 'kurt_ip', 'skew_ip',
                      'mean_dmsnr', 'std_dmsnr', 'kurt_dmsnr', 'skew_dmsnr', 'class']
    # Ensure target is named 'class'
    if df.columns[-1] != 'class':
        df = df.rename(columns={df.columns[-1]: 'class'})
    print(f"HTRU2 loaded: {df.shape}, class dist: {df['class'].value_counts().to_dict()}")
    return df


class HTRU2Preprocessor:
    """
    Simple preprocessor for HTRU2:
      • standardizes 8 features
      • stratified train/test split by label
    """

    def __init__(self, test_size: float = 0.20, random_state: int = 42):
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def preprocess(self, df: pd.DataFrame, target_col: str = "class"):
        """
        Fit the StandardScaler on train data and return standardized
        (X_train, X_test, y_train, y_test).

        Parameters
        ----------
        df : pd.DataFrame
            Raw HTRU2 DataFrame.
        target_col : str
            Name of the target column (default: 'class').

        Returns
        -------
        X_train, X_test, y_train, y_test : np.ndarray
        """
        y = df[target_col].astype(int).values
        X = df.drop(columns=[target_col]).values.astype(np.float64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.is_fitted = True
        return X_train, X_test, y_train, y_test
    


def subset_indices_for(
    n_samples: int | None, seed: int, X_train: np.ndarray, y_train: np.ndarray
) -> np.ndarray:
    """
    Return stratified training indices for a given (n_samples, seed) on X_train/y_train.

    If `n_samples` is None or >= len(X_train), returns indices for the full training set.
    Otherwise uses a stratified subsample of size `n_samples` with deterministic randomness.
    """
    N = len(X_train)
    if n_samples is None or n_samples >= N:
        return np.arange(N)

    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=n_samples, random_state=seed
    )
    (idx, _), = sss.split(X_train, y_train)
    return idx