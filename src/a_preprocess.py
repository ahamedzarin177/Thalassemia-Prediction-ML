import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_dataset(csv_path: str, target_col: str = "Diagnosis"):
    """
    Loads dataset, splits X/y, and performs GLOBAL label encoding (for consistent metric labels).
    """
    dataset = pd.read_csv(csv_path)

    X = dataset.drop(target_col, axis=1)
    y = dataset[target_col]

    # GLOBAL label encoding
    le_global = LabelEncoder()
    y_encoded = le_global.fit_transform(y)
    classes_global = np.unique(y_encoded)

    return X, y, le_global, y_encoded, classes_global


