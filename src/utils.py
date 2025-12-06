# src/utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path="data/india_housing_prices.csv"):
    df = pd.read_csv(path)
    return df

def train_test_split_xy(df, target_reg="Future_Price_5Y", target_clf="Good_Investment", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_reg, target_clf], errors='ignore')
    y_reg = df[target_reg] if target_reg in df.columns else None
    y_clf = df[target_clf] if target_clf in df.columns else None

    if y_reg is None and y_clf is None:
        raise ValueError("Targets not found in dataframe.")

    # If both exist, return both splits
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
    out = {"X_train": X_train, "X_test": X_test}
    if y_reg is not None:
        out["y_reg_train"], out["y_reg_test"] = train_test_split(y_reg, test_size=test_size, random_state=random_state)
    if y_clf is not None:
        out["y_clf_train"], out["y_clf_test"] = train_test_split(y_clf, test_size=test_size, random_state=random_state)
    return out
