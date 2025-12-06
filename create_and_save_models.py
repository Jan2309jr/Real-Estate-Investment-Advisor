# create_and_save_models.py
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# import local preprocessing utilities
from src.preprocess import basic_cleaning, fill_missing_domain, feature_engineering, create_targets, build_preprocessor
import pandas as pd

DATA_PATH = os.path.join("data", "india_housing_prices.csv")
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# load data
df = pd.read_csv(DATA_PATH)
df = basic_cleaning(df)
df = fill_missing_domain(df)
df = feature_engineering(df)
df = create_targets(df)

# build preprocessor (column transformer)
preprocessor, numeric_cols, cat_cols = build_preprocessor(df)

# Prepare X and y
X = df.drop(columns=["Future_Price_5Y", "Good_Investment"], errors="ignore")
y_reg = df["Future_Price_5Y"]
y_clf = df["Good_Investment"]

# simple train/test split (we just need a fitted model to load in Streamlit)
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

# classification pipeline
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_pipe = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
print("Training classifier...")
clf_pipe.fit(X_train, y_clf_train)
joblib.dump(clf_pipe, os.path.join(SAVE_DIR, "classifier_pipeline.joblib"))
print("Saved classifier_pipeline.joblib")

# regression pipeline
reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
reg_pipe = Pipeline([("preprocessor", preprocessor), ("reg", reg)])
print("Training regressor...")
reg_pipe.fit(X_train, y_reg_train)
joblib.dump(reg_pipe, os.path.join(SAVE_DIR, "regressor_pipeline.joblib"))
print("Saved regressor_pipeline.joblib")

# Also save preprocessor object separately (app expects it maybe)
joblib.dump(preprocessor, os.path.join(SAVE_DIR, "preprocessor.joblib"))
print("Saved preprocessor.joblib")

print("All done. Models are in the 'models' folder.")
