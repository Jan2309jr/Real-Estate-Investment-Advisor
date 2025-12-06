# src/app_streamlit.py
"""
Streamlit app for Real Estate Investment Advisor - updated to align runtime user inputs
with the pipeline's expected columns to avoid "columns are missing" errors.

Key changes:
- align_user_df_to_pipeline: attempts to load canonical expected columns from disk (models/expected_input_columns.json)
  or infer them from the pipeline's preprocessor. Adds missing columns with sensible defaults and reindexes.
- improved error handling and clearer messages when prediction fails.
- retains feature importance display but guarded against many common failure modes.
"""

import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
# --- Add at the very top of app_streamlit.py, before importing joblib or loading models ---
import importlib

try:
    _ct_mod = importlib.import_module("sklearn.compose._column_transformer")
except Exception:
    _ct_mod = None

if _ct_mod is not None and not hasattr(_ct_mod, "_RemainderColsList"):
    # Minimal compatible placeholder: behaves like a list for unpickling.
    class _RemainderColsList(list):
        # ensure pickling/unpickling behaves like a list
        def __reduce__(self):
            return (list, (list(self),))
    setattr(_ct_mod, "_RemainderColsList", _RemainderColsList)
# --- end shim ---

MODEL_DIR = "models"
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier_pipeline.joblib")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "regressor_pipeline.joblib")
EXPECTED_COLS_PATH = os.path.join(MODEL_DIR, "expected_input_columns.json")  # optional file created at training time

@st.cache_resource
def load_models():
    """Load saved classifier and regressor pipelines (joblib)."""
    if not os.path.exists(CLASSIFIER_PATH) or not os.path.exists(REGRESSOR_PATH):
        raise FileNotFoundError(f"Model files not found in {MODEL_DIR}. Expected: {CLASSIFIER_PATH}, {REGRESSOR_PATH}")
    clf = joblib.load(CLASSIFIER_PATH)
    reg = joblib.load(REGRESSOR_PATH)
    return clf, reg

def user_input_form():
    """Collect user inputs via the sidebar and return as a single-row DataFrame."""
    st.sidebar.header("Property details")
    city = st.sidebar.text_input("City", "Bengaluru")
    property_type = st.sidebar.selectbox("Property Type", ["Apartment", "Villa", "House", "Independent"])
    bhk = st.sidebar.number_input("BHK", min_value=1, max_value=10, value=2)
    size = st.sidebar.number_input("Size in SqFt", min_value=200, value=900)
    price_lakhs = st.sidebar.number_input("Current Price (in Lakhs)", min_value=1.0, value=60.0, format="%.2f")
    furnished = st.sidebar.selectbox("Furnished Status", ["Unfurnished","Semi","Fully"])
    floor_no = st.sidebar.number_input("Floor No", value=1)
    total_floors = st.sidebar.number_input("Total Floors", value=4)
    age = st.sidebar.number_input("Age of Property (years)", value=5)
    nearby_schools = st.sidebar.number_input("Nearby Schools (count/rating)", value=3)
    nearby_hospitals = st.sidebar.number_input("Nearby Hospitals", value=2)
    transport = st.sidebar.selectbox("Public Transport Accessibility", ["Poor","Medium","Good","Excellent"])
    parking = st.sidebar.number_input("Parking Spots", value=1)
    amenities = st.sidebar.text_input("Amenities (comma separated)", "Gym,Pool")
    availability = st.sidebar.selectbox("Availability_Status", ["Available","Under Construction","Sold","Ready to move"])
    owner_type = st.sidebar.selectbox("Owner_Type", ["Individual","Builder","Agent"])
    rera = st.sidebar.checkbox("RERA Registered?")

    input_dict = {
        "City": city,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Price_in_Lakhs": price_lakhs,
        "Furnished_Status": furnished,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Age_of_Property": age,
        "Nearby_Schools": nearby_schools,
        "Nearby_Hospitals": nearby_hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Amenities": amenities,
        "Availability_Status": availability,
        "Owner_Type": owner_type,
        "RERA": int(rera),
    }
    return pd.DataFrame([input_dict])

def load_expected_columns_from_file(path=EXPECTED_COLS_PATH):
    """Try to load expected input column list saved at training time."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            # allow wrapper dict { "expected_columns": [...] }
            if isinstance(data, dict) and "expected_columns" in data and isinstance(data["expected_columns"], list):
                return data["expected_columns"]
        except Exception:
            return None
    return None

def infer_expected_columns_from_pipeline(pipeline):
    """Infer expected input column names from a sklearn ColumnTransformer preprocessor inside pipeline."""
    expected_cols = []

    pre = None
    try:
        pre = pipeline.named_steps.get("preprocessor", None)
    except Exception:
        pre = None

    if pre is None:
        # maybe the pipeline itself is a ColumnTransformer or has feature_names_in_
        if hasattr(pipeline, "feature_names_in_"):
            return list(getattr(pipeline, "feature_names_in_"))
        return []

    # ColumnTransformer: transformers_ is a list of tuples (name, transformer, columns)
    try:
        for name, transformer, cols in pre.transformers_:
            # handle common cases where cols is list/tuple/np.ndarray of column names
            if isinstance(cols, (list, tuple, np.ndarray)):
                expected_cols.extend(list(cols))
            elif isinstance(cols, str):
                expected_cols.append(cols)
            else:
                # cols might be slice or callable or 'remainder' - skip in that case
                # if transformer supports get_feature_names_out and cols is a list, we'd handle above
                pass
    except Exception:
        # fallback to any available attribute
        pass

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for c in expected_cols:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped

def sensible_default_for_column(col_name):
    """Return a sensible default value for a missing column based on heuristics."""
    cname = col_name.lower()
    # numeric-ish heuristics
    if any(token in cname for token in ["count", "score", "density", "spots", "no", "number", "age", "years", "year", "rating"]):
        return 0
    if any(token in cname for token in ["price", "per", "sqft", "lakhs", "amount", "value", "rent"]):
        return 0.0
    if any(token in cname for token in ["flag", "rera", "ready", "facing", "yesno", "binary"]):
        return 0
    # fallback to NaN for unknowns (often categorical)
    return np.nan

def align_user_df_to_pipeline(df: pd.DataFrame, pipeline):
    """
    Return a DataFrame reindexed to the pipeline's expected input columns.
    Steps:
    1. Try to load canonical expected columns from models/expected_input_columns.json (recommended at training).
    2. If not available, infer from pipeline.preprocessor.transformers_.
    3. Add missing columns with heuristics and reorder columns to match expected.
    """
    # 1) try file
    expected_cols = load_expected_columns_from_file()
    # 2) infer from pipeline if file missing
    if not expected_cols:
        try:
            expected_cols = infer_expected_columns_from_pipeline(pipeline)
        except Exception:
            expected_cols = []

    # 3) fallback to pipeline.feature_names_in_ if still empty
    if not expected_cols:
        if hasattr(pipeline, "feature_names_in_"):
            expected_cols = list(getattr(pipeline, "feature_names_in_"))
        else:
            expected_cols = []

    # If still empty, return original df as we cannot align
    if not expected_cols:
        return df.copy()

    # Add missing expected columns with defaults
    df_copy = df.copy()
    missing = [c for c in expected_cols if c not in df_copy.columns]
    for c in missing:
        df_copy[c] = sensible_default_for_column(c)

    # If the user supplied extra columns not expected by model, drop them
    df_aligned = df_copy.reindex(columns=expected_cols)

    return df_aligned

def show_feature_importances_if_available(clf_pipeline):
    """Try to extract and display feature importances for a classifier pipeline if possible."""
    try:
        pre = clf_pipeline.named_steps.get("preprocessor", None)
        clf_step = clf_pipeline.named_steps.get("clf", None) or clf_pipeline.named_steps.get("classifier", None)
        if pre is None or clf_step is None:
            st.write("Feature importances not available (pipeline structure not recognized).")
            return

        # numeric columns: commonly first transformer; categorical: second with onehot
        # This part is heuristic and will fail for complex pipelines, so guard with try/except.
        num_features = []
        cat_features = []
        try:
            for name, transformer, cols in pre.transformers_:
                if isinstance(cols, (list, tuple, np.ndarray)):
                    if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
                        cat_features.extend(list(cols))
                    else:
                        num_features.extend(list(cols))
                elif isinstance(cols, str):
                    num_features.append(cols)
        except Exception:
            pass

        feature_names = []
        # Try to get onehot names if possible
        try:
            # Find the categorical transformer which has OneHotEncoder with get_feature_names_out
            for name, transformer, cols in pre.transformers_:
                if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
                    ohe = transformer.named_steps["onehot"]
                    # if cols is iterable
                    if isinstance(cols, (list, tuple, np.ndarray)):
                        ohe_names = list(ohe.get_feature_names_out(cols))
                    else:
                        ohe_names = []
                    feature_names.extend(ohe_names)
                elif isinstance(cols, (list, tuple, np.ndarray)):
                    # for numeric, just append names
                    feature_names.extend(list(cols))
        except Exception:
            # fallback: combine collected lists
            feature_names = list(num_features) + list(cat_features)

        if not feature_names and hasattr(clf_step, "feature_names_in_"):
            feature_names = list(clf_step.feature_names_in_)

        if not feature_names:
            st.write("Could not determine feature names for importances.")
            return

        if hasattr(clf_step, "feature_importances_"):
            imps = clf_step.feature_importances_
            if len(imps) != len(feature_names):
                # If lengths mismatch, try trimming/extension heuristics
                min_len = min(len(imps), len(feature_names))
                imps = imps[:min_len]
                feature_names = feature_names[:min_len]
            fi = pd.DataFrame({"feature": feature_names, "importance": imps})
            fi = fi.sort_values("importance", ascending=False).head(30)
            st.table(fi.set_index("feature"))
        else:
            st.write("Model does not expose feature_importances_.")
    except Exception as e:
        st.write("Could not compute feature importances:", str(e))

def main():
    st.title("Real Estate Investment Advisor")
    st.markdown("Predict if property is a **Good Investment** and estimate **Price after 5 years**.")

    # Load models
    try:
        clf, reg = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return

    st.sidebar.header("Model controls")
    show_feature_importance = st.sidebar.checkbox("Show feature importances (classifier)", value=True)

    user_df = user_input_form()
    st.subheader("Input summary")
    st.write(user_df.T)

    if st.button("Predict"):
        # Align user input to pipeline expected columns
        aligned_user_df = align_user_df_to_pipeline(user_df, clf)

        # Debug: show which columns were added/used (helpful when expected list saved)
        expected_cols_from_file = load_expected_columns_from_file()
        if expected_cols_from_file:
            st.info(f"Using expected columns loaded from file (first 20 shown): {expected_cols_from_file[:20]}")
        else:
            # try to display inferred columns (don't raise if empty)
            inferred = infer_expected_columns_from_pipeline(clf)
            if inferred:
                st.info(f"Inferred expected columns from pipeline (first 20 shown): {inferred[:20]}")

        # show aligned columns if they differ from user-provided columns
        if set(aligned_user_df.columns) != set(user_df.columns):
            added = [c for c in aligned_user_df.columns if c not in user_df.columns]
            dropped = [c for c in user_df.columns if c not in aligned_user_df.columns]
            if added:
                st.write("Added missing columns with defaults:", added[:20])
            if dropped:
                st.write("Dropped extra columns not expected by model:", dropped)

        # Attempt prediction
        try:
            # Classifier prediction
            pred_clf = clf.predict(aligned_user_df)[0]
            clf_proba = None
            try:
                # Some classifiers support predict_proba
                clf_proba = clf.predict_proba(aligned_user_df)[0]
                # if binary, take probability for positive class if shape matches
                if clf_proba.ndim == 1:
                    pass
                elif clf_proba.shape[0] >= 1:
                    # If shape is (n_samples, n_classes)
                    clf_proba = clf_proba[1] if len(clf_proba) > 1 else clf_proba[0]
                else:
                    clf_proba = None
            except Exception:
                clf_proba = None

            # Regressor prediction
            pred_reg = reg.predict(aligned_user_df)[0]

        except Exception as e:
            st.error("Prediction failed. Likely mismatch between app inputs and model preprocessing.")
            st.exception(e)
            # Provide quick tips
            st.write("Tips:")
            st.write("- When training the model, save the exact list of input column names used by the preprocessor")
            st.write(f"  to `{EXPECTED_COLS_PATH}` so the app can load and reindex inputs correctly.")
            st.write("- Alternatively, include all feature-engineering steps inside the saved pipeline so the runtime")
            st.write("  pipeline accepts the raw fields collected by this app (City, Price_in_Lakhs, Size_in_SqFt, etc.).")
            return

        # Display results
        st.subheader("Results")
        st.markdown(f"**Good Investment?** : {'Yes' if int(pred_clf) == 1 else 'No'}")
        if clf_proba is not None:
            # If a vector, try to print probability for positive class
            try:
                if isinstance(clf_proba, (list, tuple, np.ndarray)):
                    # If predict_proba returned vector of class probabilities, try to locate positive class
                    # We'll show the max probability for simplicity if we can't determine mapping
                    if len(clf_proba) == 2:
                        prob_positive = float(clf_proba[1])
                        st.markdown(f"**Confidence (probability)**: {prob_positive:.2f}")
                    else:
                        st.markdown(f"**Confidence (probabilities)**: {np.array(clf_proba).round(3).tolist()}")
                else:
                    st.markdown(f"**Confidence (probability)**: {float(clf_proba):.2f}")
            except Exception:
                st.write("Confidence (probability): (unavailable)")
        st.markdown(f"**Estimated Price after 5 years (Lakhs)**: {float(pred_reg):.2f}")

        # Rationale / simple rules
        st.subheader("Rationale (simple rules & features)")
        try:
            price_per_sqft = (user_df["Price_in_Lakhs"].iloc[0] * 100000) / user_df["Size_in_SqFt"].iloc[0]
            st.write("Price per SqFt (approx):", round(price_per_sqft, 2))
        except Exception:
            st.write("Price per SqFt (approx): could not compute (missing Size_in_SqFt or Price_in_Lakhs)")

        # Feature importances (if requested)
        if show_feature_importance:
            st.subheader("Feature importances (classifier)")
            show_feature_importances_if_available(clf)

    # Visual insights placeholder: small example using uploaded dataset
    st.sidebar.header("Visual insights")
    if st.sidebar.button("Show dataset insights"):
        dataset_path = "data/india_housing_prices.csv"
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            if "Price_in_Lakhs" in df.columns and "City" in df.columns:
                top_cities = df.groupby("City")["Price_in_Lakhs"].median().sort_values(ascending=False).head(10).reset_index()
                fig = px.bar(top_cities, x="City", y="Price_in_Lakhs", title="Top 10 Cities by Median Price (Lakhs)")
                st.plotly_chart(fig)
            else:
                st.write("Dataset exists but required columns missing ('Price_in_Lakhs' and/or 'City').")
        else:
            st.write(f"Dataset file not found at {dataset_path}")

if __name__ == "__main__":
    main()
