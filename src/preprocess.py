# src/preprocess.py
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import sklearn

CURRENT_YEAR = 2025

def basic_cleaning(df):
    # Drop exact duplicates
    df = df.drop_duplicates(subset=["ID"], keep="first") if "ID" in df.columns else df.drop_duplicates()
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def fill_missing_domain(df):
    # Simple heuristics for missing numeric fields
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())
    # Fill categorical with 'Unknown'
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")
    return df

def feature_engineering(df):
    # Price per sqft if missing or inconsistent
    if "Price_per_SqFt" not in df.columns or df["Price_per_SqFt"].isnull().any():
        if "Price_in_Lakhs" in df.columns and "Size_in_SqFt" in df.columns:
            # Price_in_Lakhs * 1e5 -> rupees, but keep in same units: lakhs/sqft
            df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 100000) / df["Size_in_SqFt"]
        else:
            df["Price_per_SqFt"] = df.get("Price_per_SqFt", np.nan).fillna(df["Price_per_SqFt"].median())

    # Age_of_Property if missing
    if "Year_Built" in df.columns and "Age_of_Property" not in df.columns:
        df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"].fillna(CURRENT_YEAR)

    # avoid chained-assignment / inplace warning by assigning result back
    df["Price_per_SqFt"] = df["Price_per_SqFt"].replace([np.inf, -np.inf], np.nan)
    df["Price_per_SqFt"] = df["Price_per_SqFt"].fillna(df["Price_per_SqFt"].median())

    # Price per BHK: avoid division by zero
    df["BHK"] = df["BHK"].fillna(1).replace(0, 1)
    df["Price_per_BHK"] = df["Price_in_Lakhs"] / df["BHK"]

    # Amenities count (if amenities string exists)
    if "Amenities" in df.columns:
        df["Amenities_Count"] = df["Amenities"].fillna("").apply(lambda s: len([x for x in str(s).split(",") if x.strip()]))
    else:
        df["Amenities_Count"] = 0

    # School density score (normalize Nearby_Schools)
    if "Nearby_Schools" in df.columns:
        df["School_Density"] = (df["Nearby_Schools"] - df["Nearby_Schools"].min()) / (df["Nearby_Schools"].max() - df["Nearby_Schools"].min() + 1e-6)
    else:
        df["School_Density"] = 0.0

    # Public transport numeric
    if "Public_Transport_Accessibility" in df.columns:
        # if it's textual, map common words to numeric
        def map_transport(x):
            if pd.isna(x): return 0
            s = str(x).lower()
            if any(k in s for k in ["excellent", "high", "very good"]): return 3
            if any(k in s for k in ["good", "medium"]): return 2
            if any(k in s for k in ["poor", "low"]): return 1
            # if number
            try:
                return float(x)
            except:
                return 1
        df["Transport_Score"] = df["Public_Transport_Accessibility"].apply(map_transport)
    else:
        df["Transport_Score"] = 1

    return df

def create_targets(df, growth_rate_lookup=None, fixed_rate=0.08, years=5):
    # Regression target: simple growth projection and/or feature-based later
    if "Price_in_Lakhs" in df.columns:
        # create future price using per-city growth rates if provided
        df["City"] = df.get("City", "Unknown")
        if growth_rate_lookup is None:
            growth_rate_lookup = {}
        def city_growth(row):
            r = growth_rate_lookup.get(row["City"], fixed_rate)
            return row["Price_in_Lakhs"] * ((1 + r) ** years)
        df["Future_Price_5Y"] = df.apply(city_growth, axis=1)
    else:
        raise ValueError("Price_in_Lakhs must exist to create Future_Price_5Y")

    # Classification target: Good_Investment (binary)
    # Rule-based score:
    median_price = df["Price_in_Lakhs"].median()
    df["Is_Cheap"] = (df["Price_in_Lakhs"] <= median_price).astype(int)
    df["Good_BHK"] = (df["BHK"] >= 3).astype(int)
    df["Ready_to_Move"] = df.get("Availability_Status", "").apply(lambda x: 1 if str(x).lower() in ["ready to move", "available"] else 0)
    df["RERA_flag"] = df.get("RERA", 0) if "RERA" in df.columns else 0

    # Combined score threshold
    df["Investment_Score"] = df["Is_Cheap"] + df["Good_BHK"] + df["Ready_to_Move"] + (df["RERA_flag"]>0).astype(int) + (df["School_Density"]>0.5).astype(int)
    df["Good_Investment"] = (df["Investment_Score"] >= 3).astype(int)
    return df

def build_preprocessor(df):
    # Define which columns to treat as numeric/categorical
    numeric_cols = [
        c for c in df.select_dtypes(include=["int64", "float64"]).columns
        if c not in ("Good_Investment", "Future_Price_5Y")
    ]
    # choose a small set of categorical columns commonly present
    cat_cols = [c for c in ["State","City","Locality","Property_Type","Furnished_Status","Security","Facing","Owner_Type","Availability_Status"] if c in df.columns]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Build OneHotEncoder kwargs depending on sklearn version
    skl_ver = getattr(sklearn, "__version__", "0.0")
    try:
        major, minor = [int(x) for x in skl_ver.split(".")[:2]]
    except Exception:
        major, minor = 0, 0

    ohe_kwargs = {}
    if (major, minor) >= (1, 2):
        # sklearn >=1.2 uses sparse_output
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", **ohe_kwargs))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols)
    ], remainder="drop")
    return preprocessor, numeric_cols, cat_cols

def save_object(obj, path):
    joblib.dump(obj, path)
