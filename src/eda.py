# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.preprocess import basic_cleaning, fill_missing_domain, feature_engineering, create_targets

def run_eda(path_in="data/india_housing_prices.csv", out_dir="artifacts/eda"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(path_in)
    df = basic_cleaning(df)
    df = fill_missing_domain(df)
    df = feature_engineering(df)
    df = create_targets(df)

    # 1. Distribution of property prices
    plt.figure(figsize=(8,5))
    sns.histplot(df["Price_in_Lakhs"], bins=50, kde=True)
    plt.title("Distribution of Property Prices (Lakhs)")
    plt.savefig(os.path.join(out_dir, "price_distribution.png"))
    plt.clf()

    # 2. Distribution of property sizes
    if "Size_in_SqFt" in df.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(df["Size_in_SqFt"], bins=50)
        plt.title("Property Sizes (SqFt)")
        plt.savefig(os.path.join(out_dir, "size_distribution.png"))
        plt.clf()

    # 3. Price per sqft by Property_Type
    if "Property_Type" in df.columns and "Price_per_SqFt" in df.columns:
        plt.figure(figsize=(10,6))
        sns.boxplot(x="Property_Type", y="Price_per_SqFt", data=df)
        plt.xticks(rotation=45)
        plt.title("Price per SqFt by Property Type")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "price_per_sqft_by_type.png"))
        plt.clf()

    # 4. Correlation heatmap for numeric features
    num = df.select_dtypes(include=["float64","int64"])
    corr = num.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Numeric Feature Correlations")
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"))
    plt.clf()

    # 5. Top 10 expensive localities
    if "Locality" in df.columns:
        top_localities = df.groupby("Locality")["Price_per_SqFt"].median().sort_values(ascending=False).head(10)
        top_localities.to_csv(os.path.join(out_dir, "top_localities_price_per_sqft.csv"))

    print(f"EDA artifacts saved to {out_dir}")
    return out_dir

if __name__ == "__main__":
    run_eda()
