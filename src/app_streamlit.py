# src/app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

MODEL_DIR = "models"
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier_pipeline.joblib")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "regressor_pipeline.joblib")

@st.cache_resource
def load_models():
    clf = joblib.load(CLASSIFIER_PATH)
    reg = joblib.load(REGRESSOR_PATH)
    return clf, reg

def user_input_form():
    st.sidebar.header("Property details")
    # Use a small representative set of inputs; extend as needed
    city = st.sidebar.text_input("City", "Bengaluru")
    property_type = st.sidebar.selectbox("Property Type", ["Apartment", "Villa", "House", "Independent"])
    bhk = st.sidebar.number_input("BHK", min_value=1, max_value=10, value=2)
    size = st.sidebar.number_input("Size in SqFt", min_value=200, value=900)
    price_lakhs = st.sidebar.number_input("Current Price (in Lakhs)", min_value=1.0, value=60.0)
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
        "RERA": int(rera)
    }
    return pd.DataFrame([input_dict])

def main():
    st.title("Real Estate Investment Advisor")
    st.markdown("Predict if property is a **Good Investment** and estimate **Price after 5 years**.")
    clf, reg = load_models()

    st.sidebar.header("Model controls")
    show_feature_importance = st.sidebar.checkbox("Show feature importances (classifier)", value=True)

    user_df = user_input_form()
    st.subheader("Input summary")
    st.write(user_df.T)

    if st.button("Predict"):
        # Preprocess & predict
        pred_clf = clf.predict(user_df)[0]
        try:
            clf_proba = clf.predict_proba(user_df)[0][1]
        except:
            clf_proba = None
        pred_reg = reg.predict(user_df)[0]

        st.subheader("Results")
        st.markdown(f"**Good Investment?** : {'Yes' if pred_clf==1 else 'No'}")
        if clf_proba is not None:
            st.markdown(f"**Confidence (probability)**: {clf_proba:.2f}")
        st.markdown(f"**Estimated Price after 5 years (Lakhs)**: {pred_reg:.2f}")

        # Show simple breakdown
        st.subheader("Rationale (simple rules & features)")
        st.write("Price per SqFt (approx):", (user_df["Price_in_Lakhs"].iloc[0]*100000)/user_df["Size_in_SqFt"].iloc[0])

        if show_feature_importance:
            st.subheader("Feature importances (classifier)")
            try:
                # classifier is a pipeline: preprocessor -> clf
                clf_step = clf.named_steps.get("clf", None)
                pre = clf.named_steps.get("preprocessor", None)
                if hasattr(clf_step, "feature_importances_") and pre is not None:
                    import numpy as np
                    # get feature names
                    num_features = pre.transformers_[0][2]
                    cat_transformer = pre.transformers_[1][1].named_steps["onehot"]
                    cat_cols = pre.transformers_[1][2]
                    cat_names = cat_transformer.get_feature_names_out(cat_cols)
                    feature_names = list(num_features) + list(cat_names)
                    imps = clf_step.feature_importances_
                    fi = pd.DataFrame({"feature": feature_names, "importance": imps})
                    fi = fi.sort_values("importance", ascending=False).head(15)
                    st.table(fi.set_index("feature"))
                else:
                    st.write("Feature importances not available for this model.")
            except Exception as e:
                st.write("Could not compute feature importances:", str(e))

    # Visual insights placeholder: small example using uploaded dataset
    st.sidebar.header("Visual insights")
    if st.sidebar.button("Show dataset insights"):
        if os.path.exists("data/india_housing_prices.csv"):
            df = pd.read_csv("data/india_housing_prices.csv")
            if "Price_in_Lakhs" in df.columns and "City" in df.columns:
                top_cities = df.groupby("City")["Price_in_Lakhs"].median().sort_values(ascending=False).head(10).reset_index()
                fig = px.bar(top_cities, x="City", y="Price_in_Lakhs", title="Top 10 Cities by Median Price (Lakhs)")
                st.plotly_chart(fig)
            else:
                st.write("Dataset exists but required columns missing.")
        else:
            st.write("Dataset file not found at data/india_housing_prices.csv")

if __name__ == "__main__":
    main()
