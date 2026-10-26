
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "workationiq_model.cbm"
SCHEMA_PATH = BASE_DIR / "artifacts" / "feature_schema.json"

st.set_page_config(
    page_title="WorkationIQ Price Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ WorkationIQ")
st.subheader("Travel Package Price Prediction")
st.write(
    "Upload a CSV containing the model's engineered feature columns "
    "to generate travel-package price predictions."
)

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    return model

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)

model = load_model()
required_columns = schema["feature_columns"]
categorical_columns = schema.get("categorical_features", [])

st.caption(
    f"Model: {schema['model_type']} | "
    f"Required features: {schema['feature_count']}"
)

uploaded_file = st.file_uploader(
    "Upload a feature-ready CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        missing = [
            col for col in required_columns
            if col not in data.columns
        ]

        if missing:
            st.error(
                "Your CSV is missing required feature columns: "
                + ", ".join(missing)
            )
        elif data.empty:
            st.warning("The uploaded CSV contains no rows.")
        else:
            X = data[required_columns].copy()

            for col in categorical_columns:
                X[col] = X[col].fillna("Unknown").astype(str)

            numeric_columns = [
                col for col in required_columns
                if col not in categorical_columns
            ]

            for col in numeric_columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

            if X.isna().any().any():
                st.error(
                    "Missing or invalid values remain after conversion. "
                    "Clean the input using the same preprocessing used "
                    "during model training."
                )
            else:
                predictions = model.predict(X)

                results = data.copy()
                results["Predicted_Price"] = predictions

                st.success(
                    f"Generated {len(results)} predictions successfully."
                )

                st.metric("Packages predicted", len(results))
                st.dataframe(results.head(100), use_container_width=True)

                st.download_button(
                    "Download predictions CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="workationiq_predictions.csv",
                    mime="text/csv"
                )

    except Exception as exc:
        st.error(f"Unable to process this file: {exc}")

st.divider()
st.caption(
    "Predictions are estimates, not guaranteed travel-package prices. "
    "Input data must follow the model's training feature definitions."
)
