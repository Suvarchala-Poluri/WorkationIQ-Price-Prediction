# WorkationIQ — Travel Package Price Prediction

## Overview

WorkationIQ is a machine-learning project that predicts travel-package
prices from structured package and travel-related features.

The project uses CatBoost Regression and includes model evaluation,
prediction outputs, feature-importance analysis, and a Streamlit
batch-prediction interface.

## Problem Statement

Travel packages vary in price depending on their characteristics.
This project explores whether machine learning can estimate package
prices from available historical data.

## Objectives

- Explore and prepare travel-package data.
- Engineer features for machine learning.
- Train a regression model.
- Evaluate predictions on held-out validation data.
- Generate predictions for the supplied test dataset.
- Provide a reusable prediction interface.

## Technology Stack

- Python
- Pandas and NumPy
- CatBoost
- Scikit-learn
- Matplotlib
- Google Colab
- Streamlit
- Git and GitHub

## Validation Results

See `reports/evaluation_metrics.json` for the saved evaluation metrics.

The current validation results are:

| Metric | Result |
|---|---:|
| MAE | 3343.8998 |
| RMSE | 5918.9744 |
| R² | 0.7622 |
| MAPE | 16.50% |

These values come from the current validation run. MAE and RMSE use
the target's original units. Confirm the target currency and price
distribution before interpreting their business significance.
MAPE excludes zero-valued actual prices and can be sensitive to small
actual values.

## Project Structure

```text
WorkationIQ_Project/
├── artifacts/
│   ├── workationiq_model.cbm
│   └── feature_schema.json
├── reports/
│   ├── evaluation_metrics.json
│   ├── validation_predictions.csv
│   ├── test_predictions.csv
│   ├── actual_vs_predicted.png
│   ├── prediction_errors.png
│   ├── feature_importance.csv
│   └── feature_importance.png
├── app.py
├── requirements.txt
└── README.md
```

The original notebook and datasets can be included separately when
they are appropriate to share.

## Run Locally

1. Install Python 3.10 or a compatible version.
2. Create a virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Start the application:

   `streamlit run app.py`

5. Upload a CSV with the exact engineered feature columns expected by
   the saved model.

## Input Requirements

The application expects the same engineered feature columns and
compatible data types used during training. Raw travel-package data
must first pass through the original feature-engineering process.

## Model Evaluation

- MAE measures mean absolute prediction error.
- RMSE penalizes larger errors more heavily.
- R² measures explained variation relative to a mean baseline.
- MAPE expresses absolute error as a percentage for nonzero targets.

## Limitations

Predictions depend on the quality and coverage of the training data.
Performance may differ for destinations, dates, or package types that
are underrepresented in the dataset. Predictions are estimates, not
quotes or guaranteed prices.

## Future Improvements

- Improve preprocessing and feature validation.
- Compare against a simple baseline and other regression models.
- Analyze large prediction errors and data leakage risks.
- Add a raw travel-data preprocessing interface.
- Deploy the application online.

## Author

Developed as a machine-learning portfolio project.
