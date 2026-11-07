from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request


# --------------------------------------------------
# Create Flask application
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# Locate project files
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


# --------------------------------------------------
# Load trained machine-learning pipelines
# --------------------------------------------------

regression_model = joblib.load(
    MODELS_DIR / "regression_random_forest.joblib"
)

classification_model = joblib.load(
    MODELS_DIR / "classification_logistic.joblib"
)


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

with open(
    MODELS_DIR / "metadata.json",
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)


category_options = metadata["category_options"]


# --------------------------------------------------
# Helper: create model input
# --------------------------------------------------

def create_input_row(
    business_age,
    months,
    paid_workers,
    unpaid_workers,
    raw_material_cost,
    operating_cost,
    sector,
    acquisition,
    funding,
    location,
    expansion,
    vat,
    income_tax,
    current_revenue=None,
):
    """
    Convert values submitted by the HTML form into
    the DataFrame structure expected by our saved
    machine-learning pipelines.
    """

    row = {
        "business_age": float(business_age),

        "months_operated_last_12_months":
            float(months),

        "paid_workers":
            float(paid_workers),

        "unpaid_workers":
            float(unpaid_workers),

        "log_monthly_raw_material_cost":
            float(np.log1p(raw_material_cost)),

        "log_monthly_other_operating_cost":
            float(np.log1p(operating_cost)),

        "business_sector":
            sector,

        "business_acquisition_method":
            acquisition,

        "startup_funding_source":
            funding,

        "business_location":
            location,

        "used_funding_for_expansion":
            expansion,

        "vat_registration_status":
            vat,

        "income_tax_registration_status":
            income_tax,
    }

    # Classification model additionally uses
    # current monthly gross revenue.
    if current_revenue is not None:
        row["log_monthly_gross_revenue"] = float(
            np.log1p(current_revenue)
        )

    return pd.DataFrame([row])


# --------------------------------------------------
# Helper: common template context
# --------------------------------------------------

def page_context(
    active_tab="regression",
    form=None,
    **extra
):
    context = {
        "options": category_options,
        "metadata": metadata,
        "active_tab": active_tab,
        "form": form or {},
        "reg_result": None,
        "reg_error": None,
        "class_result": None,
        "class_probabilities": None,
        "class_error": None,
    }

    context.update(extra)

    return context


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        **page_context()
    )


# --------------------------------------------------
# Regression prediction route
# --------------------------------------------------

@app.route(
    "/predict-regression",
    methods=["POST"]
)
def predict_regression():

    try:
        business_age = float(
            request.form["business_age"]
        )

        months = float(
            request.form["months"]
        )

        paid_workers = float(
            request.form["paid_workers"]
        )

        unpaid_workers = float(
            request.form["unpaid_workers"]
        )

        raw_material_cost = float(
            request.form["raw_material_cost"]
        )

        operating_cost = float(
            request.form["operating_cost"]
        )

        sector = request.form["sector"]
        acquisition = request.form["acquisition"]
        funding = request.form["funding"]
        location = request.form["location"]
        expansion = request.form["expansion"]
        vat = request.form["vat"]
        income_tax = request.form["income_tax"]

        input_df = create_input_row(
            business_age=business_age,
            months=months,
            paid_workers=paid_workers,
            unpaid_workers=unpaid_workers,
            raw_material_cost=raw_material_cost,
            operating_cost=operating_cost,
            sector=sector,
            acquisition=acquisition,
            funding=funding,
            location=location,
            expansion=expansion,
            vat=vat,
            income_tax=income_tax,
        )

        # Model predicts log-transformed revenue
        predicted_log_revenue = (
            regression_model.predict(input_df)[0]
        )

        # Convert prediction back to Uganda Shillings
        predicted_revenue = max(
            float(
                np.expm1(
                    predicted_log_revenue
                )
            ),
            0.0
        )

        formatted_prediction = (
            f"UGX {predicted_revenue:,.0f}"
        )

        return render_template(
            "index.html",
            **page_context(
                active_tab="regression",
                form=request.form,
                reg_result=formatted_prediction,
            )
        )

    except Exception:
        app.logger.exception(
            "Regression prediction failed"
        )

        return render_template(
            "index.html",
            **page_context(
                active_tab="regression",
                form=request.form,
                reg_error=(
                    "Prediction could not be completed. "
                    "Please check the entered values."
                ),
            )
        )


# --------------------------------------------------
# Classification prediction route
# --------------------------------------------------

@app.route(
    "/predict-classification",
    methods=["POST"]
)
def predict_classification():

    try:
        business_age = float(
            request.form["business_age"]
        )

        months = float(
            request.form["months"]
        )

        paid_workers = float(
            request.form["paid_workers"]
        )

        unpaid_workers = float(
            request.form["unpaid_workers"]
        )

        raw_material_cost = float(
            request.form["raw_material_cost"]
        )

        operating_cost = float(
            request.form["operating_cost"]
        )

        current_revenue = float(
            request.form["current_revenue"]
        )

        sector = request.form["sector"]
        acquisition = request.form["acquisition"]
        funding = request.form["funding"]
        location = request.form["location"]
        expansion = request.form["expansion"]
        vat = request.form["vat"]
        income_tax = request.form["income_tax"]

        input_df = create_input_row(
            business_age=business_age,
            months=months,
            paid_workers=paid_workers,
            unpaid_workers=unpaid_workers,
            raw_material_cost=raw_material_cost,
            operating_cost=operating_cost,
            sector=sector,
            acquisition=acquisition,
            funding=funding,
            location=location,
            expansion=expansion,
            vat=vat,
            income_tax=income_tax,
            current_revenue=current_revenue,
        )

        predicted_class = (
            classification_model.predict(
                input_df
            )[0]
        )

        probabilities = (
            classification_model.predict_proba(
                input_df
            )[0]
        )

        classes = (
            classification_model
            .named_steps["model"]
            .classes_
        )

        probability_results = []

        for class_name, probability in zip(
            classes,
            probabilities
        ):
            probability_results.append(
                {
                    "class_name": str(class_name),
                    "probability":
                        float(probability * 100),
                }
            )

        # Highest probability first
        probability_results.sort(
            key=lambda item:
                item["probability"],
            reverse=True
        )

        return render_template(
            "index.html",
            **page_context(
                active_tab="classification",
                form=request.form,
                class_result=str(
                    predicted_class
                ),
                class_probabilities=(
                    probability_results
                ),
            )
        )

    except Exception:
        app.logger.exception(
            "Classification prediction failed"
        )

        return render_template(
            "index.html",
            **page_context(
                active_tab="classification",
                form=request.form,
                class_error=(
                    "Classification could not be "
                    "completed. Please check the "
                    "entered values."
                ),
            )
        )


# --------------------------------------------------
# Run application locally
# --------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )