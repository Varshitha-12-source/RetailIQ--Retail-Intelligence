# ============================================================
# RETAILIQ — FASTAPI BACKEND
# ============================================================


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path
import json
import ollama




# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="RetailIQ",
    description="AI Retail Data Intelligence and Root-Cause Analysis Platform",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_PATH / "data" / "data.csv"

ARTIFACTS_PATH = PROJECT_PATH / "artifacts"

ANOMALY_PATH = (
    ARTIFACTS_PATH
    / "detected_anomalies.csv"
)

ROOT_CAUSE_PATH = (
    ARTIFACTS_PATH
    / "root_cause_report.json"
)


# ============================================================
# LOAD RETAIL DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

# Convert date column
df["Invoice_Date"] = pd.to_datetime(
    df["Invoice_Date"]
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "application": "RetailIQ",
        "status": "running",
        "message": "AI Retail Intelligence API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# RETAIL ANALYTICS
# ============================================================

@app.get("/analytics")
def analytics():

    total_revenue = float(
        df["Revenue"].sum()
    )

    total_units = int(
        df["Units"].sum()
    )

    total_margin = float(
        df["Margin"].sum()
    )

    average_selling_price = float(
        df["Selling_Price"].mean()
    )

    average_stock = float(
        df["Stock_On_Hand"].mean()
    )

    return {

        "total_revenue": round(
            total_revenue,
            2
        ),

        "total_units": total_units,

        "total_margin": round(
            total_margin,
            2
        ),

        "average_selling_price": round(
            average_selling_price,
            2
        ),

        "average_stock": round(
            average_stock,
            2
        )
    }


# ============================================================
# CATEGORY ANALYTICS
# ============================================================

@app.get("/analytics/categories")
def category_analytics():

    category_data = (
        df.groupby("Category")
        .agg(
            revenue=("Revenue", "sum"),
            units=("Units", "sum"),
            margin=("Margin", "sum"),
            average_price=("Selling_Price", "mean")
        )
        .reset_index()
    )

    category_data = category_data.round(2)

    return category_data.to_dict(
        orient="records"
    )


# ============================================================
# CHANNEL ANALYTICS
# ============================================================

@app.get("/analytics/channels")
def channel_analytics():

    channel_data = (
        df.groupby("Channel")
        .agg(
            revenue=("Revenue", "sum"),
            units=("Units", "sum"),
            margin=("Margin", "sum")
        )
        .reset_index()
    )

    channel_data = channel_data.round(2)

    return channel_data.to_dict(
        orient="records"
    )


# ============================================================
# ANOMALY DETECTION RESULTS
# ============================================================

@app.get("/anomalies")
def get_anomalies():

    anomaly_df = pd.read_csv(
        ANOMALY_PATH
    )

    # Keep only actual anomalies
    anomaly_df = anomaly_df[
        anomaly_df["Is_Anomaly"] == True
    ]

    # Select useful fields
    anomaly_df = anomaly_df[
        [
            "Sale_Date",
            "Actual_Revenue",
            "Predicted_Revenue",
            "Residual",
            "Residual_Z",
            "Is_Anomaly"
        ]
    ].copy()

    # Round numerical values
    anomaly_df["Actual_Revenue"] = (
        anomaly_df["Actual_Revenue"]
        .round(2)
    )

    anomaly_df["Predicted_Revenue"] = (
        anomaly_df["Predicted_Revenue"]
        .round(2)
    )

    anomaly_df["Residual"] = (
        anomaly_df["Residual"]
        .round(2)
    )

    anomaly_df["Residual_Z"] = (
        anomaly_df["Residual_Z"]
        .round(2)
    )

    return anomaly_df.to_dict(
        orient="records"
    )


# ============================================================
# ROOT-CAUSE ANALYSIS
# ============================================================

@app.get("/root-cause")
def get_root_cause():

    with open(
        ROOT_CAUSE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        root_cause = json.load(file)

    return root_cause


# ============================================================
# AI RETAIL ANALYST
# ============================================================

@app.post("/ask-retailiq")
def ask_retailiq():

    # --------------------------------------------------------
    # LOAD ROOT-CAUSE EVIDENCE
    # --------------------------------------------------------

    with open(
        ROOT_CAUSE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        evidence = json.load(file)


    # --------------------------------------------------------
    # CONVERT EVIDENCE TO JSON
    # --------------------------------------------------------

    evidence_json = json.dumps(
        evidence,
        indent=2
    )


    # --------------------------------------------------------
    # AI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are RetailIQ, an AI Retail Analyst.

Your task is to explain the detected retail
revenue anomaly using ONLY the retail evidence
provided below.

Rules:

- Do not invent facts.
- Do not invent numbers.
- Do not calculate new statistics.
- Use ₹ for currency.
- Preserve the direction of every change.
- Treat category, brand, city and channel
  observations as associated signals.
- Do not claim causation.
- Clearly state when the evidence does not
  support a possible explanation.
- A reduction in low-stock records does not
  indicate an inventory shortage.
- Keep the explanation concise and
  business-friendly.

Return exactly these sections:

1. Anomaly Summary

2. Key Associated Signals

3. Inventory Assessment

4. Pricing and Margin Assessment

5. Overall Interpretation

6. Limitation


RETAIL EVIDENCE:

{evidence_json}
"""


    # --------------------------------------------------------
    # CALL LOCAL LLAMA MODEL
    # --------------------------------------------------------

    response = ollama.chat(

        model="llama3.2:3b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        options={
            "temperature": 0
        }
    )


    # --------------------------------------------------------
    # RETURN AI RESPONSE
    # --------------------------------------------------------

    return {
        "analysis": response[
            "message"
        ][
            "content"
        ]
    }
@app.get("/analytics/monthly")
def monthly_analytics():

    monthly_data = (
        df.groupby(df["Invoice_Date"].dt.strftime("%b"))
        .agg(
            revenue=("Revenue", "sum"),
            units=("Units", "sum"),
            margin=("Margin", "sum")
        )
        .reindex([
            "Jan", "Feb", "Mar", "Apr",
            "May", "Jun", "Jul", "Aug",
            "Sep", "Oct", "Nov", "Dec"
        ])
        .reset_index()
    )

    monthly_data = monthly_data.round(2)

    return monthly_data.to_dict(orient="records")
@app.get("/analytics/monthly")
def monthly_analytics():

    monthly_data = (
        df.groupby(df["Invoice_Date"].dt.strftime("%b"))
        .agg(
            revenue=("Revenue", "sum"),
            units=("Units", "sum"),
            margin=("Margin", "sum")
        )
        .reindex([
            "Jan", "Feb", "Mar", "Apr",
            "May", "Jun", "Jul", "Aug",
            "Sep", "Oct", "Nov", "Dec"
        ])
        .reset_index()
    )

    monthly_data = monthly_data.round(2)

    return monthly_data.to_dict(orient="records")