# RetailIQ: AI Retail Intelligence & Root-Cause Analysis

RetailIQ is an end-to-end retail analytics platform. It forecasts daily revenue, flags days that deviate from the forecast, and traces each anomaly to the category, brand, city, channel and inventory signals that moved with it. A local LLM lets you ask questions about the data in plain English.

## Features

- **Revenue forecasting:** an XGBoost model trained on lag features, rolling means, calendar features and a trend term.
- **Anomaly detection:** days are flagged when the forecast residual has a large z-score (for example, |z| > 2).
- **Root-cause analysis:** each anomaly is compared against a recent baseline across category, brand, city, channel and inventory levels.
- **Natural-language Q&A:** the `/ask-retailiq` endpoint answers questions using a local Llama 3.2 model through Ollama, so no cloud API key is needed.
- **REST API and dashboard:** a FastAPI backend serves the analytics to a web frontend.

## Example finding

On **27 Oct 2024**, actual revenue was about ₹90.7K against a forecast of about ₹116.8K (residual z-score ≈ -2.35). The analysis showed the largest drops in:

| Dimension | Signal | Change |
|-----------|--------|--------|
| Category | Fruits | -31.3% |
| Brand | Parle | -36.4% |
| City | Pune | -35.7% |
| Channel | Offline | -31.9% |

> These are associated signals, not proven causes. See `docs/retail_analytics.md`.

## Tech stack

Python, FastAPI, Pandas, XGBoost, scikit-learn, Ollama (Llama 3.2 3B), HTML/CSS/JavaScript

## Project structure

```
RetailQ/
├── backend/        FastAPI app (main.py)
├── frontend/       Dashboard (index.html)
├── data/           Retail transactions dataset (data.csv, 100K rows)
├── notebooks/      Exploratory data analysis
├── artifacts/      Trained model, features, detected anomalies, root-cause report
├── outputs/        Root-cause and forecast result CSVs
└── docs/           Domain notes on analytics, anomalies and inventory
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/analytics` | Overall revenue, units and margin summary |
| GET | `/analytics/categories` | Performance by category |
| GET | `/analytics/channels` | Performance by channel |
| GET | `/analytics/monthly` | Monthly trends |
| GET | `/anomalies` | Detected revenue anomalies |
| GET | `/root-cause` | Root-cause report for an anomaly |
| POST | `/ask-retailiq` | Ask a question in natural language |

## Getting started

1. **Clone the repo**
   ```bash
   git clone https://github.com/Varshitha-12-source/RetailQ.git
   cd RetailQ
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS / Linux
   pip install -r requirements.txt
   ```

3. **Install Ollama and pull the model** (needed for the Q&A feature)
   ```bash
   ollama pull llama3.2:3b
   ```

4. **Run the backend**
   ```bash
   uvicorn backend.main:app --reload
   ```
   Interactive API docs are at http://127.0.0.1:8000/docs

5. **Open the dashboard** by opening `frontend/index.html` in your browser.

## Dataset

`data/data.csv` contains 100,000 retail transactions with invoice date, city, store format, category, brand, channel, payment mode, units, cost and selling price, margin, stock levels, lead time and customer attributes.

## Author

**Varshitha V**: [GitHub](https://github.com/Varshitha-12-source)
