# Next Purchase Prediction System — Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ECOMMERCE PLATFORM                                   │
│  Website / App  →  Orders  →  Sessions  →  Campaigns                       │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ writes events
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PostgreSQL  (ecommerce_ml DB)                           │
│                                                                             │
│  DIMENSION          FACT TABLES           ML OUTPUT                        │
│  dim_customers  ←── fact_orders       ←── ml_predictions                   │
│  dim_products   ←── fact_order_items  ←── ml_customer_segments             │
│  dim_campaigns  ←── fact_sessions                                           │
│                 ←── fact_campaign_interactions                              │
│                                                                             │
│  VIEWS (feature store):                                                     │
│  v_customer_purchase_history  ← used for training & batch inference        │
│  v_customer_rfm               ← used for RFM features                      │
│  v_product_stats              ← used for Stage 2 candidate features        │
│  v_latest_predictions         ← used by campaign system                    │
└────────────┬────────────────────────────────────┬───────────────────────────┘
             │ reads via SQLAlchemy               │ reads latest predictions
             ▼                                    ▼
┌────────────────────────────┐     ┌──────────────────────────────────────────┐
│    ML TRAINING PIPELINE    │     │         CAMPAIGN SYSTEM                  │
│  (ipynb / batch_train.py)  │     │   (your existing campaign project)       │
│                            │     │                                          │
│  Stage 1: Category Model   │     │  1. Read ml_customer_segments            │
│   LGB / XGB / RF + Optuna  │     │  2. Group by predicted_product           │
│                            │     │  3. Build campaign audience              │
│  Stage 2: LambdaRank       │     │  4. Send via MSG91 / Email / SMS         │
│   Candidate Gen + Scoring  │     │  5. Mark dispatched                      │
│                            │     │  6. Track opens / clicks                 │
│  → saves to models/        │     └──────────────────────────────────────────┘
└────────────────────────────┘
             │ trained models
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PREDICTION PIPELINE                                     │
│                                                                             │
│  BATCH (nightly cron / Airflow):                                            │
│    batch_predict.py → reads v_customer_purchase_history                     │
│                     → runs Stage1 + Stage2 per customer                     │
│                     → writes ml_predictions                                 │
│                     → writes ml_customer_segments                           │
│                                                                             │
│  REAL-TIME (Flask API at port 5000):                                        │
│    POST /predict        → single customer → top-N products                  │
│    POST /batch_predict  → list of customers → predictions + segments        │
│    GET  /health         → healthcheck                                       │
│    GET  /segments       → pending segments for campaign dispatch             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Customer makes purchase
        │
        ▼
fact_orders + fact_order_items + fact_sessions  (written by e-commerce backend)
        │
        ▼  (nightly at 2 AM)
batch_predict.py
        │
        ├── reads v_customer_purchase_history  (all customers)
        ├── rebuilds point-in-time features
        ├── Stage 1: predicts next_category per customer
        ├── Stage 2: ranks top-5 products within predicted category
        ├── Cold-start handling: popularity-based for 1-purchase customers
        ├── writes → ml_predictions       (one row per customer)
        └── writes → ml_customer_segments (customers grouped by next product)
                         │
                         ▼
                Campaign System reads ml_customer_segments
                → "200 customers will buy Samsung Electronics next"
                → sends targeted WhatsApp / Email campaign
```

## Cold-Start Strategy

| Customer State | n_purchases | Strategy |
|---|---|---|
| New (cold-start) | 1 | **Popularity-based**: Top products in their demographic (age, city, gender) |
| Warm | 2–4 | Stage 1 model (limited history signal) |
| Active | 5+ | Full Stage 1 + Stage 2 pipeline |

## Retraining Schedule

| Trigger | Action |
|---|---|
| Weekly (Sunday 3 AM) | Full retrain on last 12 months of data |
| Monthly | Optuna hyperparameter re-tuning |
| Drift detected (accuracy drop > 5%) | Alert + emergency retrain |

## Files in this Project

```
project/
├── Model1_Next_Purchase_Prediction_FINAL.ipynb   ← main notebook (train here)
├── schema.sql                                     ← PostgreSQL DDL
├── db_connector.py                                ← DB read/write layer
├── app.py                                         ← Flask API (generated by nb)
├── batch_predict.py                               ← nightly cron script
├── architecture.md                                ← this file
└── models/
    ├── stage1_category_model.lgb                  ← LightGBM booster
    ├── stage2_lambdarank_model.lgb                ← LambdaRank booster
    └── artifacts.pkl                              ← encoders + metadata
```

## Environment Variables

```bash
DB_URL=postgresql://postgres:password@localhost:5432/ecommerce_ml
MODEL_VERSION=v1.0
FLASK_PORT=5000
```
