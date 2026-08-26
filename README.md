# Next Purchase Prediction & Recommendation System

An ML-powered e-commerce recommendation system that predicts a customer's next purchase category and ranks relevant products based on historical customer behavior.

## Overview

The system uses customer purchase history, product information, session activity, and behavioral features to generate personalized next-purchase recommendations.

The recommendation pipeline follows two main stages:

1. **Category Prediction** – Predict the category a customer is most likely to purchase next.
2. **Product Ranking** – Rank candidate products within the predicted category using a learning-to-rank model.

## Key Features

- Next purchase category prediction
- Personalized product recommendation
- Customer purchase-history analysis
- Behavioral feature engineering
- Point-in-time feature engineering to prevent data leakage
- Multiple ML model comparison
- Hyperparameter optimization using Optuna
- Learning-to-rank using LambdaRank
- PostgreSQL database integration
- Flask backend and REST API
- Cold-start recommendation handling

## Machine Learning

The category prediction stage evaluates multiple machine learning models:

- LightGBM
- XGBoost
- Random Forest

**Optuna** is used for hyperparameter optimization.

For product ranking, the system uses **LightGBM LambdaRank** to rank candidate products based on customer-product relevance.

### Model Performance

| Metric | Score |
|---|---:|
| Top-1 Accuracy | 64.10% |
| Top-3 Accuracy | 78.82% |
| LambdaRank NDCG@5 | 72.84% |

## Feature Engineering

The system uses **28 customer and behavioral features**, including:

- Recency
- Purchase frequency
- Monetary value
- Category affinity
- Purchase history
- Session behavior
- Cumulative customer activity
- Customer-category interactions
- Product/category interaction signals

Point-in-time feature engineering is used to ensure that future information is not included when generating features for prediction.

## Backend

The project includes a **Flask-based backend** that provides REST API endpoints for prediction and customer recommendation workflows.

The backend handles:

- Customer prediction requests
- Purchase-history based recommendations
- Model loading and inference
- Database interaction
- API request/response processing

## Database

The project uses **PostgreSQL** for storing and managing structured e-commerce data.

The database contains information related to:

- Customers
- Products
- Orders
- Order Items
- Sessions
- Purchase History
- Event Logs

**pgAdmin** can be used to manage, query, and monitor the PostgreSQL database.

## Technology Stack

### Programming
- Python
- SQL

### Machine Learning
- LightGBM
- XGBoost
- Scikit-learn
- Optuna

### Backend
- Flask
- REST API

### Database
- PostgreSQL
- pgAdmin

### Data Processing
- Pandas
- NumPy

## Recommendation Pipeline

```text
Customer Purchase History
          │
          ▼
   Feature Engineering
          │
          ▼
  Category Prediction
          │
          ▼
 Candidate Product Generation
          │
          ▼
     LambdaRank
          │
          ▼
 Ranked Recommendations