

from flask import Flask, request, jsonify
import pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb
import pickle, json, os
from datetime import datetime, timezone
from db_connector import DBConnector

app = Flask(__name__)

# ── Load models + artifacts once on startup ──────────────────────────────────
with open("models/artifacts.pkl", "rb") as f:
    art = pickle.load(f)

if art["algo_type"] == "LightGBM":
    cat_model = lgb.Booster(model_file="models/stage1_category_model.lgb")
else:
    cat_model = xgb.Booster()
    cat_model.load_model("models/stage1_category_model.xgb")

rank_model = lgb.Booster(model_file="models/stage2_lambdarank_model.lgb")
pc = art["product_catalog"]


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Feature Engineering (point-in-time)
# Same logic as predict_new_user.py so training & serving schemas match.
# hist = db.load_customer_history(cid)  → this function builds the feature
# vector: "Total Purchases / Total Spend / Last Purchase / Favorite Category…"
# ══════════════════════════════════════════════════════════════════════════
def build_features_from_history(history_df):
    """Point-in-time features from the LATEST state of the customer.
    Returns (feat_dict, is_cold_start)."""
    if len(history_df) == 0:
        return None, True

    h = history_df.copy()
    h["purchase_date"] = pd.to_datetime(h["purchase_date"])
    h["signup_date"]   = pd.to_datetime(h["signup_date"])
    h["net_amount"]    = h["price"] * h["quantity"] * (1 - h["discount"] / 100)
    h = h.sort_values("purchase_date").reset_index(drop=True)

    n    = len(h)
    last = h.iloc[-1]
    is_cold_start = (n == 1)          # only 1 purchase → not enough signal yet
    prior = h.iloc[:-1]               # everything except the last row

    feat = {
        # customer static
        "age"   : int(last["age"]),
        "gender": str(last["gender"]),
        "city"  : str(last["city"]),
        "state" : str(last["state"]),

        # cumulative history (prior purchases only — point-in-time, no leakage)
        "cum_orders_so_far"      : n - 1,
        "cum_spent_so_far"       : round(prior["net_amount"].sum(), 2) if n > 1 else 0,
        "cum_avg_price_so_far"   : round(prior["price"].mean(), 2) if n > 1 else 0,
        "cum_avg_discount_so_far": round(prior["discount"].mean(), 2) if n > 1 else 0,
        "cum_avg_rating_so_far"  : round(prior["product_rating"].mean(), 2) if n > 1 else 3.5,

        # temporal
        "days_since_prev_purchase" : (h["purchase_date"].iloc[-1]
                                       - h["purchase_date"].iloc[-2]).days if n > 1 else 0,
        "avg_gap_between_purchases": round(h["purchase_date"].diff().dt.days.mean(), 1) if n > 1 else 0,
        "days_since_signup"        : (h["purchase_date"].iloc[-1]
                                       - h["signup_date"].iloc[-1]).days,
        "purchase_month"     : h["purchase_date"].iloc[-1].month,
        "purchase_quarter"   : h["purchase_date"].iloc[-1].quarter,
        "purchase_dayofweek" : h["purchase_date"].iloc[-1].dayofweek,
        "n_unique_categories_so_far": int(prior["category"].nunique()) if n > 1 else 0,

        # affinity — from prior purchases only
        "most_frequent_category_so_far": (
            prior["category"].value_counts().index[0] if n > 1 else "Unknown"),
        "most_frequent_brand_so_far": (
            prior["brand"].value_counts().index[0] if n > 1 else "Unknown"),
        "prev_category"   : str(h["category"].iloc[-2]) if n > 1 else "Unknown",
        "prev_subcategory": str(h["subcategory"].iloc[-2]) if n > 1 else "Unknown",
        "prev_brand"      : str(h["brand"].iloc[-2]) if n > 1 else "Unknown",

        # current session signals (intent, not outcome)
        "pages_viewed"     : int(last.get("pages_viewed", 5) or 5),
        "wishlist_items"   : int(last.get("wishlist_items", 0) or 0),
        "cart_items"       : int(last.get("cart_items", 1) or 1),
        "interaction_score": float(last.get("interaction_score", 0.5) or 0.5),
        "device"           : str(last.get("device", "Mobile") or "Mobile"),
        "traffic_source"   : str(last.get("traffic_source", "Organic Search") or "Organic Search"),
        "search_keyword"   : str(last.get("search_keyword", "Unknown") or "Unknown"),
    }
    return feat, is_cold_start


def encode_customer_features(feat_dict):
    row = pd.DataFrame([feat_dict])
    for col in art["FEATURE_COLS"]:
        if col not in row.columns: row[col] = np.nan
    num_f = row[art["FEATURE_COLS"]].select_dtypes(include=[np.number]).columns
    row[num_f] = row[num_f].fillna(0)
    for c in art["cat_cols"]:
        le  = art["s1_label_encoders"].get(c)
        val = str(row[c].iloc[0]) if c in row.columns else "Unknown"
        if le:
            val = val if val in le.classes_ else "Unknown"
            try: row[c] = le.transform([val])[0]
            except: row[c] = 0
        else: row[c] = 0
    return row[art["FEATURE_COLS"]]


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Stage 1 model: predict next category
# ══════════════════════════════════════════════════════════════════════════
def predict_category(enc_row):
    if art["algo_type"] == "LightGBM":
        prob = cat_model.predict(enc_row, num_iteration=cat_model.best_iteration)[0]
    else:
        prob = cat_model.predict(xgb.DMatrix(enc_row))[0]
    top3_idx = prob.argsort()[-3:][::-1]
    top3 = [(art["cat_target_encoder"].inverse_transform([i])[0], round(float(prob[i]), 4))
            for i in top3_idx]
    return top3


def encode_rank(ced, prod_row):
    row = ced.copy()
    row.update({
        "prod_price"       : prod_row["price"],
        "prod_avg_rating"  : prod_row["prod_global_avg_rating"],
        "prod_popularity"  : prod_row["prod_global_popularity"],
        "prod_total_orders": prod_row["prod_total_orders"],
        "prod_avg_discount": prod_row["prod_avg_discount_given"],
        "prod_avg_qty"     : prod_row["prod_avg_qty_sold"],
        "brand_match"      : int(prod_row["brand"] == ced.get("most_frequent_brand_so_far", -1)),
        "category_match"   : int(prod_row["category"] == ced.get("most_frequent_category_so_far", -1)),
        "price_diff_from_avg": prod_row["price"] - ced.get("cum_avg_price_so_far", prod_row["price"]),
    })
    pm = {"prod_category": "category", "prod_subcategory": "subcategory", "prod_brand": "brand"}
    for c in art["rank_cat_cols"]:
        le   = art["rank_encoders"].get(c)
        rval = str(prod_row.get(pm.get(c, c), "Unknown"))
        if le:
            rval = rval if rval in le.classes_ else le.classes_[0]
            row[c] = le.transform([rval])[0]
        else: row[c] = 0
    return row


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 + 6 — Candidate selection + Stage 2 LambdaRank
# ══════════════════════════════════════════════════════════════════════════
def run_pipeline(feat_dict, top_n=5):
    enc_row  = encode_customer_features(feat_dict)
    top3     = predict_category(enc_row)
    pred_cat = top3[0][0]

    cands = pc[pc["category"] == pred_cat].copy()
    if len(cands) == 0:
        cands = pc.sample(min(top_n * 3, len(pc)))

    ced   = enc_row.iloc[0].to_dict()
    rrows = [encode_rank(ced, r) for _, r in cands.iterrows()]
    rdf   = pd.DataFrame(rrows)[art["RANK_FEATURES"]].fillna(0)
    cands = cands.copy()
    cands["rank_score"] = rank_model.predict(rdf, num_iteration=rank_model.best_iteration)
    prods = (cands.sort_values("rank_score", ascending=False).head(top_n)
                  [["product_id", "product_name", "category", "brand", "price", "rank_score"]]
                  .to_dict(orient="records"))
    return top3, prods


def cold_start_recommendation(history_df, top_n=5):
    """Only 1 purchase so far → not enough signal for the ML model.
    Fall back to demographic + global-popularity products (no ML)."""
    last = history_df.iloc[-1]
    demo_cat = str(last.get("category", "Unknown"))
    top_prods = (pc.sort_values("prod_global_popularity", ascending=False)
                   .head(top_n)[["product_id", "product_name", "category", "brand", "price"]]
                   .to_dict(orient="records"))
    return [(demo_cat, 0.0)], top_prods


@app.route("/health")
def health():
    db_ok = DBConnector().health_check()
    return jsonify({"status": "ok" if db_ok else "degraded", "db_connected": db_ok,
                     "model": art["algo_type"], "version": art["model_version"]})


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 + 3 + 4 + 5 + 6 + 7 + 8 — full per-customer pipeline, DB-backed
# ══════════════════════════════════════════════════════════════════════════
def predict_multiple(customer_ids, top_n):
    db = DBConnector()
    predictions = []

    for cid in customer_ids:
        try:
            # Step 3 — collect full customer history
            hist = db.load_customer_history(cid)

            if len(hist) == 0:
                predictions.append({
                    "customer_id": cid,
                    "status": "error",
                    "message": "Customer not found"
                })
                continue

            hist = hist.sort_values("purchase_date")
            feat_dict, is_cold_start = build_features_from_history(hist)

            if is_cold_start:
                cats, prods = cold_start_recommendation(hist, top_n)
                cold_strategy = "demographic_popularity"
            else:
                # Step 4 + 5 + 6 — category model → candidates → ranking model
                cats, prods = run_pipeline(feat_dict, top_n)
                cold_strategy = None

            # Step 8 — store prediction
            try:
                prediction_df = pd.DataFrame([{
                    "customer_id": cid,
                    "predicted_next_category": cats[0][0] if cats else None,
                    "category_confidence": cats[0][1] if cats else None,
                    "top3_categories": [
                        {"category": c, "confidence": p} for c, p in cats
                    ],
                    "recommended_product_id": prods[0].get("product_id") if prods else None,
                    "recommended_product_name": prods[0].get("product_name") if prods else None,
                    "rank_score": prods[0].get("rank_score") if prods else None,
                    "top5_products": prods,
                    "is_cold_start": is_cold_start,
                    "cold_start_strategy": cold_strategy
                }])
                db.write_predictions(prediction_df)
            except Exception as e:
                print(f"DB Save Error {cid}: {e}")

            predictions.append({
                "customer_id": cid,
                "predicted_at": datetime.now(timezone.utc).isoformat(),
                "is_cold_start": is_cold_start,
                "cold_start_strategy": cold_strategy,
                "predicted_categories": [
                    {"category": c, "confidence": p} for c, p in cats
                ],
                "recommended_products": prods
            })

        except Exception as e:
            predictions.append({
                "customer_id": cid,
                "status": "error",
                "message": str(e)
            })

    return predictions


# ══════════════════════════════════════════════════════════════════════════
# STEP 9 — Segment Generation (only for bulk prediction runs)
# ══════════════════════════════════════════════════════════════════════════
def create_segments(predictions):
    segment_data = {}

    for p in predictions:
        # Cold-start customers have no reliable category/rank signal yet —
        # skip them from campaign segments.
        if "predicted_categories" not in p or p.get("is_cold_start"):
            continue

        category   = p["predicted_categories"][0]["category"]
        confidence = p["predicted_categories"][0]["confidence"]
        products   = p.get("recommended_products", [])

        product_id   = products[0].get("product_id") if products else None
        product_name = products[0].get("product_name") if products else None
        rank_score   = products[0].get("rank_score") if products else None
        price        = products[0].get("price") if products else None

        if category not in segment_data:
            segment_data[category] = {
                "customers": [],
                "confidence": [],
                "rank_scores": [],
                "prices": [],
                "product_id": product_id,
                "product_name": product_name
            }

        segment_data[category]["customers"].append(p["customer_id"])
        segment_data[category]["confidence"].append(confidence)
        if rank_score is not None:
            segment_data[category]["rank_scores"].append(rank_score)
        if price is not None:
            segment_data[category]["prices"].append(price)

    rows = []
    for category, data in segment_data.items():
        rows.append({
            "segment_label": f"{category}_segment",
            "predicted_next_category": category,
            "recommended_product_name": data["product_name"],
            "recommended_product_id": data["product_id"],
            "customer_count": len(data["customers"]),
            "customer_ids": data["customers"],
            "avg_confidence": sum(data["confidence"]) / len(data["confidence"]),
            "avg_rank_score": (sum(data["rank_scores"]) / len(data["rank_scores"]))
                                if data["rank_scores"] else None,
            "avg_product_price": (sum(data["prices"]) / len(data["prices"]))
                                if data["prices"] else None,
        })

    return pd.DataFrame(rows)


@app.route("/predict", methods=["POST"])
def predict():
    print("Predict API Hit")
    body = request.get_json()
    top_n = int(body.get("top_n", 5))

    # Support single & multiple customers
    if "customer_ids" in body:
        customer_ids = body["customer_ids"]
    elif "customer_id" in body:
        customer_ids = [body["customer_id"]]
    else:
        return jsonify({
            "status": "error",
            "message": "customer_id or customer_ids required"
        }), 400

    predictions = predict_multiple(customer_ids, top_n)

    # Step 9 — segments only for bulk prediction
    if len(customer_ids) > 1:
        segment_df = create_segments(predictions)
        print("SEGMENT DATA:")
        print(segment_df)

        if len(segment_df) > 0:
            db = DBConnector()
            db.write_segments(segment_df)

    if len(predictions) == 1:
        return jsonify(predictions[0])

    return jsonify({
        "status": "ok",
        "total_predictions": len(predictions),
        "predictions": predictions
    })


# ══════════════════════════════════════════════════════════════════════════
# STEP 10 — Campaign Engine reads pending segments
# ══════════════════════════════════════════════════════════════════════════
@app.route("/segments", methods=["GET"])
def get_segments():
    try:
        db = DBConnector()
        segs = db.get_pending_segments()
        return jsonify({"status": "ok", "segments": segs.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/segments/<int:segment_id>/dispatch", methods=["POST"])
def dispatch_segment(segment_id):
    body = request.get_json()
    campaign_id = body.get("campaign_id")
    try:
        db = DBConnector()
        db.mark_segment_dispatched(segment_id, campaign_id)
        return jsonify({"status": "ok", "segment_id": segment_id, "dispatched": True})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 + 2 — Event Tracking → Customer State Update → (Purchase) → Predict
# db.process_event() already does: insert_event → ensure_customer_exists
#   → ensure_session_exists → update_session_metrics → process_purchase
# ══════════════════════════════════════════════════════════════════════════
@app.route("/track", methods=["POST"])
def track():
    event = request.get_json()
    db = DBConnector()
    db.process_event(event)

    # Purchase closes the loop: re-run the prediction pipeline for this customer
    if event.get("event_type") == "purchase":
        predict_multiple([event["customer_id"]], 5)

    return jsonify({"status": "success"})


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)








    
# from flask import Flask, request, jsonify
# import pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb
# import pickle, json, os
# from datetime import datetime, timezone
# from db_connector import DBConnector

# from predict_new_user import write_prediction_to_db
# app = Flask(__name__)

# # ── Load on startup ───────────────────────────────────────────────────────────
# with open("models/artifacts.pkl","rb") as f:
#     art = pickle.load(f)

# if art["algo_type"] == "LightGBM":
#     cat_model = lgb.Booster(model_file="models/stage1_category_model.lgb")
# else:
#     cat_model = xgb.Booster()
#     cat_model.load_model("models/stage1_category_model.xgb")

# rank_model = lgb.Booster(model_file="models/stage2_lambdarank_model.lgb")
# pc = art["product_catalog"]

# from sqlalchemy import create_engine
# import os

# DB_URL = os.getenv(
#     "DB_URL",
#     "postgresql:"
# )

# engine = create_engine(DB_URL, pool_pre_ping=True)
# def encode_customer_features(raw_row):
#     row = pd.DataFrame([raw_row])
#     for col in art["FEATURE_COLS"]:
#         if col not in row.columns: row[col] = np.nan
#     num_f = row[art["FEATURE_COLS"]].select_dtypes(include=[np.number]).columns
#     row[num_f] = row[num_f].fillna(0)
#     for c in art["cat_cols"]:
#         le  = art["s1_label_encoders"].get(c)
#         val = str(row[c].iloc[0]) if c in row.columns else "Unknown"
#         if le:
#             val = val if val in le.classes_ else "Unknown"
#             try: row[c] = le.transform([val])[0]
#             except: row[c] = 0
#         else: row[c] = 0
#     return row[art["FEATURE_COLS"]]


# def predict_category(enc_row):
#     if art["algo_type"] == "LightGBM":
#         prob = cat_model.predict(enc_row, num_iteration=cat_model.best_iteration)[0]
#     else:
#         prob = cat_model.predict(xgb.DMatrix(enc_row))[0]
#     top3_idx = prob.argsort()[-3:][::-1]
#     top3 = [(art["cat_target_encoder"].inverse_transform([i])[0], round(float(prob[i]),4))
#             for i in top3_idx]
#     return top3


# def encode_rank(ced, prod_row):
#     row = ced.copy()
#     row.update({
#         "prod_price"       : prod_row["price"],
#         "prod_avg_rating"  : prod_row["prod_global_avg_rating"],
#         "prod_popularity"  : prod_row["prod_global_popularity"],
#         "prod_total_orders": prod_row["prod_total_orders"],
#         "prod_avg_discount": prod_row["prod_avg_discount_given"],
#         "prod_avg_qty"     : prod_row["prod_avg_qty_sold"],
#         "brand_match"      : int(prod_row["brand"]==ced.get("most_frequent_brand_so_far",-1)),
#         "category_match"   : int(prod_row["category"]==ced.get("most_frequent_category_so_far",-1)),
#         "price_diff_from_avg": prod_row["price"]-ced.get("cum_avg_price_so_far",prod_row["price"]),
#     })
#     pm = {"prod_category":"category","prod_subcategory":"subcategory","prod_brand":"brand"}
#     for c in art["rank_cat_cols"]:
#         le   = art["rank_encoders"].get(c)
#         rval = str(prod_row.get(pm.get(c,c),"Unknown"))
#         if le:
#             rval = rval if rval in le.classes_ else le.classes_[0]
#             row[c] = le.transform([rval])[0]
#         else: row[c] = 0
#     return row


# def run_pipeline(customer_features, top_n=5):
#     enc      = encode_customer_features(customer_features)
#     top3     = predict_category(enc)
#     pred_cat = top3[0][0]

#     cands = pc[pc["category"]==pred_cat].copy()
#     if len(cands)==0: cands=pc.sample(min(top_n*3,len(pc)))
#     ced   = enc.iloc[0].to_dict()
#     rrows = [encode_rank(ced, r) for _,r in cands.iterrows()]
#     rdf   = pd.DataFrame(rrows)[art["RANK_FEATURES"]].fillna(0)
#     cands = cands.copy()
#     cands["rank_score"] = rank_model.predict(rdf)
#     prods = (cands.sort_values("rank_score",ascending=False).head(top_n)
#                   [["product_id","product_name","category","brand","price","rank_score"]]
#                   .to_dict(orient="records"))
#     return top3, prods


# @app.route("/health")
# def health():
#     return jsonify({"status":"ok","model":art["algo_type"],"version":art["model_version"]})

# def predict_multiple(customer_ids, top_n):
#     from db_connector import DBConnector

#     db = DBConnector()
#     predictions = []

#     for cid in customer_ids:

#         try:
#             # Load customer history
#             hist = db.load_customer_history(cid)

#             if len(hist) == 0:
#                 predictions.append({
#                     "customer_id": cid,
#                     "status": "error",
#                     "message": "Customer not found"
#                 })
#                 continue

#             # Sort history
#             hist = hist.sort_values("purchase_date")

#             # Create features
#             customer_features = hist.iloc[-1].to_dict()

#             # Prediction
#             cats, prods = run_pipeline(
#                 customer_features,
#                 top_n
#             )

#             # Save prediction
#             try:
#                 prediction_df = pd.DataFrame([{

#                     "customer_id": cid,

#                     "predicted_next_category": cats[0][0] if cats else None,

#                     "category_confidence": cats[0][1] if cats else None,

#                     "top3_categories": [
#                     {
#                          "category": c,
#                          "confidence": p
#                     }
#                     for c, p in cats
#                     ],

#                     "recommended_product_id": prods[0].get("product_id") if prods else None,

#                     "recommended_product_name": prods[0].get("product_name") if prods else None,

#                     "rank_score": prods[0].get("score") if prods else None,

#                     "top5_products": prods,

#                     "is_cold_start": False,

#                     "cold_start_strategy": None
#                 }])


#                 db.write_predictions(prediction_df)
#             except Exception as e:
#                 print(f"DB Save Error {cid}: {e}")

#             predictions.append({
#                 "customer_id": cid,
#                 "predicted_at": datetime.now(timezone.utc).isoformat(),
#                 "is_cold_start": False,
#                 "predicted_categories": [
#                     {
#                         "category": c,
#                         "confidence": p
#                     }
#                     for c, p in cats
#                 ],
#                 "recommended_products": prods
#             })

#         except Exception as e:
#             predictions.append({
#                 "customer_id": cid,
#                 "status": "error",
#                 "message": str(e)
#             })

#     return predictions

# # 
# def create_segments(predictions):

#     segment_data = {}

#     for p in predictions:

#         if "predicted_categories" not in p:
#             continue

#         category = p["predicted_categories"][0]["category"]
#         confidence = p["predicted_categories"][0]["confidence"]

#         products = p.get("recommended_products", [])

#         product_id = None
#         product_name = None

#         if products:
#             product_id = products[0].get("product_id")
#             product_name = products[0].get("product_name")


#         if category not in segment_data:
#             segment_data[category] = {
#                 "customers": [],
#                 "confidence": [],
#                 "product_id": product_id,
#                 "product_name": product_name
#             }


#         segment_data[category]["customers"].append(
#             p["customer_id"]
#         )

#         segment_data[category]["confidence"].append(
#             confidence
#         )


#     rows = []

#     for category, data in segment_data.items():

#         rows.append({

#             "segment_label": f"{category}_segment",

#             "predicted_next_category": category,

#             "recommended_product_name": data["product_name"],

#             "recommended_product_id": data["product_id"],

#             "customer_count": len(data["customers"]),

#             "customer_ids": data["customers"],

#             "avg_confidence": sum(data["confidence"]) / len(data["confidence"]),

#             "avg_rank_score": None,

#             "avg_product_price": None

#         })


#     return pd.DataFrame(rows)

# @app.route("/predict", methods=["POST"])
# def predict():
#     print("Predict API Hit")

#     body = request.get_json()

#     top_n = int(body.get("top_n", 5))

#     # Support single & multiple customers
#     if "customer_ids" in body:
#         customer_ids = body["customer_ids"]

#     elif "customer_id" in body:
#         customer_ids = [body["customer_id"]]

#     else:
#         return jsonify({
#             "status": "error",
#             "message": "customer_id or customer_ids required"
#         }), 400

#     predictions = predict_multiple(
#         customer_ids,
#         top_n
#     )
#     # Create customer segments only for bulk prediction
#     if len(customer_ids) > 1:

#         segment_df = create_segments(predictions)

#         print("SEGMENT DATA:")
#         print(segment_df)

#         if len(segment_df) > 0:
#             db = DBConnector()
#             db.write_segments(segment_df)

#     # Single response
#     if len(predictions) == 1:
#         return jsonify(predictions[0])

#     # Multiple response
#     return jsonify({
#         "status": "ok",
#         "total_predictions": len(predictions),
#         "predictions": predictions
#     })


# @app.route("/segments", methods=["GET"])
# def get_segments():
#     # Returns pending segments for campaign dispatch
#     try:
#         from db_connector import DBConnector
#         db = DBConnector()
#         segs = db.get_pending_segments()
#         return jsonify({"status":"ok","segments":segs.to_dict(orient="records")})
#     except Exception as e:
#         return jsonify({"status":"error","message":str(e)}), 500


# @app.route("/segments/<int:segment_id>/dispatch", methods=["POST"])
# def dispatch_segment(segment_id):
#     # Mark a segment as dispatched after campaign system picks it up
#     body = request.get_json()
#     campaign_id = body.get("campaign_id")
#     try:
#         from db_connector import DBConnector
#         db = DBConnector()
#         db.mark_segment_dispatched(segment_id, campaign_id)
#         return jsonify({"status":"ok","segment_id":segment_id,"dispatched":True})
#     except Exception as e:
#         return jsonify({"status":"error","message":str(e)}), 500

# @app.route("/track",methods=["POST"])
# def track():

#     event=request.get_json()

#     db=DBConnector()

#     db.insert_event(event)

#     db.ensure_customer_exists(
#         event["customer_id"]
#     )

#     db.ensure_session_exists(
#         event["customer_id"]
#     )

#     db.update_session_metrics(event)

#     if event["event_type"]=="purchase":

#         db.process_purchase(event)

#         predict_multiple(
#             [event["customer_id"]],
#             5
#         )

#     return jsonify({
#         "status":"success"
#     })

# if __name__ == "__main__":
#     port = int(os.getenv("FLASK_PORT","5000"))
#     app.run(host="0.0.0.0", port=port, debug=False)


