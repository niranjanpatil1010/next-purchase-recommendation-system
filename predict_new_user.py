"""
predict_new_user.py
────────────────────────────────────────────────────────────────
New user aaya DB mein → 10-20 purchases track hue → predict karo

FLOW:
  1. User registers     → dim_customers mein insert
  2. User buys product  → fact_orders + fact_order_items + fact_sessions insert
  3. Enough history?    → predict_for_new_user() call karo
  4. Result             → ml_predictions table + segment output

Run:
    python predict_new_user.py
    (models/ folder hona chahiye — notebook pehle run karo)
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, date
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/ecommerce_ml")

# ── Load trained models ───────────────────────────────────────────────────────
print("Loading models...")
with open("models/artifacts.pkl", "rb") as f:
    art = pickle.load(f)

cat_model  = lgb.Booster(model_file="models/stage1_category_model.lgb")
rank_model = lgb.Booster(model_file="models/stage2_lambdarank_model.lgb")
pc         = art["product_catalog"]                  # product catalog DataFrame
cat_to_prods = art["cat_to_products"]
print(f"Models loaded. Categories: {art['cat_target_encoder'].classes_}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Write new user events to DB (simulates your e-commerce backend)
# ═══════════════════════════════════════════════════════════════════════════

def register_customer(engine, customer_id, age, gender, city, state, signup_date=None):
    """Called when user signs up."""
    signup_date = signup_date or date.today().isoformat()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO dim_customers
                (customer_id, age, gender, city, state, signup_date, email)
            VALUES
                (:cid, :age, :gender, :city, :state, :sdate, :email)
            ON CONFLICT (customer_id) DO NOTHING
        """), {
            "cid"   : customer_id,
            "age"   : age,
            "gender": gender,
            "city"  : city,
            "state" : state,
            "sdate" : signup_date,
            "email" : f"{customer_id.lower()}@example.com",
        })
    print(f"  [DB] Customer registered: {customer_id}")


def record_purchase(engine, order_id, customer_id, product_id, purchase_number,
                    unit_price, discount_pct, quantity,
                    session_id, device, traffic_source, search_keyword,
                    pages_viewed, wishlist_items, cart_items, interaction_score,
                    purchase_date=None):
    """Called every time a user completes a purchase."""
    purchase_date = purchase_date or datetime.now().isoformat()
    net_amount    = unit_price * quantity * (1 - discount_pct / 100)

    with engine.begin() as conn:
        # fact_orders
        conn.execute(text("""
            INSERT INTO fact_orders
                (order_id, customer_id, purchase_date, purchase_number,
                 total_amount, coupon_used, traffic_source)
            VALUES (:oid, :cid, :pdate, :pnum, :total, false, :tsrc)
            ON CONFLICT (order_id) DO NOTHING
        """), {
            "oid"  : order_id,
            "cid"  : customer_id,
            "pdate": purchase_date,
            "pnum" : purchase_number,
            "total": round(net_amount, 2),
            "tsrc" : traffic_source,
        })

        # fact_order_items
        conn.execute(text("""
            INSERT INTO fact_order_items
                (order_id, product_id, quantity, unit_price, discount_pct)
            VALUES (:oid, :pid, :qty, :price, :disc)
        """), {
            "oid"  : order_id,
            "pid"  : product_id,
            "qty"  : quantity,
            "price": round(unit_price, 2),
            "disc" : round(discount_pct, 2),
        })

        # fact_sessions
        conn.execute(text("""
            INSERT INTO fact_sessions
                (session_id, customer_id, order_id, session_date,
                 device, traffic_source, search_keyword,
                 pages_viewed, wishlist_items, cart_items, interaction_score)
            VALUES
                (:sid, :cid, :oid, :sdate,
                 :dev, :tsrc, :kw,
                 :pv, :wi, :ci, :iscore)
            ON CONFLICT (session_id) DO NOTHING
        """), {
            "sid"   : session_id,
            "cid"   : customer_id,
            "oid"   : order_id,
            "sdate" : purchase_date,
            "dev"   : device,
            "tsrc"  : traffic_source,
            "kw"    : search_keyword,
            "pv"    : pages_viewed,
            "wi"    : wishlist_items,
            "ci"    : cart_items,
            "iscore": round(interaction_score, 3),
        })

    print(f"  [DB] Purchase recorded: {order_id} | product={product_id} | ₹{net_amount:.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Read customer history from DB and build features
# ═══════════════════════════════════════════════════════════════════════════

def load_customer_history_from_db(engine, customer_id):
    """
    Reads everything from DB for this customer.
    Returns flat DataFrame like the training CSV — same columns.
    """
    sql = """
        SELECT
            c.customer_id,
            c.age,
            c.gender,
            c.city,
            c.state,
            c.signup_date,
            o.order_id,
            o.purchase_date,
            o.purchase_number,
            p.product_id,
            p.product_name,
            p.brand,
            p.category,
            p.subcategory,
            oi.unit_price          AS price,
            oi.discount_pct        AS discount,
            oi.quantity,
            p.product_rating,
            p.product_popularity,
            s.device,
            s.traffic_source,
            s.search_keyword,
            s.pages_viewed,
            s.wishlist_items,
            s.cart_items,
            s.interaction_score
        FROM dim_customers        c
        JOIN fact_orders          o  ON c.customer_id = o.customer_id
        JOIN fact_order_items     oi ON o.order_id    = oi.order_id
        JOIN dim_products         p  ON oi.product_id = p.product_id
        LEFT JOIN fact_sessions   s  ON o.order_id    = s.order_id
        WHERE c.customer_id = :cid
        ORDER BY o.purchase_date, o.purchase_number
    """
    df = pd.read_sql(text(sql), engine, params={"cid": customer_id})
    return df


def build_features_from_history(history_df):
    """
    Point-in-time feature engineering on the LATEST state of the customer.
    Returns (feature_dict, is_cold_start).
    """
    if len(history_df) == 0:
        return None, True   # no history at all

    h = history_df.copy()
    h["purchase_date"] = pd.to_datetime(h["purchase_date"])
    h["signup_date"]   = pd.to_datetime(h["signup_date"])
    h["net_amount"]    = h["price"] * h["quantity"] * (1 - h["discount"] / 100)
    h = h.sort_values("purchase_date").reset_index(drop=True)

    n    = len(h)
    last = h.iloc[-1]
    is_cold_start = (n == 1)

    # All history EXCEPT the last row = "prior purchases" (point-in-time)
    prior = h.iloc[:-1]

    feat = {
        # customer static
        "age"         : int(last["age"]),
        "gender"      : str(last["gender"]),
        "city"        : str(last["city"]),
        "state"       : str(last["state"]),

        # cumulative history (prior only)
        "cum_orders_so_far"         : n - 1,
        "cum_spent_so_far"          : round(prior["net_amount"].sum(), 2) if n > 1 else 0,
        "cum_avg_price_so_far"      : round(prior["price"].mean(), 2) if n > 1 else 0,
        "cum_avg_discount_so_far"   : round(prior["discount"].mean(), 2) if n > 1 else 0,
        "cum_avg_rating_so_far"     : round(prior["product_rating"].mean(), 2) if n > 1 else 3.5,

        # temporal
        "days_since_prev_purchase"  : (h["purchase_date"].iloc[-1]
                                        - h["purchase_date"].iloc[-2]).days if n > 1 else 0,
        "avg_gap_between_purchases" : round(h["purchase_date"].diff().dt.days.mean(), 1) if n > 1 else 0,
        "days_since_signup"         : (h["purchase_date"].iloc[-1]
                                        - h["signup_date"].iloc[-1]).days,
        "purchase_month"            : h["purchase_date"].iloc[-1].month,
        "purchase_quarter"          : h["purchase_date"].iloc[-1].quarter,
        "purchase_dayofweek"        : h["purchase_date"].iloc[-1].dayofweek,
        "n_unique_categories_so_far": int(prior["category"].nunique()) if n > 1 else 0,

        # affinity — from prior purchases only
        "most_frequent_category_so_far": (
            prior["category"].value_counts().index[0] if n > 1 else "Unknown"),
        "most_frequent_brand_so_far": (
            prior["brand"].value_counts().index[0] if n > 1 else "Unknown"),
        "prev_category"             : str(h["category"].iloc[-2]) if n > 1 else "Unknown",
        "prev_subcategory"          : str(h["subcategory"].iloc[-2]) if n > 1 else "Unknown",
        "prev_brand"                : str(h["brand"].iloc[-2]) if n > 1 else "Unknown",

        # session signals (current session — intent signal, NOT the outcome)
        "pages_viewed"              : int(last.get("pages_viewed", 5) or 5),
        "wishlist_items"            : int(last.get("wishlist_items", 0) or 0),
        "cart_items"                : int(last.get("cart_items", 1) or 1),
        "interaction_score"         : float(last.get("interaction_score", 0.5) or 0.5),
        "device"                    : str(last.get("device", "Mobile") or "Mobile"),
        "traffic_source"            : str(last.get("traffic_source", "Organic Search") or "Organic Search"),
        "search_keyword"            : str(last.get("search_keyword", "Unknown") or "Unknown"),
    }
    return feat, is_cold_start


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Encode + predict
# ═══════════════════════════════════════════════════════════════════════════

def encode_features(feat_dict):
    row = pd.DataFrame([feat_dict])
    for col in art["FEATURE_COLS"]:
        if col not in row.columns:
            row[col] = np.nan
    num_f = row[art["FEATURE_COLS"]].select_dtypes(include=[np.number]).columns
    row[num_f] = row[num_f].fillna(0)
    for c in art["cat_cols"]:
        le  = art["s1_label_encoders"].get(c)
        val = str(row[c].iloc[0]) if c in row.columns else "Unknown"
        if le:
            val = val if val in le.classes_ else "Unknown"
            try:    row[c] = le.transform([val])[0]
            except: row[c] = 0
        else:
            row[c] = 0
    return row[art["FEATURE_COLS"]]


def encode_rank_row(cust_enc, prod_row):
    row = cust_enc.copy()
    row.update({
        "prod_price"        : prod_row["price"],
        "prod_avg_rating"   : prod_row["prod_global_avg_rating"],
        "prod_popularity"   : prod_row["prod_global_popularity"],
        "prod_total_orders" : prod_row["prod_total_orders"],
        "prod_avg_discount" : prod_row["prod_avg_discount_given"],
        "prod_avg_qty"      : prod_row["prod_avg_qty_sold"],
        "brand_match"       : int(prod_row["brand"] == cust_enc.get("most_frequent_brand_so_far", -1)),
        "category_match"    : int(prod_row["category"] == cust_enc.get("most_frequent_category_so_far", -1)),
        "price_diff_from_avg": prod_row["price"] - cust_enc.get("cum_avg_price_so_far", prod_row["price"]),
    })
    pm = {"prod_category": "category", "prod_subcategory": "subcategory", "prod_brand": "brand"}
    for c in art["rank_cat_cols"]:
        le   = art["rank_encoders"].get(c)
        rval = str(prod_row.get(pm.get(c, c), "Unknown"))
        if le:
            rval = rval if rval in le.classes_ else le.classes_[0]
            row[c] = le.transform([rval])[0]
        else:
            row[c] = 0
    return row


def run_pipeline(feat_dict, top_n=5):
    """Stage 1 (category) + Stage 2 (rank) prediction."""
    enc_row  = encode_features(feat_dict)
    cat_prob = cat_model.predict(enc_row, num_iteration=cat_model.best_iteration)[0]
    top3_idx = cat_prob.argsort()[-3:][::-1]
    top3_cats = [
        (art["cat_target_encoder"].inverse_transform([i])[0], round(float(cat_prob[i]), 4))
        for i in top3_idx
    ]
    pred_category = top3_cats[0][0]

    candidates = pc[pc["category"] == pred_category].copy()
    if len(candidates) == 0:
        candidates = pc.sample(min(top_n * 3, len(pc)))

    ced   = enc_row.iloc[0].to_dict()
    rrows = [encode_rank_row(ced, p) for _, p in candidates.iterrows()]
    rdf   = pd.DataFrame(rrows)[art["RANK_FEATURES"]].fillna(0)
    scores = rank_model.predict(rdf, num_iteration=rank_model.best_iteration)

    candidates = candidates.copy()
    candidates["rank_score"] = scores
    top_prods = (
        candidates.sort_values("rank_score", ascending=False)
        .head(top_n)
        [["product_id", "product_name", "category", "subcategory",
          "brand", "price", "prod_global_avg_rating", "prod_global_popularity", "rank_score"]]
        .reset_index(drop=True)
    )
    return top3_cats, top_prods


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Write result back to DB
# ═══════════════════════════════════════════════════════════════════════════

def write_prediction_to_db(engine, customer_id, top3_cats, top_prods, is_cold_start=False):
    top1_cat  = top3_cats[0][0] if top3_cats else None
    top1_conf = top3_cats[0][1] if top3_cats else None
    top1_prod = top_prods.iloc[0] if len(top_prods) > 0 else None

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO ml_predictions (
                customer_id, model_version,
                predicted_next_category, category_confidence, top3_categories,
                recommended_product_id, recommended_product_name,
                rank_score, top5_products,
                is_cold_start
            ) VALUES (
                :cid, :mv,
                :cat, :conf, :top3::jsonb,
                :pid, :pname,
                :rs,  :top5::jsonb,
                :cs
            )
        """), {
            "cid"  : customer_id,
            "mv"   : art.get("model_version", "v1.0"),
            "cat"  : top1_cat,
            "conf" : top1_conf,
            "top3" : json.dumps([{"category": c, "prob": p} for c, p in top3_cats]),
            "pid"  : top1_prod["product_id"] if top1_prod is not None else None,
            "pname": top1_prod["product_name"] if top1_prod is not None else None,
            "rs"   : float(top1_prod["rank_score"]) if top1_prod is not None else None,
            "top5" : json.dumps(top_prods[["product_id","product_name","rank_score"]]
                                .to_dict(orient="records")) if len(top_prods) > 0 else "[]",
            "cs"   : is_cold_start,
        })
    print(f"  [DB] Prediction saved for {customer_id}: '{top1_cat}' → '{top1_prod['product_name'] if top1_prod is not None else None}'")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Full end-to-end demo
# ═══════════════════════════════════════════════════════════════════════════

def predict_for_new_user(customer_id, top_n=5, use_db=True):
    """
    Main entry point.
    Given a customer_id, reads their history from DB (or simulated),
    builds features, runs pipeline, returns + writes prediction.
    """
    print(f"\n{'='*60}")
    print(f"  Predicting for customer: {customer_id}")
    print(f"{'='*60}")

    if use_db:
        try:
            engine = create_engine(DB_URL, pool_pre_ping=True)
            history_df = load_customer_history_from_db(engine, customer_id)
        except Exception as e:
            print(f"  DB not available: {e}")
            return None
    else:
        # Simulated history (for testing without DB)
        history_df = _simulate_history(customer_id)

    print(f"  History loaded: {len(history_df)} purchase events")

    if len(history_df) == 0:
        print("  No history found — cannot predict (cold-start)")
        return None

    feat_dict, is_cold_start = build_features_from_history(history_df)

    if is_cold_start:
        print("  Cold-start: only 1 purchase — using demographic popularity model")
        # Use cold-start fallback (from notebook)
        age    = int(history_df.iloc[0]["age"])
        gender = str(history_df.iloc[0]["gender"])
        print(f"  Demographic: age={age}, gender={gender}")
        print("  Recommendation: Top products in age-gender demographic")
        return {"is_cold_start": True, "customer_id": customer_id}

    # Full Stage 1 + Stage 2 pipeline
    print(f"\n  Running Stage 1 (category prediction)...")
    top3_cats, top_prods = run_pipeline(feat_dict, top_n=top_n)

    # Print result
    print(f"\n  Top-3 Predicted Next Categories:")
    for cat, prob in top3_cats:
        bar = "█" * int(prob * 30)
        print(f"    {cat:<22s}  {prob:.4f}  {bar}")

    print(f"\n  Top-{top_n} Recommended Products (Stage 2 LambdaRank):")
    print(f"  {'#':<3} {'Product':<35} {'Brand':<14} {'Price':>8}  {'Score':>6}")
    print(f"  {'-'*3} {'-'*35} {'-'*14} {'-'*8}  {'-'*6}")
    for i, (_, row) in enumerate(top_prods.iterrows(), 1):
        print(f"  {i:<3} {row['product_name'][:34]:<35} {row['brand'][:13]:<14} "
              f"₹{row['price']:>7.0f}  {row['rank_score']:.4f}")

    print(f"\n  Customer has {len(history_df)} purchases in history")
    print(f"  Favorite category (so far): {feat_dict['most_frequent_category_so_far']}")
    print(f"  Favorite brand (so far)   : {feat_dict['most_frequent_brand_so_far']}")

    # Write to DB if available
    if use_db:
        try:
            write_prediction_to_db(engine, customer_id, top3_cats, top_prods, is_cold_start)
        except Exception as e:
            print(f"  [WARN] Could not write to DB: {e}")

    return {
        "customer_id"          : customer_id,
        "is_cold_start"        : False,
        "predicted_categories" : top3_cats,
        "recommended_products" : top_prods.to_dict(orient="records"),
        "features_used"        : feat_dict,
    }


def _simulate_history(customer_id):
    """
    Fallback: read from CSV instead of DB (for testing without PostgreSQL).
    Simulates exactly what load_customer_history_from_db() would return.
    """
    try:
        df = pd.read_csv("ecommerce_next_purchase_dataset.csv")
        df["purchase_date"] = pd.to_datetime(df["purchase_date"])
        df["signup_date"]   = pd.to_datetime(df["signup_date"])
        hist = df[df["customer_id"] == customer_id].sort_values("purchase_date")
        if len(hist) == 0:
            # Simulate a brand-new user with N purchases
            print(f"  [SIM] No history for {customer_id} — simulating 15 purchases")
            sample = df.sample(15, random_state=42).copy()
            sample["customer_id"] = customer_id
            return sample
        return hist
    except:
        return pd.DataFrame()


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Try real DB first, fall back to CSV simulation
    try:
        engine_test = create_engine(DB_URL, pool_pre_ping=True)
        with engine_test.connect() as c:
            c.execute(text("SELECT 1"))
        USE_DB = True
        print("PostgreSQL connected.")
    except:
        USE_DB = False
        print("PostgreSQL not available — using CSV simulation mode.")

    print("\n" + "="*60)
    print("  DEMO: Predicting for 3 different customers")
    print("="*60)

    # Customer with long history (warm)
    result1 = predict_for_new_user("C00013", top_n=5, use_db=USE_DB)

    # Customer with fewer purchases (simulate new user with 12 events)
    result2 = predict_for_new_user("C00099", top_n=5, use_db=USE_DB)

    # Simulate a completely new user (ID not in dataset)
    result3 = predict_for_new_user("C_NEW_001", top_n=5, use_db=USE_DB)

    print("\n" + "="*60)
    print("  HOW NEW USER FLOW WORKS")
    print("="*60)
    print("""
  1. USER REGISTERS
     → register_customer(engine, customer_id, age, gender, ...)
     → Row added to dim_customers

  2. USER BUYS (repeat for each purchase)
     → record_purchase(engine, order_id, customer_id, product_id, ...)
     → Rows added to: fact_orders, fact_order_items, fact_sessions

  3. PREDICT (call anytime after 2+ purchases)
     → predict_for_new_user(customer_id)
     → Reads full history from DB via v_customer_purchase_history view
     → Builds point-in-time features from history
     → Stage 1: predicts next category (LightGBM)
     → Stage 2: ranks products within category (LambdaRank)
     → Writes result to ml_predictions table

  4. CAMPAIGN SYSTEM picks up ml_predictions
     → Groups customers by recommended_product_name
     → Sends targeted campaign

  COLD-START (only 1 purchase):
     → Demographic + popularity fallback
     → No ML model used (not enough signal)
     → After 2nd purchase → full pipeline kicks in

  MIN HISTORY RECOMMENDED: 3-5 purchases for good accuracy
  SWEET SPOT              : 8+ purchases → strong category affinity signal
""")
