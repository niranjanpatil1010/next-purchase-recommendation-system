"""
seed_database.py
────────────────
Reads the flat ecommerce CSV and populates all PostgreSQL tables.

Run ONCE after schema.sql:
    psql -U postgres -d ecommerce_ml -f schema.sql
    python seed_database.py

Environment variable:
    DB_URL = postgresql://user:password@localhost:5432/ecommerce-ml
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = os.getenv("CSV_PATH", "dataset.csv")
DB_URL = os.getenv(
    "DB_URL",
    "postgresql://postgres:pg%401224@localhost:5432/ecommerce-ml  "
)
BATCH    = 5000      # rows per insert batch

engine = create_engine(DB_URL, pool_pre_ping=True)

def log(msg): print(f"  {msg}")

def bulk_insert(conn, table, df, conflict_col):
    """Insert with ON CONFLICT DO NOTHING using temp copy trick."""
    if df.empty:
        return 0
    df.to_sql("_tmp_seed", conn, if_exists="replace", index=False)
    cols = ", ".join(df.columns)
    conn.execute(text(f"""
        INSERT INTO {table} ({cols})
        SELECT {cols} FROM _tmp_seed
        ON CONFLICT ({conflict_col}) DO NOTHING
    """))
    conn.execute(text("DROP TABLE IF EXISTS _tmp_seed"))
    return len(df)


# ── Load CSV ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SEED DATABASE — Reading CSV")
print("="*60)
t0 = time.time()

df = pd.read_csv(CSV_PATH)
df["purchase_date"] = pd.to_datetime(df["purchase_date"])
df["signup_date"]   = pd.to_datetime(df["signup_date"])
print(f"  Loaded {len(df):,} rows from {CSV_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. dim_customers
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] dim_customers ...")
customers = (
    df.groupby("customer_id")
    .agg(
        age         = ("age",         "first"),
        gender      = ("gender",      "first"),
        city        = ("city",        "first"),
        state       = ("state",       "first"),
        signup_date = ("signup_date", "first"),
    )
    .reset_index()
)
# add placeholder email/phone (not in dataset — real system would have these)
customers["email"] = customers["customer_id"].str.lower() + "@example.com"
customers["phone"] = None

with engine.begin() as conn:
    n = bulk_insert(conn, "dim_customers", customers, "customer_id")
log(f"Inserted {n:,} customers")


# ─────────────────────────────────────────────────────────────────────────────
# 2. dim_products
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] dim_products ...")
products = (
    df.groupby("product_id")
    .agg(
        product_name       = ("product_name",       "first"),
        brand              = ("brand",              "first"),
        category           = ("category",           "first"),
        subcategory        = ("subcategory",        "first"),
        base_price         = ("price",              "mean"),    # avg across orders as base
        product_rating     = ("product_rating",     "mean"),
        product_popularity = ("product_popularity", "max"),
    )
    .reset_index()
    .rename(columns={"product_id": "product_id"})
)
products["base_price"]     = products["base_price"].round(2)
products["product_rating"] = products["product_rating"].round(1)
products["is_active"]      = True

with engine.begin() as conn:
    n = bulk_insert(conn, "dim_products", products, "product_id")
log(f"Inserted {n:,} products")


# ─────────────────────────────────────────────────────────────────────────────
# 3. dim_campaigns  (upsert by name)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] dim_campaigns ...")
camp_names = df[df["campaign_name"] != "No Campaign"]["campaign_name"].unique()

TYPE_MAP = {
    "Diwali Sale"              : "EMAIL",
    "Summer Bonanza"           : "EMAIL",
    "New Year Offer"           : "EMAIL",
    "Flash Sale Weekend"       : "PUSH",
    "Loyalty Rewards"          : "EMAIL",
    "Cart Abandon Reminder"    : "WHATSAPP",
    "New Arrivals Alert"       : "PUSH",
    "Republic Day Sale"        : "EMAIL",
    "Monsoon Mega Sale"        : "EMAIL",
    "Birthday Special"         : "WHATSAPP",
}

with engine.begin() as conn:
    for name in camp_names:
        conn.execute(text("""
            INSERT INTO dim_campaigns (campaign_name, campaign_type)
            VALUES (:name, :ctype)
            ON CONFLICT DO NOTHING
        """), {"name": name, "ctype": TYPE_MAP.get(name, "EMAIL")})

# Fetch back mapping name -> uuid
with engine.connect() as conn:
    camp_df = pd.read_sql("SELECT campaign_id, campaign_name FROM dim_campaigns", conn)
camp_map = dict(zip(camp_df["campaign_name"], camp_df["campaign_id"]))
log(f"Campaigns: {len(camp_map)} rows")


# ─────────────────────────────────────────────────────────────────────────────
# 4. fact_orders
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] fact_orders ...")
orders = df[["order_id","customer_id","purchase_date","purchase_number",
             "coupon_used","traffic_source"]].drop_duplicates("order_id").copy()
orders["coupon_used"] = orders["coupon_used"].astype(bool)

# total_amount per order
order_amounts = (
    (df["price"] * df["quantity"] * (1 - df["discount"]/100))
    .groupby(df["order_id"]).sum().reset_index()
    .rename(columns={0: "total_amount"})
)
orders = orders.merge(order_amounts, on="order_id", how="left")
orders["total_amount"] = orders["total_amount"].round(2)

with engine.begin() as conn:
    for i in range(0, len(orders), BATCH):
        bulk_insert(conn, "fact_orders",
                    orders.iloc[i:i+BATCH], "order_id")
log(f"Inserted {len(orders):,} orders")


# ─────────────────────────────────────────────────────────────────────────────
# 5. fact_order_items
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] fact_order_items ...")
items = df[["order_id","product_id","quantity","price","discount"]].copy()
items = items.rename(columns={"price": "unit_price", "discount": "discount_pct"})
items["unit_price"]   = items["unit_price"].round(2)
items["discount_pct"] = items["discount_pct"].round(2)

with engine.begin() as conn:
    # fact_order_items has BIGSERIAL PK (no natural key) — plain insert
    for i in range(0, len(items), BATCH):
        batch = items.iloc[i:i+BATCH]
        batch.to_sql("fact_order_items", conn, if_exists="append", index=False)
        if i % 20000 == 0:
            log(f"  ... {i:,}/{len(items):,}")
log(f"Inserted {len(items):,} order items")


# ─────────────────────────────────────────────────────────────────────────────
# 6. fact_sessions
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] fact_sessions ...")
sessions = df[["session_id","customer_id","order_id","purchase_date",
               "device","traffic_source","search_keyword",
               "pages_viewed","wishlist_items","cart_items",
               "interaction_score"]].copy()
sessions = sessions.rename(columns={"purchase_date": "session_date"})
sessions["interaction_score"] = sessions["interaction_score"].round(3)

with engine.begin() as conn:
    for i in range(0, len(sessions), BATCH):
        bulk_insert(conn, "fact_sessions",
                    sessions.iloc[i:i+BATCH], "session_id")
log(f"Inserted {len(sessions):,} sessions")


# ─────────────────────────────────────────────────────────────────────────────
# 7. fact_campaign_interactions
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] fact_campaign_interactions ...")
camp_rows = df[df["campaign_sent"] == 1][
    ["customer_id","campaign_name","campaign_sent",
     "campaign_opened","campaign_clicked","coupon_used","purchase_date"]
].copy()

camp_rows["campaign_id"] = camp_rows["campaign_name"].map(camp_map)
camp_rows = camp_rows.dropna(subset=["campaign_id"])

camp_rows = camp_rows.rename(columns={"purchase_date": "sent_at"})
camp_rows["opened_at"]  = camp_rows.apply(
    lambda r: r["sent_at"] if r["campaign_opened"] == 1 else None, axis=1)
camp_rows["clicked_at"] = camp_rows.apply(
    lambda r: r["sent_at"] if r["campaign_clicked"] == 1 else None, axis=1)
camp_rows["coupon_used"] = camp_rows["coupon_used"].astype(bool)
camp_rows["coupon_code"] = camp_rows.apply(
    lambda r: f"COUP{abs(hash(r['customer_id']))%10000:04d}" if r["coupon_used"] else None, axis=1)

camp_insert = camp_rows[["customer_id","campaign_id","sent_at",
                          "opened_at","clicked_at","coupon_code","coupon_used"]]

with engine.begin() as conn:
    for i in range(0, len(camp_insert), BATCH):
        camp_insert.iloc[i:i+BATCH].to_sql(
            "fact_campaign_interactions", conn, if_exists="append", index=False)
log(f"Inserted {len(camp_insert):,} campaign interactions")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
elapsed = round(time.time() - t0, 1)
print()
print("="*60)
print("  SEED COMPLETE")
print("="*60)

with engine.connect() as conn:
    for table in ["dim_customers","dim_products","dim_campaigns",
                  "fact_orders","fact_order_items","fact_sessions",
                  "fact_campaign_interactions"]:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        print(f"  {table:<35s}: {count:>8,} rows")

print(f"\n  Total time: {elapsed}s")
print()
print("  Next step:")
print("    python Model1_Next_Purchase_Prediction_FINAL.ipynb  (train models)")
print("    python app.py                                        (start Flask API)")
