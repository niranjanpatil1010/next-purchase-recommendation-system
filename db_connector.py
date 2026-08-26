"""
db_connector.py
───────────────
PostgreSQL connector for the Next Purchase ML pipeline.
Reads training data from DB, writes predictions & segments back.

Usage:
    from db_connector import DBConnector
    db = DBConnector()                       # reads DB_URL from env
    df = db.load_purchase_history()          # for training
    db.write_predictions(predictions_df)
    db.write_segments(segments_df)
"""

import os
import uuid
import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ── Connection string ───────────────────────────────────────────────────────
# Set this in your environment or .env file:
#   export DB_URL="postgresql://user:password@localhost:5432/ecommerce_ml"
DB_URL = os.getenv("DB_URL")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")


class DBConnector:
    def __init__(self, db_url: str = DB_URL):
        self.engine: Engine = create_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,      # auto-reconnect on stale connections
            connect_args={"options": "-c timezone=UTC"},
        )
        logger.info("DBConnector initialized — %s", db_url.split("@")[-1])

    # ── READ: full purchase history for model training ──────────────────────
    def load_purchase_history(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Returns the flat purchase history view used by the ML pipeline.
        One row = one order-item event, sorted by customer + purchase_date.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        sql = f"""
            SELECT *
            FROM   v_customer_purchase_history
            ORDER  BY customer_id, purchase_date, purchase_number
            {limit_clause}
        """
        df = pd.read_sql(sql, self.engine)
        logger.info("Loaded %d rows from v_customer_purchase_history", len(df))
        return df

    # ── READ: single customer's history (real-time inference) ───────────────
    def load_customer_history(self, customer_id: str) -> pd.DataFrame:
        sql = """
            SELECT *
            FROM   v_customer_purchase_history
            WHERE  customer_id = :cid
            ORDER  BY purchase_date, purchase_number ASC
        """
        df = pd.read_sql(text(sql), self.engine, params={"cid": customer_id})
        logger.info("Loaded %d rows for customer %s", len(df), customer_id)
        return df

    # ── READ: product catalog for candidate generation ───────────────────────
    def load_product_catalog(self) -> pd.DataFrame:
        sql = "SELECT * FROM v_product_stats ORDER BY category, product_name"
        df = pd.read_sql(sql, self.engine)
        logger.info("Loaded %d products", len(df))
        return df

    # ── READ: all customers needing predictions (batch run) ──────────────────
    def load_customers_for_prediction(self) -> pd.DataFrame:
        """
        Returns latest state per customer — the model will predict their
        NEXT purchase. Excludes customers predicted in last 24 hours to
        avoid redundant runs.
        """
        sql = """
            SELECT rfm.*
            FROM   v_customer_rfm rfm
            WHERE  rfm.customer_id NOT IN (
                SELECT customer_id
                FROM   ml_predictions
                WHERE  predicted_at >= NOW() - INTERVAL '24 hours'
            )
            ORDER  BY rfm.customer_id
        """
        df = pd.read_sql(sql, self.engine)
        logger.info("Loaded %d customers for prediction", len(df))
        return df

    # ── WRITE: per-customer prediction results ────────────────────────────────
    def write_predictions(self, predictions_df: pd.DataFrame) -> int:
        """
        Insert rows into ml_predictions.
        predictions_df must have columns:
            customer_id, predicted_next_category, category_confidence,
            top3_categories (list of dicts), recommended_product_id,
            recommended_product_name, rank_score, top5_products (list),
            is_cold_start, cold_start_strategy
        Returns number of rows written.
        """
        rows_written = 0
        with self.engine.begin() as conn:
            for _, row in predictions_df.iterrows():
                top3  = json.dumps(row.get("top3_categories", []))
                top5  = json.dumps(row.get("top5_products", []))
                conn.execute(text("""
                    INSERT INTO ml_predictions (
                        customer_id, model_version,
                        predicted_next_category, category_confidence, top3_categories,
                        recommended_product_id, recommended_product_name,
                        rank_score, top5_products,
                        is_cold_start, cold_start_strategy
                    ) VALUES (
                        :cid, :mv,
                        :cat, :conf, CAST(:top3 AS jsonb),
                        :pid, :pname,
                        :rs, CAST(:top5 AS jsonb),
                        :cs, :cstrat
                    )
                """), {
                    "cid"   : row["customer_id"],
                    "mv"    : MODEL_VERSION,
                    "cat"   : row.get("predicted_next_category"),
                    "conf"  : row.get("category_confidence"),
                    "top3"  : top3,
                    "pid"   : row.get("recommended_product_id"),
                    "pname" : row.get("recommended_product_name"),
                    "rs"    : row.get("rank_score"),
                    "top5"  : top5,
                    "cs"    : bool(row.get("is_cold_start", False)),
                    "cstrat": row.get("cold_start_strategy"),
                })
                rows_written += 1
        logger.info("Wrote %d predictions to ml_predictions", rows_written)
        return rows_written

    # ── WRITE: customer segments (for campaign dispatch) ──────────────────────
    def write_segments(self, segments_df: pd.DataFrame,
                       model_version: str = MODEL_VERSION) -> int:
        """
        Insert rows into ml_customer_segments.
        segments_df must have columns:
            segment_label, predicted_next_category, recommended_product_name,
            recommended_product_id, customer_count, customer_ids (list),
            avg_confidence, avg_rank_score, avg_product_price
        Returns number of segments written.
        """
        rows_written = 0
        with self.engine.begin() as conn:
            for _, row in segments_df.iterrows():
                cids = list(row.get("customer_ids", []))
                conn.execute(text("""
                    INSERT INTO ml_customer_segments (
                        model_version, segment_label,
                        predicted_category, recommended_product_name,
                        recommended_product_id,
                        customer_count, customer_ids,
                        avg_confidence, avg_rank_score, avg_product_price
                    ) VALUES (
                        :mv, :label,
                        :cat, :pname, :pid,
                        :count, :cids,
                        :conf, :rs, :price
                    )
                """), {
                    "mv"   : model_version,
                    "label": row["segment_label"],
                    "cat"  : row.get("predicted_next_category"),
                    "pname": row.get("recommended_product_name"),
                    "pid"  : row.get("recommended_product_id"),
                    "count": int(row.get("customer_count", len(cids))),
                    "cids" : cids,
                    "conf" : row.get("avg_confidence"),
                    "rs"   : row.get("avg_rank_score"),
                    "price": row.get("avg_product_price"),
                })
                rows_written += 1
        logger.info("Wrote %d segments to ml_customer_segments", rows_written)
        return rows_written

    # ── READ: fetch segment for campaign dispatch ─────────────────────────────
    def get_pending_segments(self) -> pd.DataFrame:
        sql = """
            SELECT segment_id, segment_label, predicted_category,
                   recommended_product_name, customer_count, customer_ids,
                   avg_confidence, created_at
            FROM   ml_customer_segments
            WHERE  campaign_dispatched = FALSE
            ORDER  BY customer_count DESC
        """
        return pd.read_sql(sql, self.engine)

    def mark_segment_dispatched(self, segment_id: int, campaign_id: str):
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE ml_customer_segments
                SET campaign_dispatched = TRUE,
                    campaign_id = :cid,
                    dispatched_at = NOW()
                WHERE segment_id = :sid
            """), {"sid": segment_id, "cid": campaign_id})

    def health_check(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error("DB health check failed: %s", e)
            return False
        


    def insert_event(self, event: dict):
        """
        Raw event log — every event saved as-is.
        Useful for audit trail and reprocessing.
        """
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO customer_events (
                    customer_id, event_type, product_id, quantity,
                    session_id, device, traffic_source, search_keyword, metadata
                ) VALUES (
                    :cid, :etype, :pid, :qty,
                    :sid, :device, :source, :search, CAST(:meta AS jsonb)
                )
            """), {
                "cid"   : event["customer_id"],
                "etype" : event["event_type"],
                "pid"   : event.get("product_id"),
                "qty"   : event.get("quantity", 1),
                "sid"   : event.get("session_id"),
                "device": event.get("device"),
                "source": event.get("traffic_source"),
                "search": event.get("search_keyword"),
                "meta"  : json.dumps(event.get("metadata", {})),
            })
 
    def ensure_customer_exists(self, customer_id: str):
        """
        Insert customer with defaults if not already present.
        Called on every event for new/unknown customer_ids.
        """
        with self.engine.begin() as conn:
            exists = conn.execute(text("""
                SELECT 1 FROM dim_customers WHERE customer_id = :cid
            """), {"cid": customer_id}).fetchone()
 
            if not exists:
                conn.execute(text("""
                    INSERT INTO dim_customers (customer_id, signup_date)
                    VALUES (:cid, CURRENT_DATE)
                    ON CONFLICT (customer_id) DO NOTHING
                """), {"cid": customer_id})
                logger.info("Auto-created customer: %s", customer_id)
 
    def ensure_session_exists(self, customer_id: str, session_id: Optional[str] = None):
        """
        Ensures there is an ACTIVE session (order_id IS NULL) for this customer.
        If none exists, creates a new one.
        Active session = purchase not yet linked to an order.
        """
        with self.engine.begin() as conn:
            active = conn.execute(text("""
                SELECT session_id
                FROM   fact_sessions
                WHERE  customer_id = :cid
                  AND  order_id IS NULL
                ORDER  BY session_date DESC
                LIMIT  1
            """), {"cid": customer_id}).fetchone()
 
            if not active:
                new_sid = session_id or ("S_" + uuid.uuid4().hex[:20])   # fits VARCHAR(30)
                conn.execute(text("""
                    INSERT INTO fact_sessions (
                        session_id, customer_id, session_date,
                        pages_viewed, wishlist_items, cart_items, interaction_score
                    ) VALUES (
                        :sid, :cid, NOW(),
                        0, 0, 0, 0.0
                    )
                    ON CONFLICT (session_id) DO NOTHING
                """), {"sid": new_sid, "cid": customer_id})
                logger.info("New session created for customer %s: %s", customer_id, new_sid)
 
    def update_session_metrics(self, event: dict):
        """
        Increment session counters on the ACTIVE session (order_id IS NULL).
        Interaction score weights (from discussion):
            page_view   → pages_viewed   +1,  score +0.05
            wishlist    → wishlist_items +1,  score +0.10
            add_to_cart → cart_items    +1,  score +0.20
        Score is capped at 1.0.
        """
        cid        = event["customer_id"]
        event_type = event["event_type"]
 
        # Map event_type → column update
        update_map = {
            "page_view"  : ("pages_viewed   = pages_viewed   + 1", 0.05),
            "product_view": ("pages_viewed  = pages_viewed   + 1", 0.05),  # alias
            "wishlist"   : ("wishlist_items = wishlist_items + 1", 0.10),
            "add_to_cart": ("cart_items     = cart_items     + 1", 0.20),
        }
 
        if event_type not in update_map:
            return   # purchase and unknown events — no counter update here
 
        col_update, score_delta = update_map[event_type]
 
        with self.engine.begin() as conn:
            conn.execute(text(f"""
                UPDATE fact_sessions
                SET    {col_update},
                       interaction_score = LEAST(interaction_score + :delta, 1.0)
                WHERE  customer_id = :cid
                  AND  order_id IS NULL
            """), {"cid": cid, "delta": score_delta})
 
    def process_purchase(self, event: dict):
        """
        On purchase event:
          1. Get next purchase_number for this customer
          2. Create fact_orders row
          3. Create fact_order_items row (price from dim_products)
          4. Link active session → new order_id
        """
        cid = event["customer_id"]
        pid = event["product_id"]
        qty = event.get("quantity", 1)
 
        with self.engine.begin() as conn:
 
            # Next sequential purchase number for this customer
            purchase_number = conn.execute(text("""
                SELECT COALESCE(MAX(purchase_number), 0) + 1
                FROM   fact_orders
                WHERE  customer_id = :cid
            """), {"cid": cid}).scalar()
 
            order_id = "O_" + uuid.uuid4().hex[:20]   # fits VARCHAR(30)
 
            # Insert order
            conn.execute(text("""
                INSERT INTO fact_orders (
                    order_id, customer_id, purchase_date,
                    purchase_number, traffic_source
                ) VALUES (
                    :oid, :cid, CURRENT_DATE,
                    :pnum, :tsrc
                )
                ON CONFLICT (order_id) DO NOTHING
            """), {
                "oid" : order_id,
                "cid" : cid,
                "pnum": purchase_number,
                "tsrc": event.get("traffic_source"),
            })
 
            # Get product price from catalog
            price = conn.execute(text("""
                SELECT base_price FROM dim_products WHERE product_id = :pid
            """), {"pid": pid}).scalar()
 
            if price is None:
                logger.warning("Product %s not found in dim_products — using price=0", pid)
                price = 0.0
 
            # Insert order item
            conn.execute(text("""
                INSERT INTO fact_order_items (
                    order_id, product_id, quantity,
                    unit_price, discount_pct
                ) VALUES (
                    :oid, :pid, :qty,
                    :price, :disc
                )
            """), {
                "oid"  : order_id,
                "pid"  : pid,
                "qty"  : qty,
                "price": float(price),
                "disc" : float(event.get("discount_pct", 0.0)),
            })
 
            # Link active session to this order (closes the session)
            conn.execute(text("""
                UPDATE fact_sessions
                SET    order_id    = :oid,
                       session_date = NOW()
                WHERE  customer_id = :cid
                  AND  order_id IS NULL
            """), {"oid": order_id, "cid": cid})
 
        logger.info(
            "Purchase processed: customer=%s order=%s product=%s qty=%d",
            cid, order_id, pid, qty
        )
        return order_id
 
    # ── ORCHESTRATOR ─────────────────────────────────────────────────────────
    def process_event(self, event: dict):
        """
        Main entry point for the SDK event layer.
 
        Flow:
            SDK Event
                ↓
            insert_event()          — raw audit log
                ↓
            ensure_customer_exists()— auto-create if new user
                ↓
            ensure_session_exists() — open a session if none active
                ↓
            update_session_metrics()— increment counters
                ↓
            process_purchase()      — only on purchase event
                ↓
            v_customer_purchase_history (view)
                ↓
            ML Model  (called from app.py /track route)
 
        Example events:
            {"customer_id":"C001","event_type":"page_view","product_id":"P001"}
            {"customer_id":"C001","event_type":"wishlist","product_id":"P001"}
            {"customer_id":"C001","event_type":"add_to_cart","product_id":"P001"}
            {"customer_id":"C001","event_type":"purchase","product_id":"P001","quantity":1}
        """
        self.insert_event(event)
        self.ensure_customer_exists(event["customer_id"])
        self.ensure_session_exists(event["customer_id"])
        self.update_session_metrics(event)
 
        if event.get("event_type") == "purchase":
            self.process_purchase(event)
 