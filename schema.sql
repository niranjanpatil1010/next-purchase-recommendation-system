-- -- ============================================================
-- --  Next Purchase Prediction System — PostgreSQL Schema
-- --  Database: ecommerce_ml
-- --  Run: psql -U postgres -d ecommerce_ml -f schema.sql
-- -- ============================================================

-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -- ─────────────────────────────────────────────────────────────
-- -- DIMENSION TABLES
-- -- ─────────────────────────────────────────────────────────────

-- CREATE TABLE IF NOT EXISTS dim_customers (
--     customer_id     VARCHAR(20)  PRIMARY KEY,
--     age             INT,
--     gender          VARCHAR(10),
--     city            VARCHAR(100),
--     state           VARCHAR(100),
--     signup_date     DATE         NOT NULL,
--     email           VARCHAR(255),
--     phone           VARCHAR(20),
--     created_at      TIMESTAMP    DEFAULT NOW(),
--     updated_at      TIMESTAMP    DEFAULT NOW()
-- );

-- CREATE INDEX idx_customers_city  ON dim_customers(city);
-- CREATE INDEX idx_customers_state ON dim_customers(state);


-- CREATE TABLE IF NOT EXISTS dim_products (
--     product_id          VARCHAR(20)    PRIMARY KEY,
--     product_name        VARCHAR(500)   NOT NULL,
--     brand               VARCHAR(100),
--     category            VARCHAR(100)   NOT NULL,
--     subcategory         VARCHAR(100),
--     base_price          NUMERIC(12,2),
--     product_rating      NUMERIC(3,1),
--     product_popularity  INT            DEFAULT 0,
--     is_active           BOOLEAN        DEFAULT TRUE,
--     created_at          TIMESTAMP      DEFAULT NOW(),
--     updated_at          TIMESTAMP      DEFAULT NOW()
-- );

-- CREATE INDEX idx_products_category    ON dim_products(category);
-- CREATE INDEX idx_products_brand       ON dim_products(brand);
-- CREATE INDEX idx_products_subcategory ON dim_products(subcategory);


-- CREATE TABLE IF NOT EXISTS dim_campaigns (
--     campaign_id     UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
--     campaign_name   VARCHAR(200) NOT NULL,
--     campaign_type   VARCHAR(50),   -- EMAIL | SMS | WHATSAPP | PUSH
--     start_date      DATE,
--     end_date        DATE,
--     description     TEXT,
--     created_at      TIMESTAMP    DEFAULT NOW()
-- );


-- -- ─────────────────────────────────────────────────────────────
-- -- FACT TABLES
-- -- ─────────────────────────────────────────────────────────────

-- CREATE TABLE IF NOT EXISTS fact_orders (
--     order_id        VARCHAR(30)    PRIMARY KEY,
--     customer_id     VARCHAR(20)    NOT NULL REFERENCES dim_customers(customer_id),
--     purchase_date   DATE           NOT NULL,
--     purchase_number INT            NOT NULL,   -- Nth purchase for this customer
--     total_amount    NUMERIC(12,2),
--     coupon_used     BOOLEAN        DEFAULT FALSE,
--     traffic_source  VARCHAR(100),
--     created_at      TIMESTAMP      DEFAULT NOW()
-- );

-- CREATE INDEX idx_orders_customer_id   ON fact_orders(customer_id);
-- CREATE INDEX idx_orders_purchase_date ON fact_orders(purchase_date);


-- CREATE TABLE IF NOT EXISTS fact_order_items (
--     order_item_id   BIGSERIAL      PRIMARY KEY,
--     order_id        VARCHAR(30)    NOT NULL REFERENCES fact_orders(order_id),
--     product_id      VARCHAR(20)    NOT NULL REFERENCES dim_products(product_id),
--     quantity        INT            NOT NULL DEFAULT 1,
--     unit_price      NUMERIC(12,2)  NOT NULL,
--     discount_pct    NUMERIC(5,2)   DEFAULT 0,
--     net_amount      NUMERIC(12,2)  GENERATED ALWAYS AS
--                     (unit_price * quantity * (1 - discount_pct/100)) STORED
-- );

-- CREATE INDEX idx_order_items_order_id   ON fact_order_items(order_id);
-- CREATE INDEX idx_order_items_product_id ON fact_order_items(product_id);


-- CREATE TABLE IF NOT EXISTS fact_sessions (
--     session_id          VARCHAR(30)    PRIMARY KEY,
--     customer_id         VARCHAR(20)    NOT NULL REFERENCES dim_customers(customer_id),
--     order_id            VARCHAR(30)    REFERENCES fact_orders(order_id),
--     session_date        TIMESTAMP,
--     device              VARCHAR(20),
--     traffic_source      VARCHAR(100),
--     search_keyword      VARCHAR(500),
--     pages_viewed        INT            DEFAULT 0,
--     wishlist_items      INT            DEFAULT 0,
--     cart_items          INT            DEFAULT 0,
--     interaction_score   NUMERIC(5,3)
-- );

-- CREATE INDEX idx_sessions_customer_id ON fact_sessions(customer_id);
-- CREATE INDEX idx_sessions_order_id    ON fact_sessions(order_id);


-- CREATE TABLE IF NOT EXISTS fact_campaign_interactions (
--     interaction_id  BIGSERIAL   PRIMARY KEY,
--     customer_id     VARCHAR(20) NOT NULL REFERENCES dim_customers(customer_id),
--     campaign_id     UUID        NOT NULL REFERENCES dim_campaigns(campaign_id),
--     sent_at         TIMESTAMP,
--     opened_at       TIMESTAMP,
--     clicked_at      TIMESTAMP,
--     coupon_code     VARCHAR(50),
--     coupon_used     BOOLEAN     DEFAULT FALSE,
--     created_at      TIMESTAMP   DEFAULT NOW()
-- );

-- CREATE INDEX idx_camp_customer ON fact_campaign_interactions(customer_id);
-- CREATE INDEX idx_camp_id       ON fact_campaign_interactions(campaign_id);


-- -- ─────────────────────────────────────────────────────────────
-- -- ML OUTPUT TABLES  (written by prediction pipeline)
-- -- ─────────────────────────────────────────────────────────────

-- CREATE TABLE IF NOT EXISTS ml_predictions (
--     prediction_id         BIGSERIAL    PRIMARY KEY,
--     customer_id           VARCHAR(20)  NOT NULL REFERENCES dim_customers(customer_id),
--     predicted_at          TIMESTAMP    DEFAULT NOW(),
--     model_version         VARCHAR(50),
--     -- Stage 1 output
--     predicted_next_category  VARCHAR(100),
--     category_confidence      NUMERIC(6,4),
--     top3_categories          JSONB,      -- [{"category":"Electronics","prob":0.72}, ...]
--     -- Stage 2 output
--     recommended_product_id   VARCHAR(20) REFERENCES dim_products(product_id),
--     recommended_product_name VARCHAR(500),
--     rank_score               NUMERIC(8,6),
--     top5_products            JSONB,      -- [{product_id, product_name, rank_score}, ...]
--     -- cold start flag
--     is_cold_start            BOOLEAN    DEFAULT FALSE,
--     cold_start_strategy      VARCHAR(50) -- 'popularity' | 'demographic' | null
-- );

-- CREATE INDEX idx_pred_customer_id  ON ml_predictions(customer_id);
-- CREATE INDEX idx_pred_predicted_at ON ml_predictions(predicted_at);
-- CREATE INDEX idx_pred_category     ON ml_predictions(predicted_next_category);

-- -- Latest prediction per customer (used by campaign system)
-- CREATE OR REPLACE VIEW v_latest_predictions AS
-- SELECT DISTINCT ON (customer_id)
--     prediction_id, customer_id, predicted_at, model_version,
--     predicted_next_category, category_confidence,
--     recommended_product_id, recommended_product_name, rank_score,
--     top5_products, is_cold_start
-- FROM ml_predictions
-- ORDER BY customer_id, predicted_at DESC;


-- CREATE TABLE IF NOT EXISTS ml_customer_segments (
--     segment_id          BIGSERIAL     PRIMARY KEY,
--     created_at          TIMESTAMP     DEFAULT NOW(),
--     model_version       VARCHAR(50),
--     -- Segment definition
--     segment_label       VARCHAR(600)  NOT NULL,  -- "Electronics > Samsung Mobile Zoom"
--     predicted_category  VARCHAR(100),
--     recommended_product_name VARCHAR(500),
--     recommended_product_id   VARCHAR(20) REFERENCES dim_products(product_id),
--     -- Segment stats
--     customer_count      INT,
--     avg_confidence      NUMERIC(6,4),
--     avg_rank_score      NUMERIC(8,6),
--     avg_product_price   NUMERIC(12,2),
--     -- Customer list (array of customer_ids)
--     customer_ids        TEXT[]        NOT NULL,
--     -- Campaign integration
--     campaign_id         UUID          REFERENCES dim_campaigns(campaign_id),
--     campaign_dispatched BOOLEAN       DEFAULT FALSE,
--     dispatched_at       TIMESTAMP
-- );

-- CREATE INDEX idx_segments_created_at ON ml_customer_segments(created_at);
-- CREATE INDEX idx_segments_category   ON ml_customer_segments(predicted_category);
-- CREATE INDEX idx_segments_product    ON ml_customer_segments(recommended_product_name);
-- CREATE INDEX idx_segments_campaign   ON ml_customer_segments(campaign_id);


-- -- ─────────────────────────────────────────────────────────────
-- -- FEATURE STORE VIEWS  (model reads these, not raw tables)
-- -- ─────────────────────────────────────────────────────────────

-- -- Customer purchase history flat view (what the model consumes per customer)
-- CREATE OR REPLACE VIEW v_customer_purchase_history AS
-- SELECT
--     c.customer_id,
--     c.age,
--     c.gender,
--     c.city,
--     c.state,
--     c.signup_date,
--     o.order_id,
--     o.purchase_date,
--     o.purchase_number,
--     p.product_id,
--     p.product_name,
--     p.brand,
--     p.category,
--     p.subcategory,
--     i.unit_price          AS price,
--     i.discount_pct        AS discount,
--     i.quantity,
--     i.net_amount,
--     p.product_rating,
--     p.product_popularity,
--     s.device,
--     s.traffic_source,
--     s.search_keyword,
--     s.pages_viewed,
--     s.wishlist_items,
--     s.cart_items,
--     s.interaction_score
-- FROM dim_customers        c
-- JOIN fact_orders          o  ON c.customer_id = o.customer_id
-- JOIN fact_order_items     i  ON o.order_id    = i.order_id
-- JOIN dim_products         p  ON i.product_id  = p.product_id
-- LEFT JOIN fact_sessions   s  ON o.order_id    = s.order_id;


-- -- Customer RFM summary (refreshed daily in production)
-- CREATE OR REPLACE VIEW v_customer_rfm AS
-- SELECT
--     c.customer_id,
--     c.age, c.gender, c.city, c.state, c.signup_date,
--     COUNT(DISTINCT o.order_id)                  AS total_orders,
--     SUM(i.net_amount)                           AS total_spent,
--     AVG(i.net_amount)                           AS avg_order_value,
--     MAX(o.purchase_date)                        AS last_purchase_date,
--     CURRENT_DATE - MAX(o.purchase_date)         AS recency_days,
--     COUNT(DISTINCT o.order_id)                  AS frequency,
--     SUM(i.net_amount)                           AS monetary,
--     MODE() WITHIN GROUP (ORDER BY p.brand)      AS favorite_brand,
--     MODE() WITHIN GROUP (ORDER BY p.category)   AS favorite_category
-- FROM dim_customers        c
-- JOIN fact_orders          o  ON c.customer_id = o.customer_id
-- JOIN fact_order_items     i  ON o.order_id    = i.order_id
-- JOIN dim_products         p  ON i.product_id  = p.product_id
-- GROUP BY c.customer_id, c.age, c.gender, c.city, c.state, c.signup_date;


-- -- Product global stats (used in Stage 2 ranking)
-- CREATE OR REPLACE VIEW v_product_stats AS
-- SELECT
--     p.product_id,
--     p.product_name,
--     p.brand,
--     p.category,
--     p.subcategory,
--     p.base_price             AS price,
--     AVG(p.product_rating)    AS prod_global_avg_rating,
--     MAX(p.product_popularity)AS prod_global_popularity,
--     COUNT(DISTINCT oi.order_id)  AS prod_total_orders,
--     AVG(oi.discount_pct)     AS prod_avg_discount_given,
--     AVG(oi.quantity)         AS prod_avg_qty_sold
-- FROM dim_products         p
-- LEFT JOIN fact_order_items oi ON p.product_id = oi.product_id
-- GROUP BY p.product_id, p.product_name, p.brand, p.category,
--          p.subcategory, p.base_price;


-- -- ─────────────────────────────────────────────────────────────
-- -- HELPER FUNCTION: write prediction batch from Python
-- -- ─────────────────────────────────────────────────────────────

-- -- Called after ML pipeline writes results
-- CREATE OR REPLACE FUNCTION fn_upsert_prediction(
--     p_customer_id         VARCHAR,
--     p_model_version       VARCHAR,
--     p_pred_category       VARCHAR,
--     p_cat_confidence      NUMERIC,
--     p_top3_categories     JSONB,
--     p_rec_product_id      VARCHAR,
--     p_rec_product_name    VARCHAR,
--     p_rank_score          NUMERIC,
--     p_top5_products       JSONB,
--     p_is_cold_start       BOOLEAN DEFAULT FALSE,
--     p_cold_start_strategy VARCHAR DEFAULT NULL
-- ) RETURNS VOID AS $$
-- BEGIN
--     INSERT INTO ml_predictions (
--         customer_id, model_version,
--         predicted_next_category, category_confidence, top3_categories,
--         recommended_product_id, recommended_product_name, rank_score, top5_products,
--         is_cold_start, cold_start_strategy
--     ) VALUES (
--         p_customer_id, p_model_version,
--         p_pred_category, p_cat_confidence, p_top3_categories,
--         p_rec_product_id, p_rec_product_name, p_rank_score, p_top5_products,
--         p_is_cold_start, p_cold_start_strategy
--     );
-- END;
-- $$ LANGUAGE plpgsql;


-- -- ─────────────────────────────────────────────────────────────
-- -- SEED: example campaign rows for integration
-- -- ─────────────────────────────────────────────────────────────
-- INSERT INTO dim_campaigns (campaign_name, campaign_type, description)
-- VALUES
--   ('Next Purchase - Electronics', 'EMAIL',    'Target customers predicted to buy Electronics next'),
--   ('Next Purchase - Fashion',     'WHATSAPP', 'Target customers predicted to buy Fashion next'),
--   ('Next Purchase - Home',        'EMAIL',    'Target customers predicted to buy Home products next'),
--   ('Next Purchase - Beauty',      'SMS',      'Target customers predicted to buy Beauty products next'),
--   ('Next Purchase - Grocery',     'PUSH',     'Target customers predicted to buy Grocery next'),
--   ('Next Purchase - Sports',      'EMAIL',    'Target customers predicted to buy Sports products next'),
--   ('Next Purchase - Books',       'EMAIL',    'Target customers predicted to buy Books next')
-- ON CONFLICT DO NOTHING;







-- ============================================================
--  Next Purchase Prediction System — PostgreSQL Schema
--  Database: ecommerce_ml
--  Run: psql -U postgres -d ecommerce_ml -f schema.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────────────────────────
-- DIMENSION TABLES
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_customers (
    customer_id     VARCHAR(20)  PRIMARY KEY,
    age             INT,
    gender          VARCHAR(10),
    city            VARCHAR(100),
    state           VARCHAR(100),
    signup_date     DATE         NOT NULL,
    email           VARCHAR(255),
    phone           VARCHAR(20),
    created_at      TIMESTAMP    DEFAULT NOW(),
    updated_at      TIMESTAMP    DEFAULT NOW()
);

CREATE INDEX idx_customers_city  ON dim_customers(city);
CREATE INDEX idx_customers_state ON dim_customers(state);


CREATE TABLE IF NOT EXISTS dim_products (
    product_id          VARCHAR(20)    PRIMARY KEY,
    product_name        VARCHAR(500)   NOT NULL,
    brand               VARCHAR(100),
    category            VARCHAR(100)   NOT NULL,
    subcategory         VARCHAR(100),
    base_price          NUMERIC(12,2),
    product_rating      NUMERIC(3,1),
    product_popularity  INT            DEFAULT 0,
    is_active           BOOLEAN        DEFAULT TRUE,
    created_at          TIMESTAMP      DEFAULT NOW(),
    updated_at          TIMESTAMP      DEFAULT NOW()
);

CREATE INDEX idx_products_category    ON dim_products(category);
CREATE INDEX idx_products_brand       ON dim_products(brand);
CREATE INDEX idx_products_subcategory ON dim_products(subcategory);


CREATE TABLE IF NOT EXISTS dim_campaigns (
    campaign_id     UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_name   VARCHAR(200) NOT NULL,
    campaign_type   VARCHAR(50),   -- EMAIL | SMS | WHATSAPP | PUSH
    start_date      DATE,
    end_date        DATE,
    description     TEXT,
    created_at      TIMESTAMP    DEFAULT NOW()
);


-- ─────────────────────────────────────────────────────────────
-- FACT TABLES
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_orders (
    order_id        VARCHAR(30)    PRIMARY KEY,
    customer_id     VARCHAR(20)    NOT NULL REFERENCES dim_customers(customer_id),
    purchase_date   DATE           NOT NULL,
    purchase_number INT            NOT NULL,   -- Nth purchase for this customer
    total_amount    NUMERIC(12,2),
    coupon_used     BOOLEAN        DEFAULT FALSE,
    traffic_source  VARCHAR(100),
    created_at      TIMESTAMP      DEFAULT NOW()
);

CREATE INDEX idx_orders_customer_id   ON fact_orders(customer_id);
CREATE INDEX idx_orders_purchase_date ON fact_orders(purchase_date);


CREATE TABLE IF NOT EXISTS fact_order_items (
    order_item_id   BIGSERIAL      PRIMARY KEY,
    order_id        VARCHAR(30)    NOT NULL REFERENCES fact_orders(order_id),
    product_id      VARCHAR(20)    NOT NULL REFERENCES dim_products(product_id),
    quantity        INT            NOT NULL DEFAULT 1,
    unit_price      NUMERIC(12,2)  NOT NULL,
    discount_pct    NUMERIC(5,2)   DEFAULT 0,
    net_amount      NUMERIC(12,2)  GENERATED ALWAYS AS
                    (unit_price * quantity * (1 - discount_pct/100)) STORED
);

CREATE INDEX idx_order_items_order_id   ON fact_order_items(order_id);
CREATE INDEX idx_order_items_product_id ON fact_order_items(product_id);


CREATE TABLE IF NOT EXISTS fact_sessions (
    session_id          VARCHAR(30)    PRIMARY KEY,
    customer_id         VARCHAR(20)    NOT NULL REFERENCES dim_customers(customer_id),
    order_id            VARCHAR(30)    REFERENCES fact_orders(order_id),
    session_date        TIMESTAMP,
    device              VARCHAR(20),
    traffic_source      VARCHAR(100),
    search_keyword      VARCHAR(500),
    pages_viewed        INT            DEFAULT 0,
    wishlist_items      INT            DEFAULT 0,
    cart_items          INT            DEFAULT 0,
    interaction_score   NUMERIC(5,3)
);

CREATE INDEX idx_sessions_customer_id ON fact_sessions(customer_id);
CREATE INDEX idx_sessions_order_id    ON fact_sessions(order_id);


CREATE TABLE IF NOT EXISTS fact_campaign_interactions (
    interaction_id  BIGSERIAL   PRIMARY KEY,
    customer_id     VARCHAR(20) NOT NULL REFERENCES dim_customers(customer_id),
    campaign_id     UUID        NOT NULL REFERENCES dim_campaigns(campaign_id),
    sent_at         TIMESTAMP,
    opened_at       TIMESTAMP,
    clicked_at      TIMESTAMP,
    coupon_code     VARCHAR(50),
    coupon_used     BOOLEAN     DEFAULT FALSE,
    created_at      TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX idx_camp_customer ON fact_campaign_interactions(customer_id);
CREATE INDEX idx_camp_id       ON fact_campaign_interactions(campaign_id);


-- ─────────────────────────────────────────────────────────────
-- RAW EVENT LOG  (written by every /track call — audit trail)
-- No FK on customer_id/product_id on purpose: insert_event() runs
-- BEFORE ensure_customer_exists(), so a brand-new customer_id would
-- fail a strict FK on their very first event. Kept as indexed only.
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS customer_events (
    event_id        BIGSERIAL      PRIMARY KEY,
    customer_id     VARCHAR(20)    NOT NULL,
    event_type      VARCHAR(30)    NOT NULL,   -- product_view | search | wishlist | add_to_cart | purchase
    product_id      VARCHAR(20),
    quantity        INT            DEFAULT 1,
    session_id      VARCHAR(30),
    device          VARCHAR(20),
    traffic_source  VARCHAR(100),
    search_keyword  VARCHAR(500),
    metadata        JSONB          DEFAULT '{}'::jsonb,
    created_at      TIMESTAMP      DEFAULT NOW()
);

CREATE INDEX idx_events_customer_id ON customer_events(customer_id);
CREATE INDEX idx_events_event_type  ON customer_events(event_type);
CREATE INDEX idx_events_created_at ON customer_events(created_at);


-- ─────────────────────────────────────────────────────────────
-- ML OUTPUT TABLES  (written by prediction pipeline)
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_predictions (
    prediction_id         BIGSERIAL    PRIMARY KEY,
    customer_id           VARCHAR(20)  NOT NULL REFERENCES dim_customers(customer_id),
    predicted_at          TIMESTAMP    DEFAULT NOW(),
    model_version         VARCHAR(50),
    -- Stage 1 output
    predicted_next_category  VARCHAR(100),
    category_confidence      NUMERIC(6,4),
    top3_categories          JSONB,      -- [{"category":"Electronics","prob":0.72}, ...]
    -- Stage 2 output
    recommended_product_id   VARCHAR(20) REFERENCES dim_products(product_id),
    recommended_product_name VARCHAR(500),
    rank_score               NUMERIC(8,6),
    top5_products            JSONB,      -- [{product_id, product_name, rank_score}, ...]
    -- cold start flag
    is_cold_start            BOOLEAN    DEFAULT FALSE,
    cold_start_strategy      VARCHAR(50) -- 'popularity' | 'demographic' | null
);

CREATE INDEX idx_pred_customer_id  ON ml_predictions(customer_id);
CREATE INDEX idx_pred_predicted_at ON ml_predictions(predicted_at);
CREATE INDEX idx_pred_category     ON ml_predictions(predicted_next_category);

-- Latest prediction per customer (used by campaign system)
CREATE OR REPLACE VIEW v_latest_predictions AS
SELECT DISTINCT ON (customer_id)
    prediction_id, customer_id, predicted_at, model_version,
    predicted_next_category, category_confidence,
    recommended_product_id, recommended_product_name, rank_score,
    top5_products, is_cold_start
FROM ml_predictions
ORDER BY customer_id, predicted_at DESC;


CREATE TABLE IF NOT EXISTS ml_customer_segments (
    segment_id          BIGSERIAL     PRIMARY KEY,
    created_at          TIMESTAMP     DEFAULT NOW(),
    model_version       VARCHAR(50),
    -- Segment definition
    segment_label       VARCHAR(600)  NOT NULL,  -- "Electronics > Samsung Mobile Zoom"
    predicted_category  VARCHAR(100),
    recommended_product_name VARCHAR(500),
    recommended_product_id   VARCHAR(20) REFERENCES dim_products(product_id),
    -- Segment stats
    customer_count      INT,
    avg_confidence      NUMERIC(6,4),
    avg_rank_score       NUMERIC(8,6),
    avg_product_price   NUMERIC(12,2),
    -- Customer list (array of customer_ids)
    customer_ids        TEXT[]        NOT NULL,
    -- Campaign integration
    campaign_id         UUID          REFERENCES dim_campaigns(campaign_id),
    campaign_dispatched BOOLEAN       DEFAULT FALSE,
    dispatched_at       TIMESTAMP
);

CREATE INDEX idx_segments_created_at ON ml_customer_segments(created_at);
CREATE INDEX idx_segments_category   ON ml_customer_segments(predicted_category);
CREATE INDEX idx_segments_product    ON ml_customer_segments(recommended_product_name);
CREATE INDEX idx_segments_campaign   ON ml_customer_segments(campaign_id);


-- ─────────────────────────────────────────────────────────────
-- FEATURE STORE VIEWS  (model reads these, not raw tables)
-- ─────────────────────────────────────────────────────────────

-- Customer purchase history flat view (what the model consumes per customer)
CREATE OR REPLACE VIEW v_customer_purchase_history AS
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
    i.unit_price          AS price,
    i.discount_pct        AS discount,
    i.quantity,
    i.net_amount,
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
JOIN fact_order_items     i  ON o.order_id    = i.order_id
JOIN dim_products         p  ON i.product_id  = p.product_id
LEFT JOIN fact_sessions   s  ON o.order_id    = s.order_id;


-- Customer RFM summary (refreshed daily in production)
CREATE OR REPLACE VIEW v_customer_rfm AS
SELECT
    c.customer_id,
    c.age, c.gender, c.city, c.state, c.signup_date,
    COUNT(DISTINCT o.order_id)                  AS total_orders,
    SUM(i.net_amount)                           AS total_spent,
    AVG(i.net_amount)                           AS avg_order_value,
    MAX(o.purchase_date)                        AS last_purchase_date,
    CURRENT_DATE - MAX(o.purchase_date)         AS recency_days,
    COUNT(DISTINCT o.order_id)                  AS frequency,
    SUM(i.net_amount)                           AS monetary,
    MODE() WITHIN GROUP (ORDER BY p.brand)      AS favorite_brand,
    MODE() WITHIN GROUP (ORDER BY p.category)   AS favorite_category
FROM dim_customers        c
JOIN fact_orders          o  ON c.customer_id = o.customer_id
JOIN fact_order_items     i  ON o.order_id    = i.order_id
JOIN dim_products         p  ON i.product_id  = p.product_id
GROUP BY c.customer_id, c.age, c.gender, c.city, c.state, c.signup_date;


-- Product global stats (used in Stage 2 ranking)
CREATE OR REPLACE VIEW v_product_stats AS
SELECT
    p.product_id,
    p.product_name,
    p.brand,
    p.category,
    p.subcategory,
    p.base_price             AS price,
    AVG(p.product_rating)    AS prod_global_avg_rating,
    MAX(p.product_popularity)AS prod_global_popularity,
    COUNT(DISTINCT oi.order_id)  AS prod_total_orders,
    AVG(oi.discount_pct)     AS prod_avg_discount_given,
    AVG(oi.quantity)         AS prod_avg_qty_sold
FROM dim_products         p
LEFT JOIN fact_order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name, p.brand, p.category,
         p.subcategory, p.base_price;


-- ─────────────────────────────────────────────────────────────
-- HELPER FUNCTION: write prediction batch from Python
-- ─────────────────────────────────────────────────────────────

-- Called after ML pipeline writes results
CREATE OR REPLACE FUNCTION fn_upsert_prediction(
    p_customer_id         VARCHAR,
    p_model_version       VARCHAR,
    p_pred_category       VARCHAR,
    p_cat_confidence      NUMERIC,
    p_top3_categories     JSONB,
    p_rec_product_id      VARCHAR,
    p_rec_product_name    VARCHAR,
    p_rank_score          NUMERIC,
    p_top5_products       JSONB,
    p_is_cold_start       BOOLEAN DEFAULT FALSE,
    p_cold_start_strategy VARCHAR DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    INSERT INTO ml_predictions (
        customer_id, model_version,
        predicted_next_category, category_confidence, top3_categories,
        recommended_product_id, recommended_product_name, rank_score, top5_products,
        is_cold_start, cold_start_strategy
    ) VALUES (
        p_customer_id, p_model_version,
        p_pred_category, p_cat_confidence, p_top3_categories,
        p_rec_product_id, p_rec_product_name, p_rank_score, p_top5_products,
        p_is_cold_start, p_cold_start_strategy
    );
END;
$$ LANGUAGE plpgsql;


-- ─────────────────────────────────────────────────────────────
-- SEED: example campaign rows for integration
-- ─────────────────────────────────────────────────────────────
INSERT INTO dim_campaigns (campaign_name, campaign_type, description)
VALUES
  ('Next Purchase - Electronics', 'EMAIL',    'Target customers predicted to buy Electronics next'),
  ('Next Purchase - Fashion',     'WHATSAPP', 'Target customers predicted to buy Fashion next'),
  ('Next Purchase - Home',        'EMAIL',    'Target customers predicted to buy Home products next'),
  ('Next Purchase - Beauty',      'SMS',      'Target customers predicted to buy Beauty products next'),
  ('Next Purchase - Grocery',     'PUSH',     'Target customers predicted to buy Grocery next'),
  ('Next Purchase - Sports',      'EMAIL',    'Target customers predicted to buy Sports products next'),
  ('Next Purchase - Books',       'EMAIL',    'Target customers predicted to buy Books next')
ON CONFLICT DO NOTHING;