"""
seed_test_users.py
────────────────────────────────────────────────────────────────
Creates 10 test customers with purchase history.

3 customers  -> 1 purchase (cold start)
7 customers  -> 2-8 purchases (ML pipeline testing)

Run:
    python seed_test_users.py
"""

import random
import uuid
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import text

from db import get_engine
from predict_new_user import register_customer, record_purchase


engine = get_engine()


CITIES = [
    ("Mumbai", "Maharashtra"),
    ("Delhi", "Delhi"),
    ("Bengaluru", "Karnataka"),
    ("Pune", "Maharashtra"),
    ("Hyderabad", "Telangana"),
    ("Chennai", "Tamil Nadu")
]

GENDERS = [
    "Male",
    "Female",
    "Other"
]

DEVICES = [
    "Mobile",
    "Desktop",
    "Tablet"
]

TRAFFIC_SOURCES = [
    "Organic Search",
    "Paid Ads",
    "Social Media",
    "Direct",
    "Email"
]


# --------------------------------------------------
# Unique customer id
# --------------------------------------------------

def random_customer_id():

    return f"C_TEST_{uuid.uuid4().hex[:8].upper()}"



# --------------------------------------------------
# Load products
# Matches dim_products schema
# --------------------------------------------------

def get_real_products(limit=100):

    query = """
        SELECT
            product_id,
            base_price AS price
        FROM dim_products
        WHERE is_active = TRUE
        ORDER BY random()
        LIMIT :limit
    """

    df = pd.read_sql(
        text(query),
        engine,
        params={
            "limit": limit
        }
    )


    if df.empty:
        raise Exception(
            "dim_products empty. Insert products first."
        )


    df["product_id"] = df["product_id"].astype(str)

    df["price"] = df["price"].astype(float)


    return df



# --------------------------------------------------
# Create customer + history
# --------------------------------------------------

def seed_customer(products_df, n_purchases):


    customer_id = random_customer_id()


    age = random.randint(
        18,
        60
    )

    gender = random.choice(
        GENDERS
    )


    city, state = random.choice(
        CITIES
    )


    signup_date = (
        datetime.now()
        -
        timedelta(
            days=random.randint(
                30,
                400
            )
        )
    ).date()



    # Insert customer

    register_customer(
        engine,
        customer_id,
        age,
        gender,
        city,
        state,
        signup_date
    )



    purchase_date = (
        datetime.now()
        -
        timedelta(
            days=random.randint(
                60,
                300
            )
        )
    )


    # Insert purchases

    for i in range(1, n_purchases + 1):


        product = products_df.sample(
            1
        ).iloc[0]


        order_id = (
            f"O_{customer_id}_{i}"
        )


        session_id = (
            f"S_{customer_id}_{i}"
        )


        purchase_date += timedelta(
            days=random.randint(
                5,
                45
            )
        )



        record_purchase(

            engine,

            order_id=order_id,

            customer_id=customer_id,

            product_id=str(
                product["product_id"]
            ),

            purchase_number=i,

            unit_price=float(
                product["price"]
            ),

            discount_pct=round(
                random.uniform(
                    0,
                    20
                ),
                1
            ),

            quantity=random.randint(
                1,
                3
            ),


            session_id=session_id,


            device=random.choice(
                DEVICES
            ),


            traffic_source=random.choice(
                TRAFFIC_SOURCES
            ),


            search_keyword="Unknown",


            pages_viewed=random.randint(
                2,
                15
            ),


            wishlist_items=random.randint(
                0,
                4
            ),


            cart_items=random.randint(
                1,
                5
            ),


            interaction_score=round(
                random.uniform(
                    0.1,
                    1.0
                ),
                3
            ),


            # DATE column
            purchase_date=purchase_date.date()

        )


    return customer_id, n_purchases



# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":


    try:

        products_df = get_real_products(
            100
        )


        print(
            f"Loaded {len(products_df)} products\n"
        )


        created = []



        # Cold start users

        for _ in range(3):

            cid, n = seed_customer(
                products_df,
                1
            )

            created.append(
                (cid,n)
            )



        # Warm users

        for _ in range(7):

            n = random.randint(
                2,
                8
            )


            cid, n = seed_customer(
                products_df,
                n
            )


            created.append(
                (cid,n)
            )



        print("="*60)

        print(
            "10 TEST CUSTOMERS CREATED"
        )

        print("="*60)


        for cid,n in created:

            tag = (
                "COLD START"
                if n == 1
                else "ML PIPELINE"
            )


            print(
                f"{cid:<20} "
                f"{n} purchases "
                f"-> {tag}"
            )



        print("\nTest API:")

        print(
            """
POST /predict

{
    "customer_id":"YOUR_CUSTOMER_ID"
}
"""
        )


    except Exception as e:

        print("\nERROR:")
        print(e)

        import traceback
        traceback.print_exc()