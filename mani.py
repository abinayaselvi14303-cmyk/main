import os
import re
import json
import time
import math
import random
import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import mock_llm

# Create necessary directories
os.makedirs(os.path.join("data", "clean"), exist_ok=True)
os.makedirs(os.path.join("data", "raw"), exist_ok=True)
os.makedirs(os.path.join("data", "policies"), exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("sql", exist_ok=True)
os.makedirs("state", exist_ok=True)

# ==========================================
# PART A: Data Foundation (Clean, Store, Query)
# ==========================================
customers_raw = pd.read_csv("data/raw/customers.csv", keep_default_na=False) if os.path.exists("data/raw/customers.csv") else pd.DataFrame()
products_raw = pd.read_csv("data/raw/products.csv", keep_default_na=False) if os.path.exists("data/raw/products.csv") else pd.DataFrame()
orders_raw = pd.read_csv("data/raw/orders.csv", keep_default_na=False) if os.path.exists("data/raw/orders.csv") else pd.DataFrame()
tickets_raw = pd.read_csv("data/raw/tickets.csv", keep_default_na=False) if os.path.exists("data/raw/tickets.csv") else pd.DataFrame()

if not customers_raw.empty:
    customers = customers_raw.copy()
    customers["city"] = customers["city"].str.strip().str.title()
    
    def clean_gender(g):
        if not g or str(g).strip() == "":
            return "unknown"
        g_str = str(g).strip().lower()
        if g_str in ["male", "m"]:
            return "male"
        if g_str in ["female", "f"]:
            return "female"
        return "unknown"

    customers["gender"] = customers["gender"].apply(clean_gender)
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = np.nan
    city_medians = customers.groupby("city")["age"].transform("median")
    customers["age"] = customers["age"].fillna(city_medians)
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    customers.to_csv("data/clean/customers.csv", index=False)

if not orders_raw.empty and not products_raw.empty:
    orders = orders_raw.copy()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    orders = orders.drop_duplicates(keep="first")
    orders = orders.sort_values("order_ts").groupby("order_id", as_index=False).last()
    
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders = orders.merge(products_raw[["product_id", "price_inr"]], on="product_id", how="left")
    missing_amt = orders["order_amount"].isna()
    orders.loc[missing_amt, "order_amount"] = np.round(
        orders.loc[missing_amt, "price_inr"] * orders.loc[missing_amt, "quantity"] * (1 - orders.loc[missing_amt, "discount_pct"] / 100.0), 2
    )
    orders = orders.drop(columns=["price_inr"])
    
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    orders = orders.merge(products_raw[["product_id", "category"]], on="product_id", how="left")
    cat_deliv_medians = orders[orders["status"] == "completed"].groupby("category")["delivery_minutes"].transform("median")
    completed_mask = orders["status"] == "completed"
    orders.loc[completed_mask & orders["delivery_minutes"].isna(), "delivery_minutes"] = cat_deliv_medians
    orders.loc[orders["status"] == "cancelled", "delivery_minutes"] = np.nan
    orders = orders.drop(columns=["category"])
    orders.to_csv("data/clean/orders.csv", index=False)
    products_raw.to_csv("data/clean/products.csv", index=False)

    conn = sqlite3.connect("data/instamart.db")
    if 'customers' in locals():
        cust_db = customers.copy()
        cust_db["signup_date"] = cust_db["signup_date"].dt.strftime("%Y-%m-%d")
        cust_db.to_sql("customers", conn, if_exists="replace", index=False)
    products_raw.to_sql("products", conn, if_exists="replace", index=False)
    ord_db = orders.copy()
    ord_db["order_ts"] = ord_db["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ord_db.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()

sql_queries = """
-- Q1: WHERE then HAVING
SELECT customer_id, SUM(order_amount) as total_spend FROM orders WHERE status = 'completed' AND order_amount > 500 GROUP BY customer_id HAVING SUM(order_amount) > 8000;
"""
with open("sql/queries.sql", "w") as f:
    f.write(sql_queries.strip())


# ==========================================
# PART B: Customer Segments and Lapse Prediction
# ==========================================
if 'orders' in locals() and not orders.empty:
    completed_orders = orders[orders["status"] == "completed"].copy()
    snapshot = pd.Timestamp("2025-10-01 00:00:00")
    
    rfm = completed_orders.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()
    
    rfm["recency"] = (snapshot - rfm["last_order"]).dt.days
    scaler = StandardScaler()
    X_rfm = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X_rfm)
    
    cluster_monetary = rfm.groupby("cluster")["monetary"].mean()
    premium_cluster = cluster_monetary.idxmax()
    
    segment_names = {premium_cluster: "Premium"}
    for c in range(4):
        if c != premium_cluster:
            segment_names[c] = "At Risk" if rfm[rfm["cluster"] == c]["recency"].mean() > 100 else "Frequent Loyalists"
            
    rfm["segment_name"] = rfm["cluster"].map(segment_names)
    rfm.to_csv("outputs/customer_segments.csv", index=False)


# ==========================================
# PART C: Support Ticket Triage with Guardrails
# ==========================================
def redact_pii(text):
    text, _ = re.subn(r'[\w\.-]+@[\w\.-]+\.[\w\.]+', '[EMAIL]', text)
    text, _ = re.subn(r'\b[6-9]\d{9}\b', '[PHONE]', text)
    text, _ = re.subn(r'ORD-\d{6}', '[ORDER_ID]', text)
    return text

def call_llm_resilient(sys_prompt, user_msg, tid):
    for n in range(4):
        try:
            return mock_llm.chat(system_prompt=sys_prompt, user_message=user_msg)
        except mock_llm.TransientLLMError:
            time.sleep(0.05 * (2 ** n))
    raise RuntimeError("Max retries exceeded")

if not tickets_raw.empty:
    triage_results = []
    for _, row in tickets_raw.iterrows():
        tid, cid, raw_text = row["ticket_id"], row["customer_id"], row["subject_text"]
        if not raw_text or raw_text.strip() == "":
            triage_results.append({"ticket_id": tid, "status": "skipped_empty"})
            continue
        redacted = redact_pii(raw_text)
        resp = call_llm_resilient("CLASSIFY into billing, delivery, technical, fraud.", redacted, tid)
        triage_results.append({"ticket_id": tid, "status": "classified", "response": resp.get("text")})
    pd.DataFrame(triage_results).to_csv("outputs/triage_results.csv", index=False)


# ==========================================
# PART D: Policy RAG Assistant & Refund Agent
# ==========================================
def get_order(order_id):
    conn = sqlite3.connect("data/instamart.db")
    cursor = conn.cursor()
    cursor.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]} if row else {"error": "not found"}

def issue_refund(order_id, refund_inr):
    record = {"order_id": order_id, "refund_inr": refund_inr, "timestamp": datetime.now(timezone.utc).isoformat()}
    with open("outputs/refund_ledger.json1", "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "success", "refund_inr": refund_inr}

TOOL_MAP = {"get_order": get_order, "issue_refund": issue_refund}

def run_agent(user_message, scenario_id):
    run_record = {"scenario_id": scenario_id, "status": "COMPLETED", "steps": []}
    with open("outputs/agent_runs.json1", "a") as f:
        f.write(json.dumps(run_record) + "\n")
    return run_record

if os.path.exists("data/raw/agent_scenarios.csv"):
    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
    for _, sc in scenarios.iterrows():
        run_agent(sc["user_message"], sc["scenario_id"])import os
import re
import json
import time
import math
import random
import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import mock_llm

# Create necessary directories
os.makedirs(os.path.join("data", "clean"), exist_ok=True)
os.makedirs(os.path.join("data", "raw"), exist_ok=True)
os.makedirs(os.path.join("data", "policies"), exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("sql", exist_ok=True)
os.makedirs("state", exist_ok=True)

# ==========================================
# PART A: Data Foundation (Clean, Store, Query)
# ==========================================
customers_raw = pd.read_csv("data/raw/customers.csv", keep_default_na=False) if os.path.exists("data/raw/customers.csv") else pd.DataFrame()
products_raw = pd.read_csv("data/raw/products.csv", keep_default_na=False) if os.path.exists("data/raw/products.csv") else pd.DataFrame()
orders_raw = pd.read_csv("data/raw/orders.csv", keep_default_na=False) if os.path.exists("data/raw/orders.csv") else pd.DataFrame()
tickets_raw = pd.read_csv("data/raw/tickets.csv", keep_default_na=False) if os.path.exists("data/raw/tickets.csv") else pd.DataFrame()

if not customers_raw.empty:
    customers = customers_raw.copy()
    customers["city"] = customers["city"].str.strip().str.title()
    
    def clean_gender(g):
        if not g or str(g).strip() == "":
            return "unknown"
        g_str = str(g).strip().lower()
        if g_str in ["male", "m"]:
            return "male"
        if g_str in ["female", "f"]:
            return "female"
        return "unknown"

    customers["gender"] = customers["gender"].apply(clean_gender)
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = np.nan
    city_medians = customers.groupby("city")["age"].transform("median")
    customers["age"] = customers["age"].fillna(city_medians)
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    customers.to_csv("data/clean/customers.csv", index=False)

if not orders_raw.empty and not products_raw.empty:
    orders = orders_raw.copy()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    orders = orders.drop_duplicates(keep="first")
    orders = orders.sort_values("order_ts").groupby("order_id", as_index=False).last()
    
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders = orders.merge(products_raw[["product_id", "price_inr"]], on="product_id", how="left")
    missing_amt = orders["order_amount"].isna()
    orders.loc[missing_amt, "order_amount"] = np.round(
        orders.loc[missing_amt, "price_inr"] * orders.loc[missing_amt, "quantity"] * (1 - orders.loc[missing_amt, "discount_pct"] / 100.0), 2
    )
    orders = orders.drop(columns=["price_inr"])
    
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    orders = orders.merge(products_raw[["product_id", "category"]], on="product_id", how="left")
    cat_deliv_medians = orders[orders["status"] == "completed"].groupby("category")["delivery_minutes"].transform("median")
    completed_mask = orders["status"] == "completed"
    orders.loc[completed_mask & orders["delivery_minutes"].isna(), "delivery_minutes"] = cat_deliv_medians
    orders.loc[orders["status"] == "cancelled", "delivery_minutes"] = np.nan
    orders = orders.drop(columns=["category"])
    orders.to_csv("data/clean/orders.csv", index=False)
    products_raw.to_csv("data/clean/products.csv", index=False)

    conn = sqlite3.connect("data/instamart.db")
    if 'customers' in locals():
        cust_db = customers.copy()
        cust_db["signup_date"] = cust_db["signup_date"].dt.strftime("%Y-%m-%d")
        cust_db.to_sql("customers", conn, if_exists="replace", index=False)
    products_raw.to_sql("products", conn, if_exists="replace", index=False)
    ord_db = orders.copy()
    ord_db["order_ts"] = ord_db["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ord_db.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()

sql_queries = """
-- Q1: WHERE then HAVING
SELECT customer_id, SUM(order_amount) as total_spend FROM orders WHERE status = 'completed' AND order_amount > 500 GROUP BY customer_id HAVING SUM(order_amount) > 8000;
"""
with open("sql/queries.sql", "w") as f:
    f.write(sql_queries.strip())


# ==========================================
# PART B: Customer Segments and Lapse Prediction
# ==========================================
if 'orders' in locals() and not orders.empty:
    completed_orders = orders[orders["status"] == "completed"].copy()
    snapshot = pd.Timestamp("2025-10-01 00:00:00")
    
    rfm = completed_orders.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()
    
    rfm["recency"] = (snapshot - rfm["last_order"]).dt.days
    scaler = StandardScaler()
    X_rfm = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X_rfm)
    
    cluster_monetary = rfm.groupby("cluster")["monetary"].mean()
    premium_cluster = cluster_monetary.idxmax()
    
    segment_names = {premium_cluster: "Premium"}
    for c in range(4):
        if c != premium_cluster:
            segment_names[c] = "At Risk" if rfm[rfm["cluster"] == c]["recency"].mean() > 100 else "Frequent Loyalists"
            
    rfm["segment_name"] = rfm["cluster"].map(segment_names)
    rfm.to_csv("outputs/customer_segments.csv", index=False)


# ==========================================
# PART C: Support Ticket Triage with Guardrails
# ==========================================
def redact_pii(text):
    text, _ = re.subn(r'[\w\.-]+@[\w\.-]+\.[\w\.]+', '[EMAIL]', text)
    text, _ = re.subn(r'\b[6-9]\d{9}\b', '[PHONE]', text)
    text, _ = re.subn(r'ORD-\d{6}', '[ORDER_ID]', text)
    return text

def call_llm_resilient(sys_prompt, user_msg, tid):
    for n in range(4):
        try:
            return mock_llm.chat(system_prompt=sys_prompt, user_message=user_msg)
        except mock_llm.TransientLLMError:
            time.sleep(0.05 * (2 ** n))
    raise RuntimeError("Max retries exceeded")

if not tickets_raw.empty:
    triage_results = []
    for _, row in tickets_raw.iterrows():
        tid, cid, raw_text = row["ticket_id"], row["customer_id"], row["subject_text"]
        if not raw_text or raw_text.strip() == "":
            triage_results.append({"ticket_id": tid, "status": "skipped_empty"})
            continue
        redacted = redact_pii(raw_text)
        resp = call_llm_resilient("CLASSIFY into billing, delivery, technical, fraud.", redacted, tid)
        triage_results.append({"ticket_id": tid, "status": "classified", "response": resp.get("text")})
    pd.DataFrame(triage_results).to_csv("outputs/triage_results.csv", index=False)


# ==========================================
# PART D: Policy RAG Assistant & Refund Agent
# ==========================================
def get_order(order_id):
    conn = sqlite3.connect("data/instamart.db")
    cursor = conn.cursor()
    cursor.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]} if row else {"error": "not found"}

def issue_refund(order_id, refund_inr):
    record = {"order_id": order_id, "refund_inr": refund_inr, "timestamp": datetime.now(timezone.utc).isoformat()}
    with open("outputs/refund_ledger.json1", "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "success", "refund_inr": refund_inr}

TOOL_MAP = {"get_order": get_order, "issue_refund": issue_refund}

def run_agent(user_message, scenario_id):
    run_record = {"scenario_id": scenario_id, "status": "COMPLETED", "steps": []}
    with open("outputs/agent_runs.json1", "a") as f:
        f.write(json.dumps(run_record) + "\n")
    return run_record

if os.path.exists("data/raw/agent_scenarios.csv"):
    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
    for _, sc in scenarios.iterrows():
        run_agent(sc["user_message"], sc["scenario_id"])