# pip install qdrant-client sentence-transformers pandas scikit-learn openpyxl requests
import os, uuid, re, hashlib
import pandas as pd
import requests
from io import BytesIO

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import unicodedata  # For accent normalization
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector
)
from httpx import Timeout
from dotenv import load_dotenv
load_dotenv()
# ---------------- Config ----------------
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION = "hk_restaurants"  # reuse an existing collection name if you already have one

EXCEL_SOURCE = "https://docs.google.com/spreadsheets/d/1U16glnBeVgMRG359bGxwnJnOlS_2oBBbpmvr5wUD6UU/export?format=xlsx"

# Which fields to index & how
PAYLOAD_INDEXES = {
    "name": "keyword",
    "cuisine": "keyword",
    "district": "keyword",
    "address": "keyword",
    "source_platform": "keyword",
    "channel_account": "keyword",
    "parsed_cost_breakfast": "float",
    "parsed_cost_lunch": "float",
    "parsed_cost_dinner": "float",
    "latitude": "float",
    "longitude": "float",
    # only if you plan to do full-text filtering; otherwise skip:
    "description_highlights": "text",
    "why_recommended": "text",
    "unique_tips": "text",
    "source_title": "text",
}

# ---------------- Helpers ----------------
def load_excel(path_or_url: str) -> pd.DataFrame:
    if path_or_url.lower().startswith("http"):
        r = requests.get(path_or_url, timeout=60)
        r.raise_for_status()
        return pd.read_excel(BytesIO(r.content), sheet_name="工作表2")
    return pd.read_excel(path_or_url, sheet_name="工作表2")

def tokenize_len(s: str) -> int:
    return len(re.findall(r"\w+", str(s).lower()))

def parse_cost(cost_raw: str):
    if cost_raw is None:
        return None
    s = str(cost_raw).replace("HKD", "").replace("$", "").strip()
    # very simple: take first number if present or handle range
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else None

def parse_coordinates(coord_raw: str):
    if not coord_raw or pd.isna(coord_raw):
        return None, None
    s = str(coord_raw).strip()
    if '(' in s and ')' in s:
        coords = s.strip('()').split(',')
        if len(coords) == 2:
            try:
                lat = float(coords[0].strip())
                lon = float(coords[1].strip())
                return lat, lon
            except ValueError:
                return None, None
    return None, None

def stable_id(name: str, district: str) -> str:
    # Stable ID so re-runs update instead of duplicating
    base = f"{str(name).strip().lower()}|{str(district).strip().lower()}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

# ---------------- Main ----------------
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=180.0,  # ↑ give writes time
    prefer_grpc=True,
    grpc_options={
        "grpc.max_send_message_length": 64 * 1024 * 1024,     # 64MB
        "grpc.max_receive_message_length": 64 * 1024 * 1024,
    },
)

# Create collection once (if missing)
if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
        sparse_vectors_config={"text": SparseVectorParams()},
    )
    print(f"Created collection: {COLLECTION}")
else:
    print(f"Using existing collection: {COLLECTION}")

# Add payload indexes (idempotent-ish)
for field, schema in PAYLOAD_INDEXES.items():
    try:
        client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)
        print(f"Indexed '{field}' as {schema}")
    except Exception as e:
        # Likely already exists; ignore unless it's a different error
        if "already exists" not in str(e).lower():
            print(f"Index warn for {field}: {e}")

df = load_excel(EXCEL_SOURCE)
print(f"Loaded Excel rows: {len(df)}")

print("Starting removing duplicates in excel...")
df['normalized_name'] = df['店名'].astype(str).str.strip().str.lower()
dropped_rows = df[df.duplicated(subset=['normalized_name'], keep=False)]
print("Dropped rows due to duplicates:", dropped_rows['店名'].tolist())
df = df.drop_duplicates(subset=['normalized_name'], keep='last').drop(columns=['normalized_name']).reset_index(drop=True)
print(f"Deduplicated to {len(df)} unique rows")

required_cols = [
    "店名", "所在區", "餐廳亮點", "推薦原因",
    "消費預算—早餐（HKD/人）", "消費預算—午餐（HKD/人）", "消費預算—晚餐（HKD/人）", "獨門貼士"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Prepare texts for hybrid vectors
texts = [
    " ".join([
        str(r.get("店名", "")),
        str(r.get("菜系", "")),
        str(r.get("所在區", "")),
        str(r.get("地址", "")),
        str(r.get("餐廳亮點", "")),
        str(r.get("推薦原因", "")),
        str(r.get("消費預算—早餐（HKD/人）", "")),
        str(r.get("消費預算—午餐（HKD/人）", "")),
        str(r.get("消費預算—晚餐（HKD/人）", "")),
        str(r.get("獨門貼士", "")),
        str(r.get("來源平台", "")),
        str(r.get("來源標題", "")),
        str(r.get("來源連結（URL）", "")),
        str(r.get("頻道／帳號", "")),
    ])
    for _, r in df.iterrows()
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
dense_vectors = embedder.encode(texts, show_progress_bar=True).tolist()

tfidf = TfidfVectorizer(max_features=10000, sublinear_tf=True, min_df=2, token_pattern=r"(?u)\b\w+\b")
sparse = tfidf.fit_transform(texts)

points = []
for i, row in df.iterrows():
    # lightweight validation
    desc_len = tokenize_len(row["餐廳亮點"])
    why_len = tokenize_len(row["推薦原因"])
    tips_len = tokenize_len(row["獨門貼士"])
    breakfast_len = tokenize_len(row["消費預算—早餐（HKD/人）"])
    lunch_len = tokenize_len(row["消費預算—午餐（HKD/人）"])
    dinner_len = tokenize_len(row["消費預算—晚餐（HKD/人）"])
    ok = (
        desc_len >= 3 and
        why_len >= 2 and
        tips_len >= 2 and
        breakfast_len >= 1 and
        lunch_len >= 1 and
        dinner_len >= 1
    )
    if not ok:
        print(f"Skipping row {i} ({row['店名']}): "
              f"desc={desc_len}, why={why_len}, tips={tips_len}, "
              f"breakfast={breakfast_len}, lunch={lunch_len}, dinner={dinner_len}")
        continue

    indices = sparse[i].indices.tolist()
    values = sparse[i].data.tolist()

    # Parse coordinates
    latitude, longitude = parse_coordinates(row["座標（lat, lon）"])

    # Parse cost estimates
    cost_breakfast = row["消費預算—早餐（HKD/人）"]
    cost_lunch = row["消費預算—午餐（HKD/人）"]
    cost_dinner = row["消費預算—晚餐（HKD/人）"]
    parsed_cost_breakfast = parse_cost(cost_breakfast)
    parsed_cost_lunch = parse_cost(cost_lunch)
    parsed_cost_dinner = parse_cost(cost_dinner)

    payload = {
        "name": str(row["店名"]),
        "cuisine": str(row["菜系"]),
        "district": str(row["所在區"]),
        "address": str(row["地址"]),
        "latitude": latitude,
        "longitude": longitude,
        "description_highlights": str(row["餐廳亮點"]),
        "why_recommended": str(row["推薦原因"]),
        "cost_estimate_breakfast": str(cost_breakfast),
        "parsed_cost_breakfast": parsed_cost_breakfast,
        "cost_estimate_lunch": str(cost_lunch),
        "parsed_cost_lunch": parsed_cost_lunch,
        "cost_estimate_dinner": str(cost_dinner),
        "parsed_cost_dinner": parsed_cost_dinner,
        "unique_tips": str(row["獨門貼士"]),
        "source_platform": str(row["來源平台"]),
        "source_title": str(row["來源標題"]),
        "source_url": str(row["來源連結（URL）"]),
        "channel_account": str(row["頻道／帳號"]),
        "last_checked_date": str(row["最後核對日期（YYYY-MM-DD）"]),
    }

    points.append(PointStruct(
        id=stable_id(row["店名"], row["所在區"]),
        vector={
            "dense": dense_vectors[i],
            "text": SparseVector(indices=indices, values=values),
        },
        payload=payload,
    ))

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

if points:
    for batch in chunked(points, 64):  # 64 is conservative; 128/256 also fine
        client.upsert(collection_name=COLLECTION, points=batch, wait=True)
    print(f"Upserted {len(points)} points into '{COLLECTION}'.")
else:
    print("No valid rows to upsert.")

client.close()