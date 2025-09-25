# load_data.py
# ------------------------------------------
# pip install qdrant-client sentence-transformers pandas scikit-learn openpyxl requests joblib

import os, uuid, re, hashlib, joblib
import pandas as pd
import requests
from io import BytesIO

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector,
)

from dotenv import load_dotenv
load_dotenv()

# ---------------- Config ----------------
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION = "hk_restaurants"
EXCEL_SOURCE = "https://docs.google.com/spreadsheets/d/1U16glnBeVgMRG359bGxwnJnOlS_2oBBbpmvr5wUD6UU/export?format=xlsx"

TFIDF_PATH = "tfidf.pkl"   # persist sparse model here

# ---------------- Helpers ----------------
def load_excel(path_or_url: str) -> pd.DataFrame:
    if path_or_url.lower().startswith("http"):
        r = requests.get(path_or_url, timeout=60)
        r.raise_for_status()
        return pd.read_excel(BytesIO(r.content), sheet_name="工作表2")
    return pd.read_excel(path_or_url, sheet_name="工作表2")

def parse_cost(cost_raw: str):
    if cost_raw is None or pd.isna(cost_raw):
        return None
    s = str(cost_raw).replace("HKD", "").replace("$", "").strip()
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else None

def parse_coordinates(coord_raw: str):
    if not coord_raw or pd.isna(coord_raw):
        return None, None
    s = str(coord_raw).strip()
    if "(" in s and ")" in s:
        coords = s.strip("()").split(",")
        if len(coords) == 2:
            try:
                lat = float(coords[0].strip())
                lon = float(coords[1].strip())
                return lat, lon
            except ValueError:
                return None, None
    return None, None

def stable_id(name: str, district: str) -> str:
    base = f"{str(name).strip().lower()}|{str(district).strip().lower()}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

def row_to_text(r: pd.Series) -> str:
    # build text for sparse indexing (weighted fields)
    parts = [
        str(r.get("店名","")),
        str(r.get("菜系","")),
        # str(r.get("所在區","")),
        str(r.get("地址","")),
        str(r.get("餐廳亮點","")),
        # str(r.get("推薦原因","")),
        # str(r.get("獨門貼士","")),
        # str(r.get("來源平台","")),
        # str(r.get("來源標題","")),
        # str(r.get("頻道／帳號","")),
        # str(r.get("消費預算—早餐（HKD/人）","")),
        # str(r.get("消費預算—午餐（HKD/人）","")),
        # str(r.get("消費預算—晚餐（HKD/人）","")),
    ]
    return " ".join(parts).strip()

# ---------------- Main ----------------
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=180.0,
    prefer_grpc=True,
    grpc_options={
        "grpc.max_send_message_length": 64 * 1024 * 1024,
        "grpc.max_receive_message_length": 64 * 1024 * 1024,
    },
)

# Create collection if missing
if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
        sparse_vectors_config={"text": SparseVectorParams()},
    )
    print(f"Created collection: {COLLECTION}")
else:
    print(f"Using existing collection: {COLLECTION}")

# ---------------- Payload Indexes ----------------
# Define which payload fields you plan to filter/search on
PAYLOAD_INDEXES = {
    "district": "keyword",
    "cuisine": "keyword",
    "name": "keyword",
    "parsed_cost_breakfast": "float",
    "parsed_cost_lunch": "float",
    "parsed_cost_dinner": "float",
    # Add others if you plan to filter/sort on them
}

for field, schema in PAYLOAD_INDEXES.items():
    try:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=schema,
        )
        print(f"[ok] Indexed '{field}' as {schema}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"[ok] Index already exists: {field}")
        else:
            print(f"[warn] Index error for {field}: {e}")

# Load Excel
df = load_excel(EXCEL_SOURCE)
print(f"Loaded Excel rows: {len(df)}")

# Deduplicate
df["normalized_name"] = df["店名"].astype(str).str.strip().str.lower()
df = df.drop_duplicates(subset=["normalized_name"], keep="last").drop(columns=["normalized_name"]).reset_index(drop=True)
print(f"Deduplicated to {len(df)} unique rows")

# Required cols
required_cols = ["店名","所在區","餐廳亮點","推薦原因","消費預算—早餐（HKD/人）","消費預算—午餐（HKD/人）","消費預算—晚餐（HKD/人）","獨門貼士"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Build corpus texts
texts = [row_to_text(r) for _, r in df.iterrows()]

# Dense embeddings
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
dense_vectors = embedder.encode(texts, show_progress_bar=True).tolist()

# Sparse vectors: fit once or reuse existing
if os.path.exists(TFIDF_PATH):
    print(f"Loading existing TF-IDF from {TFIDF_PATH}")
    tfidf = joblib.load(TFIDF_PATH)
    sparse = tfidf.transform(texts)
else:
    print("Fitting TF-IDF on corpus and saving...")
    tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2,4),  # good for Chinese
        min_df=2, max_features=100000, sublinear_tf=True,
    )
    sparse = tfidf.fit_transform(texts)
    joblib.dump(tfidf, TFIDF_PATH)

# Build Qdrant points
points = []
for i, row in df.iterrows():
    indices = sparse[i].indices.tolist()
    values = sparse[i].data.tolist()

    lat, lon = parse_coordinates(row.get("座標（lat, lon）"))
    payload = {
        "name": str(row.get("店名")),
        "cuisine": str(row.get("菜系")),
        "district": str(row.get("所在區")),
        "address": str(row.get("地址")),
        "latitude": lat,
        "longitude": lon,
        "description_highlights": str(row.get("餐廳亮點")),
        "why_recommended": str(row.get("推薦原因")),
        "unique_tips": str(row.get("獨門貼士")),
        "source_platform": str(row.get("來源平台")),
        "source_title": str(row.get("來源標題")),
        "source_url": str(row.get("來源連結（URL）")),
        "channel_account": str(row.get("頻道／帳號")),
        "last_checked_date": str(row.get("最後核對日期（YYYY-MM-DD）")),
        "cost_estimate_breakfast": str(row.get("消費預算—早餐（HKD/人）")),
        "parsed_cost_breakfast": parse_cost(row.get("消費預算—早餐（HKD/人）")),
        "cost_estimate_lunch": str(row.get("消費預算—午餐（HKD/人）")),
        "parsed_cost_lunch": parse_cost(row.get("消費預算—午餐（HKD/人）")),
        "cost_estimate_dinner": str(row.get("消費預算—晚餐（HKD/人）")),
        "parsed_cost_dinner": parse_cost(row.get("消費預算—晚餐（HKD/人）")),
    }

    points.append(PointStruct(
        id=stable_id(row["店名"], row["所在區"]),
        vector={
            # "dense": dense_vectors[i],
            "text": SparseVector(indices=indices, values=values),
        },
        payload=payload,
    ))

# Upsert in batches
def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

if points:
    for batch in chunked(points, 64):
        client.upsert(collection_name=COLLECTION, points=batch, wait=True)
    print(f"Upserted {len(points)} points into '{COLLECTION}'")
else:
    print("No valid rows to upsert.")

client.close()
