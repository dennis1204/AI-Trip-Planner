# pip install qdrant-client sentence-transformers pandas scikit-learn openpyxl requests
import os, uuid, re, hashlib
import pandas as pd
import requests
from io import BytesIO

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector
)
from httpx import Timeout

# ---------------- Config ----------------
QDRANT_URL ="https://3e75098e-82de-4a84-97a2-c8d451f8b12f.us-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.2N5NtTzlma6s4laWrqmm_-NHJpkKP6sGEJU79RgBj74"
COLLECTION = "hk_restaurants_v2"  # reuse an existing collection name if you already have one

EXCEL_SOURCE =  "https://docs.google.com/spreadsheets/d/1U16glnBeVgMRG359bGxwnJnOlS_2oBBbpmvr5wUD6UU/export?format=xlsx"


# Which fields to index & how
PAYLOAD_INDEXES = {
    "name": "keyword",
    "location": "keyword",
    "source": "keyword",
    "parsed_cost": "float",
    # only if you plan to do full-text filtering; otherwise skip:
    "description_highlights": "text",
    "why_recommended": "text",
    "unique_tips": "text",
}

# ---------------- Helpers ----------------
def load_excel(path_or_url: str) -> pd.DataFrame:
    if path_or_url.lower().startswith("http"):
        r = requests.get(path_or_url, timeout=60)
        r.raise_for_status()
        return pd.read_excel(BytesIO(r.content))
    return pd.read_excel(path_or_url)

def tokenize_len(s: str) -> int:
    return len(re.findall(r"\w+", str(s).lower()))

def parse_cost(cost_raw: str):
    if cost_raw is None:
        return None
    s = str(cost_raw).replace("HKD", "").replace("$", "").strip()
    # very simple: take first number if present
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else None

def stable_id(name: str, location: str) -> str:
    # Stable ID so re-runs update instead of duplicating
    base = f"{str(name).strip().lower()}|{str(location).strip().lower()}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

# ---------------- Main ----------------
client = QdrantClient(  url=QDRANT_URL, 
                        api_key=QDRANT_API_KEY,
                        timeout=180.0,  # ↑ give writes time
                        prefer_grpc=True,
                        grpc_options=[
                        ("grpc.max_send_message_length", 64 * 1024 * 1024),     # 64MB
                        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                    ],
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

required_cols = [
    "Name", "Location", "Description/Highlights", "Why Recommended",
    "Cost Estimate", "Unique Tips"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Prepare texts for hybrid vectors
texts = [
    " ".join([
        str(r.get("Name", "")),
        str(r.get("Location", "")),
        str(r.get("Description/Highlights", "")),
        str(r.get("Why Recommended", "")),
        str(r.get("Cost Estimate", "")),
        str(r.get("Unique Tips", "")),
    ])
    for _, r in df.iterrows()
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
dense_vectors = embedder.encode(texts, show_progress_bar=True).tolist()

tfidf = TfidfVectorizer(max_features=10000, sublinear_tf=True, min_df=2, token_pattern=r"(?u)\b\w+\b")
sparse = tfidf.fit_transform(texts)

points = []
for i, row in df.iterrows():
    # lightweight validation (no NLTK needed)
    ok = (
        tokenize_len(row["Description/Highlights"]) >= 5 and
        tokenize_len(row["Why Recommended"]) >= 3 and
        tokenize_len(row["Unique Tips"]) >= 2 and
        tokenize_len(row["Cost Estimate"]) >= 1
    )
    if not ok:
        continue

    indices = sparse[i].indices.tolist()
    values = sparse[i].data.tolist()

    payload = {
        "name": str(row["Name"]),
        "location": str(row["Location"]),
        "description_highlights": str(row["Description/Highlights"]),
        "why_recommended": str(row["Why Recommended"]),
        "parsed_cost": parse_cost(row["Cost Estimate"]),
        "unique_tips": str(row["Unique Tips"]),
        "source": str(row.get("Source", "Unknown")),
    }

    points.append(PointStruct(
        id=stable_id(row["Name"], row["Location"]),
        vector={
            "dense": dense_vectors[i],         
            "text":  SparseVector(indices=indices, values=values),  #sparse vector
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

# if points:
#     client.upsert(collection_name=COLLECTION, points=points, wait=True)
#     print(f"Upserted {len(points)} points into '{COLLECTION}'.")
# else:
#     print("No valid rows to upsert.")
