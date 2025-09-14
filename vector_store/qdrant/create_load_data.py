import os
import uuid
from io import BytesIO
import urllib.request
import pandas as pd

from sentence_transformers import SentenceTransformer

# Qdrant models (use the modern path)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    SparseVector,
)
from sklearn.feature_extraction.text import TfidfVectorizer

import re

# Qdrant client (cloud setup)
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.2N5NtTzlma6s4laWrqmm_-NHJpkKP6sGEJU79RgBj74"
URL = "https://3e75098e-82de-4a84-97a2-c8d451f8b12f.us-west-1-0.aws.cloud.qdrant.io:6333"
COLLECTION_NAME = "hk_restaurants"

client = QdrantClient(url=URL, api_key=API_KEY)
print("Existing collections:", [c.name for c in client.get_collections().collections])

# Embedding models and TF-IDF
dense_embedder = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim
tfidf_vectorizer = TfidfVectorizer( 
    max_features=10000,
    sublinear_tf=True,
    min_df=2,
    token_pattern=r'(?u)\b\w+\b'
)
# Load data from Google Sheets (Excel)
sheet_url = 'https://docs.google.com/spreadsheets/d/1U16glnBeVgMRG359bGxwnJnOlS_2oBBbpmvr5wUD6UU/export?format=xlsx'
try:
    with urllib.request.urlopen(sheet_url) as resp:
        data = resp.read()
    df = pd.read_excel(BytesIO(data))
    print("Excel data loaded successfully. Rows:", len(df))
except Exception as e:
    raise RuntimeError(f"Error loading Excel: {e}")

# Ensure required columns exist
required_cols = [
    'Name', 'Location', 'Description/Highlights', 'Why Recommended',
    'Cost Estimate', 'Unique Tips'
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in sheet: {missing}")


# Token length validation function
def validate_token_length(text, field_name, min_tokens, max_tokens):
    tokens = re.findall(r"\w+", str(text).lower())  # letters/numbers/underscore
    token_count = len(tokens)
    if token_count < min_tokens:
        return False, f"{field_name} too short: {token_count} tokens (min {min_tokens})"
    if token_count > max_tokens:
        return False, f"{field_name} too long: {token_count} tokens (max {max_tokens})"
    return True, token_count

# Prepare texts for vectorization
texts = []
for _, row in df.iterrows():
    combined_text = " ".join([
        str(row.get('Name', '')),
        str(row.get('Location', '')),
        str(row.get('Description/Highlights', '')),
        str(row.get('Why Recommended', '')),
        str(row.get('Cost Estimate', '')),
        str(row.get('Unique Tips', '')),
    ])
    texts.append(combined_text)


# Create collection (hybrid: dense + sparse)
# all-MiniLM-L6-v2 outputs 384 dims
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
    sparse_vectors_config={"text": SparseVectorParams()},
)

# Build dense and sparse vectors
dense_vectors = dense_embedder.encode(texts, show_progress_bar=True).tolist()
sparse_matrix = tfidf_vectorizer.fit_transform(texts)

# Upsert points
points = []
for i, row in df.iterrows():
    dense_vector = dense_vectors[i]
    sparse_indices = sparse_matrix[i].indices.tolist()
    sparse_values = sparse_matrix[i].data.tolist()

    # Validate fields
    desc_valid, desc_msg = validate_token_length(row['Description/Highlights'], "Description/Highlights", 5, 100)
    why_valid,  why_msg  = validate_token_length(row['Why Recommended'], "Why Recommended", 3, 50)
    tips_valid, tips_msg = validate_token_length(row['Unique Tips'], "Unique Tips", 2, 30)
    cost_valid, cost_msg = validate_token_length(row['Cost Estimate'], "Cost Estimate", 1, 50)

    if not (desc_valid and why_valid and tips_valid and cost_valid):
        print(f"Skipping row {i} due to validation errors: {desc_msg}, {why_msg}, {tips_msg}, {cost_msg}")
        continue
    # Optional: parse numeric cost (very simple heuristic)
    cost_raw = str(row['Cost Estimate']).strip()
    parsed_cost = None
    try:
        if "HKD" in cost_raw and "-" not in cost_raw:
            num = cost_raw.replace("HKD", "").replace("$", "").strip()
            parsed_cost = float(num)
    except Exception:
        parsed_cost = None


   
    payload = {
        'Name': str(row['Name']),
        'Location': str(row['Location']),
        'Description/Highlights': str(row['Description/Highlights']),
        'Why Recommended': str(row['Why Recommended']),
        'Cost Estimate': cost_raw,
        'Parsed Cost': parsed_cost,
        'Unique Tips': str(row['Unique Tips']),
        'Source': str(row.get('Source', 'Unknown')),
    }
    
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            # IMPORTANT: hybrid shapes
            vector={"dense": dense_vector},
            payload=payload,
            sparse_vectors={"text": SparseVector(indices=sparse_indices, values=sparse_values)},
        )
    )

# Upsert data to Qdrant
if points:
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Uploaded {len(points)} points to '{COLLECTION_NAME}'.")
else:
    print("No valid points to upload.")