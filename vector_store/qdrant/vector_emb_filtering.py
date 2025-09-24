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

# Enhanced deduplication using vector embeddings for fuzzy matching
print("Starting removing duplicates in excel...");
def normalize_string(s: str) -> str:
    # Normalize accents/diacritics to improve matching (e.g., "Café" -> "Cafe")
    return unicodedata.normalize('NFKD', s.strip().lower()).encode('ascii', 'ignore').decode('utf-8')

# Apply normalization to Name and Location
df['normalized_name'] = df['Name'].apply(normalize_string)
df['normalized_location'] = df['Location'].apply(normalize_string)

# Combine for embedding (you can include more fields if needed, e.g., + ' ' + normalize_string(row['Description/Highlights']))
combined = df['normalized_name'] + ' ' + df['normalized_location']

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Reuse your model; move this up if you want to avoid reloading later
embeddings = embedder.encode(combined.tolist())

# Compute pairwise cosine similarities
sim_matrix = cosine_similarity(embeddings)

# Threshold for considering near-duplicates (0.8-0.9 works well; tune based on your data)
threshold = 0.85

# Find groups of near-duplicates
visited = np.zeros(len(df), dtype=bool)
groups = []
for i in range(len(df)):
    if visited[i]:
        continue
    group = [i]
    for j in range(i + 1, len(df)):
        if sim_matrix[i, j] >= threshold:
            group.append(j)
    if len(group) > 1:
        groups.append(group)
    visited[group] = True

# For each group, decide which to keep (e.g., the one with the longest 'Description/Highlights' as a proxy for most detailed)
to_keep = []
for group in groups:
    # Compute lengths of a key field (or use another criteria, like max 'parsed_cost' if relevant)
    lengths = [tokenize_len(df.iloc[idx]['Description/Highlights']) for idx in group]
    keep_idx = group[np.argmax(lengths)]  # Keep the one with max length
    to_keep.append(keep_idx)

# All non-grouped so are unique, add them
unique_indices = [i for i in range(len(df)) if not any(i in g for g in groups)]
to_keep.extend(unique_indices)

# Sort to preserve original order if desired
to_keep.sort()

# Create deduplicated df
df = df.iloc[to_keep].drop(columns=['normalized_name', 'normalized_location']).reset_index(drop=True)
print(f"Deduplicated to {len(df)} unique rows (removed {len(visited) - len(df)} near-duplicates)")
