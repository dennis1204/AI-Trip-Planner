# test_vector_store.py
# ---------------------------------------------------------
# pip install qdrant-client sentence-transformers joblib python-dotenv

import os
import sys
import joblib
from dotenv import load_dotenv

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


# ---------------- Config ----------------
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "hk_restaurants_v2"
TFIDF_PATH = "tfidf.pkl"   # persisted by load_data.py

# Named vectors used in your collection schema
DENSE_NAME = "dense"
SPARSE_NAME = "text"

# ---------------- Client & Models ----------------
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=True,
    timeout=180.0,
)

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # 384-dim


# ---------------- Helpers ----------------
def build_query_text(user_text: str, district: str | None = None, budget_hint: str | None = None) -> str:
    """
    Build query text in a similar style to your indexed documents to improve sparse overlap.
    """
    parts = [user_text or ""]
    if district:
        parts.append(str(district))
    if budget_hint:
        parts.append(str(budget_hint))
    return " ".join(parts).strip()


def load_tfidf_or_none():
    """
    Load the persisted TF-IDF model (created by load_data.py).
    Return None if missing, so we can gracefully fall back to dense-only.
    """
    if not os.path.exists(TFIDF_PATH):
        print(f"[warn] {TFIDF_PATH} not found. Falling back to dense-only search.")
        return None
    try:
        tfidf = joblib.load(TFIDF_PATH)
        return tfidf
    except Exception as e:
        print(f"[warn] Failed to load {TFIDF_PATH}: {e}. Falling back to dense-only.")
        return None


def make_sparse_query(tfidf, q: str):
    """
    Transform a query string into (indices, values) using the SAME TF-IDF used at ingest.
    """
    Xq = tfidf.transform([q]).tocoo()
    return Xq.col.tolist(), Xq.data.tolist()


def pretty_print(points, title: str, limit: int = 5):
    print(f"\n{title}")
    for p in points[:limit]:
        pl = p.payload or {}
        name = pl.get("name")
        district = pl.get("district")
        highlights = (pl.get("description_highlights") or "")[:60].replace("\n", " ")
        print(f"- {name} | {district} | score={p.score:.4f} | {highlights}...")


# ---------------- Search Routines ----------------
def dense_only_search(user_text: str, top_k: int = 10):
    qtext = build_query_text(user_text)
    dense_vec = embedder.encode(qtext).tolist()

    resp = client.query_points(
        collection_name=COLLECTION,
        query=dense_vec,        # list[float]
        using=DENSE_NAME,       # named dense vector
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return resp.points


def hybrid_search(user_text: str, top_k: int = 10, prefetch_mul: int = 3, fusion: str = "RRF"):
    """
    Server-side hybrid: prefetch sparse and dense, then fuse via RRF or DBSF.
    Falls back to dense-only if tfidf is missing.
    """
    tfidf = load_tfidf_or_none()
    qtext = build_query_text(user_text)

    dense_vec = embedder.encode(qtext).tolist()

    if tfidf is None:
        # No sparse model available -> dense-only
        return dense_only_search(user_text, top_k=top_k)

    # Sparse indices/values from SAME TF-IDF as ingest
    idx, val = make_sparse_query(tfidf, qtext)

    # Build fusion model
    fusion_mode = models.Fusion.RRF if fusion.upper() == "RRF" else models.Fusion.DBSF
    # print("````````````", models.FusionQuery(fusion=fusion_mode))
    resp = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(indices=idx, values=val),
                using=SPARSE_NAME,
                limit=top_k * prefetch_mul,
            ),
            models.Prefetch(
                query=dense_vec,
                using=DENSE_NAME,
                limit=top_k * prefetch_mul,
            ),
        ],
        query=models.FusionQuery(fusion=fusion_mode),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return resp.points


# ---------------- Optional: metadata filter example ----------------
def hybrid_search_with_filter(user_text: str, district_exact: str, top_k: int = 3):
    tfidf = load_tfidf_or_none()
    qtext = build_query_text(user_text, district=district_exact)
    dense_vec = embedder.encode(qtext).tolist()

    flt = models.Filter(
        must=[models.FieldCondition(
            key="district",
            match=models.MatchValue(
                value=district_exact))]
    )

    # print("This is filter", flt)

    # if tfidf is None:
    #     # fallback: dense-only with filter
    #     resp = client.query_points(
    #         collection_name=COLLECTION,
    #         query=dense_vec,
    #         using="dense",
    #         limit=top_k,
    #         with_payload=True,
    #         with_vectors=False,
    #         query_filter=flt,
    #     )
    #     return resp.points

    idx, val = make_sparse_query(tfidf, qtext)

    # print("+++++", models.FusionQuery(fusion=models.Fusion.RRF))

    resp = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(indices=idx, values=val),
                using="text",
                limit=top_k * 3,
                filter=flt,          # <-- FIXED (was query_filter)
            ),
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=top_k * 3,
                filter=flt,          # <-- FIXED
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return resp.points


# ---------------- Demo ----------------
if __name__ == "__main__":
    # Quick sanity check: collection + named vectors exist
    try:
        info = client.get_collection(COLLECTION)
        vectors_cfg = info.config.params.vectors
        print(f"[info] Collection '{COLLECTION}' vectors config: {vectors_cfg}")
    except Exception as e:
        print(f"[error] Could not get collection '{COLLECTION}': {e}")
        sys.exit(1)

    test_queries = [
        "印度菜",
        "budget seafood Sham Shui Po dinner",
        "affordable dim sum Kowloon",
        "cozy cafe Central",
    ]

    for q in test_queries:
        # pts_f = hybrid_search_with_filter(q, district_exact="九龍城區")
        # pretty_print(pts_f, "Hybrid + filter(district='九龍城區')")
        pts_h = hybrid_search(q, top_k=3, fusion="RRF")
        pretty_print(pts_h, f"Hybrid results for: {q!r}")

    # Example with a hard district filter:
    # pts_f = hybrid_search_with_filter("dim sum", district_exact="Kowloon City", top_k=10)
    # pretty_print(pts_f, "Hybrid + filter(district='Kowloon City') for 'dim sum'")
