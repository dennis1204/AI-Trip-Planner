from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION = "hk_restaurants"

# Initialize Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=180.0,
    prefer_grpc=True,
)

# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def test_vector_search(keywords, top_k=3):
    # Embed the keywords
    query_vector = embedder.encode(keywords).tolist()

    # Search the collection
    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,  # Include payload data
        with_vectors=False,   # Exclude vector data to save space
        search_params={"vectors": "dense"}
    )

    # Process and display results
    print(f"Top {top_k} matches for keywords: '{keywords}'")
    for result in results:
        payload = result.payload
        score = result.score
        print(f"- Name: {payload['name']}, District: {payload['district']}, "
              f"Score: {score:.4f}, Highlights: {payload['description_highlights'][:50]}...")

# Test with some keywords
test_keywords = [
    "budget seafood Sham Shui Po dinner",
    "affordable dim sum Kowloon",
    "cozy cafe Central"
]

for keywords in test_keywords:
    test_vector_search(keywords)