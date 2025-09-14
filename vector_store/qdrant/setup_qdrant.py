from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import json
import uuid
import pandas as pd
import urllib.request
from io import BytesIO

API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.2N5NtTzlma6s4laWrqmm_-NHJpkKP6sGEJU79RgBj74"
URL="https://3e75098e-82de-4a84-97a2-c8d451f8b12f.us-west-1-0.aws.cloud.qdrant.io:6333"
# Qdrant client (local or cloud; replace for cloud)
# client = QdrantClient(host="localhost", port=6333)  # Local Docker
# For cloud: 
client = QdrantClient(url=URL, api_key=API_KEY)
print(client.get_collections())
# Local embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
# with open('all_extracted_restaurants.json', 'r') as f:
#     data = json.load(f)

# file_path = 'hk_restaurants.xlsx' 
# df = pd.read_excel(file_path)

url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTcHC-D2w-1YC1bbzbGiVK_jlcPz7BVOb_I2f44iLWf42Yf0HqnZAts2_0UdYiPOX8O2QGrt0u0QC7e/pub?output=xlsx'
with urllib.request.urlopen(url) as response:
    df = pd.read_excel(BytesIO(response.read()))

# Create collection (run once)
# collection_name = "hk_restaurants"
# vector_size = 384  # Dimension of all-MiniLM-L6-v2
# if not client.collection_exists(collection_name):
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)  # Cosine for similarity
#     )

# Vectorize and upload
points = []
for _, row in df.iterrows():  # Iterate over DataFrame rows
    text = f"{row['Name']} in {row['Location']}: {row['Description/Highlights']}. Why: {row['Why Recommended']}. Cost: {row['Cost Estimate']}. Tips: {row['Unique Tips']}."
    embedding = embedder.encode([text]).tolist()[0]  # Encode and get the vector
    point_id = str(uuid.uuid4())  # Unique ID
    payload = {
        'Name': row['Name'],
        'Location': row['Location'],
        'Description/Highlights': row['Description/Highlights'],
        'Why Recommended': row['Why Recommended'],
        'Cost Estimate': row['Cost Estimate'],
        'Unique Tips': row['Unique Tips'],
        'Source': row.get('Source', 'Unknown')  # Handle optional Source
    }
    points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

client.upsert(collection_name=collection_name, points=points)
print("Data vectorized and uploaded!")