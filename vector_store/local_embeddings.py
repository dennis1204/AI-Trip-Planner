from sentence_transformers import SentenceTransformer
import numpy as np

# Load model (downloads ~80MB first time)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example texts from your extracted restaurants
texts = [
    "Restaurant: Oi Man Sang in Sham Shui Po. Highlights: Fresh seafood like razor clams. Recommended because local fave. Cost: HKD 150. Tips: Add special sauce.",
    "Restaurant: Tsui Wah in Central. Highlights: Cha chaan teng dishes. Recommended because quick bites for visitors. Cost: HKD 50-80. Tips: Busy lunch."
]

# Generate embeddings
embeddings = model.encode(texts, normalize_embeddings=True)  # Array of vectors

# Save or use (e.g., for FAISS vector store)
np.save('hk_restaurant_embeddings.npy', embeddings)
print("Embeddings generated:", embeddings.shape)  # (num_texts, 384)