from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# 1. Load preprocessed dataset (text chunks)
dataset = load_from_disk("data/tokenized_notes/")
texts = dataset["text"]

# 2. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Generate embeddings for each chunk
print("Generating embedding")
embeddings = model.encode(texts, show_progress_bar = True)

# 4. Initialize FAISS index
dimension = embeddings[0].shape[0] # e.g., 384
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 5. Save FAISS index and metadata (text chunks)
os.makedirs("app/vector_store", exist_ok=True)
faiss.write_index(index, "app/vector_store/memory.index")

with open("app/vector_store/memory.pkl", "wb") as f:
    pickle.dump(texts, f)

print("FAISS index and memory saved")