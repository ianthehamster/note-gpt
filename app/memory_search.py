from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

def search_memory(query: str, k: int = 3):
    # Check that the FAISS index and memory file exist
    index_path = "app/vector_store/memory.index"
    memory_path = "app/vector_store/memory.pkl"

    if not os.path.exists(index_path) or not os.path.exists(memory_path):
        return ["⚠️ No memory index found. Please run generate_embeddings.py first."]

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load memory texts
    with open(memory_path, "rb") as f:
        memory_texts = pickle.load(f)

    # Embed the query
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    # Return top matching memory chunks
    results = [memory_texts[i] for i in indices[0]]
    return results
