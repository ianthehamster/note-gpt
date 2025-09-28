from sentence_transformers import SentenceTransformer # Loads the pretrained transformer model that turns text into dense vectors (embeddings). Using this to embed query and note chunks
import faiss # Facebook's FAISS library for fast similarity seach on dense vector indexes (semantic search engine)
import pickle # Loads the library to save and load Python objects like lists or dictionaries (e.g., note text chunks - memory.pkl)
import os 

model = SentenceTransformer("all-MiniLM-L6-v2") # Model used to convert query into embedding for similarity search

def search_memory(query: str, k: int = 3):
    # Check that the FAISS index and memory file exist
    index_path = "app/vector_store/memory.index" # memory.index stores embeddings
    memory_path = "app/vector_store/memory.pkl" # memory.pkl stores actual text from the embeddings

    if not os.path.exists(index_path) or not os.path.exists(memory_path):
        return ["⚠️ No memory index found. Please run generate_embeddings.py first."] # Checking if the index and memory files exist
        # If either is missing, generate warning to user

    # Load FAISS index from disk to read query embeddings in the memory.index
    index = faiss.read_index(index_path)

    # Load memory texts
    with open(memory_path, "rb") as f:
        memory_texts = pickle.load(f) # Loads and stores the memory text chunks from memory.pkl into memory_texts variable

    # Embed the query
    query_embedding = model.encode([query]) # all-MiniLM-L6-v2 converts the user's query into a vector embedding
    distances, indices = index.search(query_embedding, k) # index.search() looks for the k nearest vectors in the FAISS index (i.e., the most semantically similar memory chunks)
    '''
    distances: how similar each result is (not used here)
    indices: the indexes of the top k closes matches
    '''

    # Retrieves and returns the actual text chunks corresponding to the top k most similar vector matches
    results = [memory_texts[i] for i in indices[0]]  
    return results

'''
Calling search_memory("what did I say about fasting?") might return:

[
  "I fasted until 5:30pm and broke it with Zhang Liang mala soup.",
  "Felt hunger spike at 4:40pm, maybe due to midnight noodles.",
  "Said I was going to do 4 runs a week and not care anymore."
]

'''