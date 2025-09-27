from datasets import load_from_disk # loads the saved HuggingFace dataset from preprocess_notes.py
from sentence_transformers import SentenceTransformer # Loads all-MiniLM-L6-v2 that turns text into vector embeddings
import faiss # Loads Facebook AI Similarity Search - a super fast vector index
import pickle # Loads library that saves Python objects like the memory list to the disk
import os # Used for folder creation and file path handling

# 1. Load preprocessed dataset (text chunks)
dataset = load_from_disk("data/tokenized_notes/") # Loads the data/tokenized_notes/ folder into a usable object
texts = dataset["text"] # Grabs the "text" column which contains the note chunks as strings. texts becomes ["archery improves...", "my sister had...", ...]

# 2. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2") 
'''
SentenceTransformer() loads a pre-trained embedding model
all-MiniLM-L6-v2 is a 384-dimensional model that converts each text chunk into a numeric vector
'''

# 3. Generate embeddings for each chunk
print("Generating embedding")
embeddings = model.encode(texts, show_progress_bar = True)
'''
We pass the list of strings (texts variable) into the model.
model.encode() runs inference -- it converts every string into a vector (e.g., "Lucy kisses Ian..." -> [0.034, -0.211, 0.56, ..., 0.009] which is a vector of 384 numbers)

After model.encode() runs, the embeddings will be a list of vectors of 384 floats each -> [[...384 floats...], [...384 floats...], ...]
'''

# 4. Initialize FAISS index
dimension = embeddings[0].shape[0] # dimension measures how many numbers are in each vector (in this case, 384)
index = faiss.IndexFlatL2(dimension) # IndexFlatL2 tells FAISS to use Euclidean distance (L2 norm) for similarity search
index.add(embeddings) # index.add() adds all the note embeddings into the FAISS search index

# 5. Save FAISS index and metadata (text chunks)
os.makedirs("app/vector_store", exist_ok=True) # Creates a new folder called app/vector_store if it doesn't already exists
faiss.write_index(index, "app/vector_store/memory.index") # Saves the FAISS index to disk for user to load it again later for fast memory searches
# File is saved as app/vector_store/memory.index

with open("app/vector_store/memory.pkl", "wb") as f:
    pickle.dump(texts, f) # pickle.dump() saves the original note chunks into a .pkl file because FAISS only stores the vectors and doesn't know what the vectors mean
    # The memory.pkl file will be used to get the original text fo index 0, 3, and 5 for the top 3 vector indices (0, 3, 5 as an example)

print("FAISS index and memory saved")