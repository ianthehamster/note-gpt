from datasets import Dataset # Import HuggingFace's Dataset class to create a dataset for training or embedding
import os, glob # os to create folders if needed and glob to find all .txt files in a folder (e.g., store/notes/*.txt)

notes_dir = "store/notes" # where the notes are stored
os.makedirs(notes_dir, exist_ok=True) # If store/notes does not exist, create the folder

# Takes the full note and split it into smaller chunks of ~500 characters
def chunk_text(text, chunk_size = 500):
    # List comprehension to build the list of chunks (e.g., ["chunk 1", "chunk 2", "chunk 3"])
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)] 
    # range(0, len(text), chunk_size) creates a list of numbers from 0 to length of text, in multiples of chunk_size (e.g., [0, 500, 1000, 1500])
    # text[i:i+chunk_size] slices the string text from index i to i + chunk_size. It extracts chunks like text[0:500], text[500:1000] etc.

'''
    For example, a text of length 1800 will create a list of numbers first from 0 to 1800 -> [0, 500, 1000, 1500]
    Then, it slices and extracts the original text string of length 1800 from index i to i + chunk_size 

    Note: Embeddings and LLMs work best with bite-sized chunks (typically < 1,000 tokens)
'''


def load_notes():
    all_texts = [] # empty list to hold all the chunks from all notes
    for filepath in glob.glob(notes_dir + "/*.txt"): # glob.glob(notes_dir + "/*.txt") finds all .txt files in the folder store/notes
        # glob.glob() returns a list offile paths like ["store/notes/archery.txt", "store/notes/hypnaraai.txt"]
        with open(filepath, "r", encoding="utf-8") as f: # Opens each .txt file in read mode, "r", and assigns it as variable f
            chunks = chunk_text(f.read()) # f.read() reads the entire .txt file as one big string and chunk_text splits the .txt file string into 500-character chunks
            all_texts.extend(chunks) # .extend() adds each item from chunks to all_texts

    return all_texts

if __name__ == "__main__":
    chunks = load_notes() # Calls load_notes() to construct the all_texts list containing all chunks from all .txt files in store/notes folder
    dataset = Dataset.from_dict({"text": chunks}) # Dataset.from_dict() converts a Python dictionary into a HuggingFace dataset. {"text": chunks} creates one column called text and each item in chunks becomes a row in the dataset
    '''
    Example of dataset variable
    | text              |
    |-------------------|
    | chunk 1 of note A |
    | chunk 2 of note A |
    | chunk 1 of note B |

    '''
    dataset.save_to_disk("data/tokenized_notes/") # writes the dataset to disk in HuggingFace's Arrow format (e.g., data-00000-of-00001.arrow, dataset_info.json, state.json)
    print("Notes chunked and saved") 