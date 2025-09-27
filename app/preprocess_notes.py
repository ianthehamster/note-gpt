from datasets import Dataset
import os, glob

notes_dir = "store/notes"
os.makedirs(notes_dir, exist_ok=True)

def chunk_text(text, chunk_size = 500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)] 

def load_notes():
    all_texts = []
    for filepath in glob.glob(notes_dir + "/*.txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            chunks = chunk_text(f.read())
            all_texts.extend(chunks)

    return all_texts

if __name__ == "__main__":
    chunks = load_notes()
    dataset = Dataset.from_dict({"text": chunks})
    dataset.save_to_disk("data/tokenized_notes/")
    print("Notes chunked and saved") 