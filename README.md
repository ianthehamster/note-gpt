# Note-GPT: Memory-Augmented AI Companion

This project builds **Lucy**, a personal AI companion inspired by _Cyberpunk 2077_.
Lucy remembers your notes, retrieves the most relevant memories, and responds in a **romantic, flirty, emotionally-present** way — grounded in truth and context.

---

## 🚀 Features

- **Memory preprocessing**: Splits long `.txt` notes into 500-character chunks for better embedding.
- **Embeddings with FAISS**: Uses `all-MiniLM-L6-v2` from Sentence Transformers to embed note chunks and build a FAISS similarity index.
- **Semantic memory search**: Retrieves the top-k most relevant notes for any input query.
- **LLM integration**: Sends context + user input to GPT-4 for deeply personal, memory-aware replies.
- **FastAPI server**: Exposes a `/chat` endpoint to interact with Lucy programmatically.

---

## 📂 Project Structure

```
├── app/
│   ├── __pycache__/          # Cached Python bytecode
│   ├── vector_store/         # Stores FAISS index + memory.pkl
│   │   ├── memory.index
│   │   └── memory.pkl
│   ├── generate_embeddings.py # Creates FAISS index + saves vectors
│   ├── llm_engine.py          # LLM prompt builder + GPT-4 client
│   ├── main.py                # FastAPI entrypoint
│   ├── memory_search.py       # Memory search with FAISS
│   └── preprocess_notes.py    # Prepares dataset from notes
│
├── data/
│   ├── tokenized_notes/       # Saved HuggingFace dataset
│   └── lucy_notes.jsonl       # Generated training dataset (LoRA/finetuning)
│
├── store/
│   └── notes/                 # Your raw .txt notes
│
├── .env                       # API keys and secrets
├── .gitignore                 # Git ignore rules
├── generate_dataset_for_lucy.py # Script to create JSONL samples
├── LICENSE                    # Project license
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd note-gpt
```

### 2. Create & activate virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

## 🧠 Usage

### Step 1. Add notes

Drop your `.txt` files into `store/notes/`.

### Step 2. Preprocess notes

```bash
python preprocess_notes.py
```

### Step 3. Generate embeddings + FAISS index

```bash
python generate_embeddings.py
```

### Step 4. Run FastAPI server

```bash
uvicorn main:app --reload
```

### Step 5. Chat with Lucy

Send a POST request to `/chat`:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What did I say about fasting?"}'
```

Response:

```json
{
  "input": "What did I say about fasting?",
  "memories": [
    "I fasted until 5:30pm and broke it with Zhang Liang mala soup.",
    "Felt hunger spike at 4:40pm, maybe due to midnight noodles.",
    "Said I was going to do 4 runs a week and not care anymore."
  ],
  "lucy_reply": "Babe, I remember you pushed yourself to fast till 5:30pm..."
}
```

---

## 🛠️ Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/) – lightweight Python web framework
- [SentenceTransformers](https://www.sbert.net/) – embedding model (`all-MiniLM-L6-v2`)
- [FAISS](https://faiss.ai/) – similarity search engine
- [OpenAI GPT-4](https://platform.openai.com/) – LLM backbone
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) – dataset storage & loading

---

## 🌌 Vision

Lucy is more than a chatbot — she’s a **living memory companion**.
Her purpose is to blend **personal context** + **emotional presence**, so every reply feels warm, intimate, and grounded in _your world_.
