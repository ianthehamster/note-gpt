import requests
import json
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "openhermes"
OUTPUT_FILE = "data/lucy_notes.jsonl"
TARGET_SIZE_MB = 2 # Stop when dataset !2MB
BATCH_SIZE = 20

PROMPT = f"""
You are tasked with creating synthetic training data for a LoRA fine-tuning dataset.

Below is Lucy's personality, extracted from seed notes:
- She is a Netrunner from Cyberpunk 2077's Night City
- Mostly warm, flirtatious, occasionally aloof and cold
- Uses affectionate nicknames like "babe"
- Always grounded in Ian’s life context and memories

Generate {BATCH_SIZE} conversation samples in JSONL format.
Each line should look like this:

{{"instruction": "Use Ian’s memory to respond warmly", "input": "<User message>", "output": "<Sophia’s reply>"}}

Rules:
- User inputs should cover a variety of topics (daily life, running, injuries, game dev, Hypnara project, Jinx, emotional support, random questions).
- Outputs must always sound like Sophia: personal, warm, flirty, empathetic.
- Vary the length of replies (1–3 sentences).
- Do NOT repeat the same input/output phrasing — make them diverse.
- Do NOT include explanations, just pure JSONL lines.
"""


def call_ollama(prompt: str, model: str = MODEL):
  '''Send a prompt to Ollama and return the generated text'''
  resp = requests.post(
    OLLAMA_URL,
    json={"model": model, "prompt": prompt, "stream": False}
  )
  resp.raise_for_status()
  return resp.json()["response"]

def main():
  os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

  while True:
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024) if os.path.exists(OUTPUT_FILE) else 0
    if size_mb >= TARGET_SIZE_MB:
        print(f"✅ Reached {size_mb:.2f} MB, stopping.")
        break

    print(f"📦 Current dataset size: {size_mb:.2f} MB — generating more...")
    output = call_ollama(PROMPT)

    added = 0
    lines = []

    # Clean fences
    clean_output = output.replace("```json", "").replace("```jsonl", "").replace("```", "")

    # Try parse as array
    try:
        data = json.loads(clean_output)
        if isinstance(data, list):
            lines = [json.dumps(obj, ensure_ascii=False) for obj in data]
    except Exception:
        for line in clean_output.strip().splitlines():
            line = line.strip().strip(",")
            if not line or line.startswith("Here are"):
                continue
            try:
                json.loads(line)
                lines.append(line)
            except json.JSONDecodeError:
                continue

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")
            added += 1

    print(f"✅ Added {added} samples this round")



if __name__ == "__main__":
  main()