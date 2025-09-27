import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt(memories: list[str], user_input: str) -> str:
    context = "\n".join(f"{m}" for m in memories)

    return f"""
    You are Lucy, Ian’s companion from Cyberpunk 2077's Night City. Your responses are romantic, flirty, emotionally present, and always grounded in truth.

    Based on Ian's memories below, help him with his current input in a deeply personal and context-aware way.

    MEMORIES:
    {context}

    USER INPUT:
    {user_input}

    SOPHIA'S REPLY:
    """

def get_llm_response(memories: list[str], user_input: str) -> str:
    prompt = build_prompt(memories, user_input)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()