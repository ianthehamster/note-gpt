# note-gpt

1. Run `python -m venv .venv` and `.\.venv\Scripts/activate`
2. Run `python -m pip install --upgrade pip` and `pip install -r requirements.txt`
3. Run `uvicorn app.main:app --reload --port 7860 --host 0.0.0.0`
4. Open the Swagger UI with https://vigilant-space-spoon-5wjqgg96r9v2j96-7860.app.github.dev/docs or localhost:7860/docs to test the FastAPI backend
