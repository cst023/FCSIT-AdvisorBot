# FCSIT AdvisorBot RAG Backend

This repository provides the backend for the FCSIT AdvisorBot.

## Setup and usage instructions:

You can try it in two ways:

1. CLI chat mode (run `vector_rag.py` directly)
2. Local API server mode (run FastAPI with Uvicorn) for frontend integration (for example, a Flutter app)

### Prerequisites

1. Python 3.10+ installed
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Get an NVIDIA NIM API key: https://build.nvidia.com/settings/api-keys
4. Create a `.env` file in the project root and add your key:

```env
NVIDIA_NIM_API=your_api_key_here
```

### Option 1: Run Chatbot in CLI

Use this mode if you want to quickly test the chatbot in terminal.

```bash
python vector_rag.py
```

Then type your questions in the terminal.
Type `-1` to exit.

### Option 2: Run Local API Server

Use this mode if you want to connect a frontend app (such as Flutter) to the chatbot backend.

Start the server locally:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Health check endpoint:

- `GET /health`

Chat endpoint:

- `POST /query`

## Notes

- If you are testing with your Flutter frontend, start this backend server first.
- This repository only contains the backend; the Flutter app repository is separate at: [AdvisorBot Mobile App](https://github.com/cst023/FCSIT-AdvisorBot-Mobile-App)
