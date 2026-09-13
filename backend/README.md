# DermaAI: FastAPI Backend

This directory contains the FastAPI microservice that wraps the PyTorch inference logic.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The backend also depends on the `ai` module, so ensure you have installed the `ai/requirements.txt` dependencies as well).*

2. Start the server:
   ```bash
   python main.py
   # or
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## Endpoints

- `GET /api/health`: Returns the status of the API and underlying ML model.
- `POST /api/analyze`: Accepts `multipart/form-data` with an `image` file and returns the prediction JSON.

## Frontend Connection
To connect the Next.js frontend to this real backend, set the following environment variables in the frontend's `.env.local`:
```
NEXT_PUBLIC_AI_MODE=real
NEXT_PUBLIC_AI_API_URL=http://localhost:8000
```
