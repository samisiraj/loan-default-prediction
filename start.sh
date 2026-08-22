#!/bin/sh

echo "Starting FastAPI..."
uv run uvicorn src.api:app --host 0.0.0.0 --port 8080 &

echo "Starting Gradio..."
uv run python src/app.py