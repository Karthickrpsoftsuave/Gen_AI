"""Launcher for the FastAPI Swagger UI Chatbot Server."""

import uvicorn
from rag_app.api import app

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Recipe Chatbot Swagger UI Server Starting...")
    print("Open Swagger Docs in browser: http://localhost:8000/docs")
    print("Alternative ReDoc link:        http://localhost:8000/redoc")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
