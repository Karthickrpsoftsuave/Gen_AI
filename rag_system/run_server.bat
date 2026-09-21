@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found. Follow README.md setup steps first.
  pause
  exit /b 1
)
echo Starting Recipe Chatbot Swagger UI server...
echo Open http://localhost:8000/docs in your browser.
.venv\Scripts\python.exe server.py
pause
