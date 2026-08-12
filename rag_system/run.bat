@echo off
REM Week 3 M2 Set B - RAG evaluation launcher (double-click to run)
cd /d "%~dp0"
.venv\Scripts\python evaluate.py
echo.
echo Done - results.md has been regenerated. Open it in any Markdown viewer.
pause
