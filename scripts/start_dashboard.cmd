@echo off
cd /d "%~dp0\.."
call venv\Scripts\activate.bat
streamlit run dashboard/main.py