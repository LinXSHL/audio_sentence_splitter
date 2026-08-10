@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m audio_sentence_splitter --gui
) else (
  python -m audio_sentence_splitter --gui
)
if errorlevel 1 pause
endlocal
