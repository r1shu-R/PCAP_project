@echo off
rem Double-click to build and launch the simulator UI inside WSL Ubuntu.
rem The browser opens after a few seconds, once the server is up.
start "" /min cmd /c "timeout /t 8 >nul & start http://localhost:8501"
wsl -d Ubuntu --cd "%~dp0" -- bash ./run_ui.sh
pause
