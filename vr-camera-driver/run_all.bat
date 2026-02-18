@echo off
setlocal

set "ROOT=%~dp0"
set "STEAMVR_DIR=%STEAMVR_DIR%"
if "%STEAMVR_DIR%"=="" set "STEAMVR_DIR=C:\Program Files (x86)\Steam\steamapps\common\SteamVR"

call "%ROOT%deploy_to_steamvr.bat" || exit /b 1

start "SteamVR" "steam://rungameid/250820"

if exist "%STEAMVR_DIR%\bin\win64\vrmonitor.exe" (
  start "SteamVR" "%STEAMVR_DIR%\bin\win64\vrmonitor.exe"
)

set "PYTHON_EXE=%ROOT%..\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

start "HandTracker" /D "%ROOT%" "%PYTHON_EXE%" "%ROOT%hand_tracker_gui.py"

echo Started SteamVR driver and hand_tracker_gui.py
exit /b 0
