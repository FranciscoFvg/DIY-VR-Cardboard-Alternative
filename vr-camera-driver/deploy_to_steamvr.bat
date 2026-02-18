@echo off
setlocal

set "ROOT=%~dp0"
set "STEAMVR_DIR=%STEAMVR_DIR%"
if "%STEAMVR_DIR%"=="" set "STEAMVR_DIR=C:\Program Files (x86)\Steam\steamapps\common\SteamVR"

if not exist "%STEAMVR_DIR%\bin\win64\vrpathreg.exe" (
  echo SteamVR not found. Set STEAMVR_DIR env var to your SteamVR folder.
  exit /b 1
)

set "DRIVER_DIR=%STEAMVR_DIR%\drivers\cameravr"
set "BIN_DIR=%DRIVER_DIR%\bin\win64"
set "INPUT_DIR=%DRIVER_DIR%\resources\input"

if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"
if not exist "%INPUT_DIR%" mkdir "%INPUT_DIR%"

set "SOURCE_DLL=%ROOT%build\Release\driver_cameravr.dll"
if not exist "%SOURCE_DLL%" set "SOURCE_DLL=%ROOT%build\x64\Release\driver_cameravr.dll"
if not exist "%SOURCE_DLL%" (
  echo driver_cameravr.dll not found. Build Release first.
  exit /b 1
)

copy /Y "%SOURCE_DLL%" "%BIN_DIR%\" >nul
copy /Y "%ROOT%driver\driver.vrdrivermanifest" "%DRIVER_DIR%\driver.vrdrivermanifest" >nul
copy /Y "%ROOT%driver\input\controller_profile.json" "%INPUT_DIR%\controller_profile.json" >nul
copy /Y "%ROOT%driver\controller_profile.json" "%DRIVER_DIR%\controller_profile.json" >nul

"%STEAMVR_DIR%\bin\win64\vrpathreg.exe" adddriver "%DRIVER_DIR%"

echo Driver deployed to %DRIVER_DIR%
exit /b 0
