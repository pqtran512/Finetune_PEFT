@echo off
title AJAC Frontend Build
cd /d "%~dp0frontend"

if not exist "node_modules\" (
    echo [INFO] Cai dat npm dependencies...
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install that bai.
        pause
        exit /b 1
    )
)

echo Building SPA -^> frontend\dist ...
call npm run build
if errorlevel 1 (
    echo [ERROR] Build that bai.
    pause
    exit /b 1
)

echo.
echo OK. Flask se phuc vu SPA tai http://127.0.0.1:5000
echo UI cu: http://127.0.0.1:5000/legacy
pause
