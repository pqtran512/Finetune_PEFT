@echo off
title AJAC Frontend (Vite)
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

echo.
echo  Vite UI:  http://127.0.0.1:5173
echo  Can Flask API tren http://127.0.0.1:5000
echo  (chay run_prototype_mock.bat hoac run_prototype.bat o terminal khac)
echo.
call npm run dev
pause
