@echo off
title Java Code Generator Prototype Launcher
echo ==================================================
echo   Khoi chay Java Code Generator Prototype
echo ==================================================

:: Kiem tra virtual environment 'env'
if not exist "env\Scripts\activate.bat" (
    echo [ERROR] Khong tim thay thu muc moi truong ao "env".
    echo Vui long dam bao thu muc "env" ton tai va da duoc thiet lap nhu trong README.md.
    pause
    exit /b 1
)

echo Kich hoat moi truong ao (env)...
call env\Scripts\activate.bat

:: Kiem tra Flask
python -c "import flask" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Thieu thu vien Flask. Dang cai dat Flask...
    pip install flask
)

:: Mo trinh duyet
echo Dang khoi dong Web Server tren http://127.0.0.1:5000...
echo Trinh duyet se tu dong mo trong giay lat...
start http://127.0.0.1:5000

:: Hugging Face / Torch cache (tránh ghi vào C:\)
set HF_HOME=D:/Temp/.cache/huggingface
set HUGGINGFACE_HUB_CACHE=D:\Temp\.cache\huggingface
set TRANSFORMERS_CACHE=D:/Temp/.cache/huggingface
set HF_HUB_CACHE=D:/Temp/.cache/huggingface/hub
set TORCH_HOME=D:/Temp/.cache/torch
set TEMP=D:\Temp
set TMP=D:\Temp

:: Chay Flask backend
python prototype/app.py

pause
