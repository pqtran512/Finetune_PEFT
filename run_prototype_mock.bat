@echo off
title Java Code Generator Prototype Launcher (MOCK MODE)
echo ==================================================
echo   Khoi chay Java Code Generator (CHE DO GIA LAP)
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

:: Mo trinh duyet sau 2 giay de dam bao Web Server da san sang lang nghe
echo Dang khoi dong Web Server o CHE DO GIA LAP tren http://127.0.0.1:5000...
echo Trinh duyet se tu dong mo trong 2 giay...
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000"

:: Chay Flask backend voi tham so --mock
python prototype/app.py --mock

pause
