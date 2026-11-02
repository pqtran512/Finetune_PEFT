@echo off
title AJAC Frontend Build
cd /d "%~dp0frontend"

rem Vite 8 / rolldown can Node 20.19+ hoac 22.12+. Node 20.18 se bo qua native binding.
set "NODE_FALLBACK=%LOCALAPPDATA%\Programs\nodejs-lts"
node -e "const [maj,min]=process.versions.node.split('.').map(Number); const ok=(maj===20&&min>=19)||maj>22||(maj===22&&min>=12); process.exit(ok?0:1)"
if not errorlevel 1 goto node_ok
if not exist "%NODE_FALLBACK%\node.exe" goto node_missing
set "PATH=%NODE_FALLBACK%;%PATH%"
echo [INFO] Node he thong qua cu. Dung %NODE_FALLBACK%
goto node_ok

:node_missing
echo [ERROR] Vite 8 can Node.js 20.19+ hoac 22.12+.
echo         Phien ban hien tai:
node -v
echo         Tai ban LTS: https://nodejs.org/
pause
exit /b 1

:node_ok
if exist "node_modules\@rolldown\binding-win32-x64-msvc\" goto deps_ok
echo [INFO] Cai dat npm dependencies...
call npm install
if errorlevel 1 (
    echo [ERROR] npm install that bai.
    pause
    exit /b 1
)

:deps_ok
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
