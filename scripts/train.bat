@echo off
setlocal
cd "%~dp0.."

call scripts/prepare.bat
call .venv\Scripts\activate

python tools/train.py %*

endlocal
