@echo off
setlocal
cd "%~dp0.."

python tools/naive.py %*

endlocal
