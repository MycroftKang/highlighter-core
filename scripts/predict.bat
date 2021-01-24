@echo off
setlocal
cd "%~dp0.."

python tools/predict.py %*

endlocal
