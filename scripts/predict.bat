@echo off
setlocal
cd "%~dp0.."

python -m tools.predict %*

endlocal
