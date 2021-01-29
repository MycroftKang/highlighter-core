@echo off
setlocal
cd "%~dp0"

python -m tools.%*

endlocal
