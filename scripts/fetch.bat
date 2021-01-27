@echo off
setlocal
cd "%~dp0.."

python -m tools.fetch %*

endlocal
