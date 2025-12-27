@echo off
REM Run Node Editor sample application

set PYTHONPATH=%~dp0..\..
call %~dp0..\..\..\..\..\.venv\Scripts\python.exe %~dp0main.py %*
