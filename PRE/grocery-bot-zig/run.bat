@echo off
if "%~1"=="" (
    echo Usage: run.bat ^<websocket-url^>
    echo Example: run.bat wss://game-dev.ainm.no/ws?token=YOUR_TOKEN
    exit /b 1
)
"%~dp0zig-out\bin\grocery-bot.exe" "%~1"
