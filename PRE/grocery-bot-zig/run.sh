#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: ./run.sh <websocket-url>"
    echo "Example: ./run.sh 'wss://game-dev.ainm.no/ws?token=YOUR_TOKEN'"
    exit 1
fi
"$(dirname "$0")/zig-out/bin/grocery-bot.exe" "$1"
