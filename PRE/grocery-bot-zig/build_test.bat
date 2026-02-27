@echo off
where zig > zig_path.txt 2>&1
echo --- >> zig_path.txt
zig build >> zig_path.txt 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> zig_path.txt
dir /B zig-out\bin\grocery-bot.exe >> zig_path.txt 2>&1
