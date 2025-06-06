@echo off
chcp 65001 >nul
echo 正在启动TSP求解器...
echo.

cd /d "%~dp0"
.\.venv\Scripts\python.exe -m src.main

echo.
echo 程序执行完成，按任意键退出...
pause >nul
