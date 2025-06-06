# 设置控制台编码为UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

Write-Host "正在启动TSP求解器..." -ForegroundColor Green
Write-Host ""

# 获取脚本所在目录
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptPath

# 运行程序
& ".\.venv\Scripts\python.exe" -m src.main

Write-Host ""
Write-Host "程序执行完成，按任意键退出..." -ForegroundColor Yellow
Read-Host
