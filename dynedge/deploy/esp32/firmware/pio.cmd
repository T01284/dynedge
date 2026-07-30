@echo off
REM Prefer PlatformIO penv (Python 3.11). System Python 3.12 pio.exe breaks
REM pioarduino littlefs/fatfs native extensions on Windows.
REM Force UTF-8 so esptool progress bars (░) don't crash under GBK.
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "PIO=%USERPROFILE%\.platformio\penv\Scripts\pio.exe"
if not exist "%PIO%" set "PIO=pio"
"%PIO%" %*
