@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ============================================================================
echo   EJECUTANDO SISTEMA AUTONOMO DE ARBITRAJE AUTOMOTRIZ (CHILE)
echo ============================================================================
python main.py
echo.
echo Proceso finalizado. Revisa tu Telegram o el archivo data\oportunidades_detectadas.csv
pause
