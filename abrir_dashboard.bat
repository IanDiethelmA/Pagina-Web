@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ============================================================================
echo   INICIANDO DASHBOARD INTERACTIVO DE ARBITRAJE AUTOMOTRIZ (STREAMLIT)
echo   URL Local: http://localhost:8501
echo   URL Red Local (Wi-Fi): http://192.168.4.68:8501
echo ============================================================================
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
pause
