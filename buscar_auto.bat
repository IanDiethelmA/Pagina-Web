@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ============================================================================
echo   BUSCADOR FOCALIZADO Y PROFUNDO MULTI-PORTAL (CHILE)
echo   Portales: Chileautos + MercadoLibre + Autocosmos + Yapo
echo ============================================================================
set /p MARCA="Ingresa la MARCA (ej. NISSAN, TOYOTA, MAZDA, SUBARU): "
set /p MODELO="Ingresa el MODELO (ej. X-TRAIL, RAV4, 3, XV): "
set /p ANIO_MIN="Ano minimo (opcional, ej. 2018, Enter para omitir): "
set /p ANIO_MAX="Ano maximo (opcional, ej. 2022, Enter para omitir): "
set /p PRECIO_MAX="Precio maximo CLP (opcional, ej. 17000000, Enter para omitir): "
set /p KM_MAX="Kilometraje maximo (opcional, ej. 85000, Enter para omitir): "
set /p DEEP="Hacer busqueda profunda ano por ano en Chileautos? (S/N, defecto S): "
set /p PAGINAS="Paginas a profundizar por portal (recomendado 2 a 5): "
if "%PAGINAS%"=="" set PAGINAS=2

set CMD_ARGS=--marca "%MARCA%" --modelo "%MODELO%" --pages %PAGINAS% --max-alerts 6 --recheck-all
if not "%ANIO_MIN%"=="" set CMD_ARGS=%CMD_ARGS% --anio-min %ANIO_MIN%
if not "%ANIO_MAX%"=="" set CMD_ARGS=%CMD_ARGS% --anio-max %ANIO_MAX%
if not "%PRECIO_MAX%"=="" set CMD_ARGS=%CMD_ARGS% --precio-max %PRECIO_MAX%
if not "%KM_MAX%"=="" set CMD_ARGS=%CMD_ARGS% --km-max %KM_MAX%
if /i "%DEEP%"=="S" set CMD_ARGS=%CMD_ARGS% --deep
if "%DEEP%"=="" set CMD_ARGS=%CMD_ARGS% --deep

echo.
echo Escaneando portales en vivo con filtros seleccionados...
python main.py %CMD_ARGS%
echo.
echo Busqueda completada! Revisa las notificaciones en Telegram o data\oportunidades_detectadas.csv
pause
