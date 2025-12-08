@echo off
echo.
echo ======================================================================
echo LIMPIEZA DE ARCHIVOS CON DATOS SINTETICOS
echo ======================================================================
echo.
echo Este script eliminara archivos generados con datos hardcodeados
echo o predicciones sinteticas que NO son aptos para rigor cientifico.
echo.
echo ARCHIVOS A ELIMINAR:
echo   - 01_comparacion_metricas_completa.png (sin REAL)
echo   - 02_*_ranking.png (4 archivos)
echo   - 03_prediccion_vs_real_*.png (series temporales fake)
echo   - 04_comparacion_predicciones_unificada.png (datos sinteticos)
echo   - 05_scatter_predicho_vs_real.png (sin REAL)
echo   - reporte_metricas_modelos.txt (sin REAL)
echo.
echo ARCHIVOS A CONSERVAR:
echo   - *_REAL.png (todos los que tienen sufijo REAL)
echo   - predicciones_reales.csv
echo   - metricas_reales.json
echo.
pause

cd outputs\metricas_ML

echo.
echo Eliminando archivos sinteticos...
echo.

del 01_comparacion_metricas_completa.png 2>nul && echo [OK] 01_comparacion_metricas_completa.png
del 02_1_r2_ranking.png 2>nul && echo [OK] 02_1_r2_ranking.png
del 02_2_rmse_ranking.png 2>nul && echo [OK] 02_2_rmse_ranking.png
del 02_3_mae_ranking.png 2>nul && echo [OK] 02_3_mae_ranking.png
del 02_4_mape_ranking.png 2>nul && echo [OK] 02_4_mape_ranking.png
del 03_prediccion_vs_real_*.png 2>nul && echo [OK] 03_prediccion_vs_real_*.png
del 04_comparacion_predicciones_unificada.png 2>nul && echo [OK] 04_comparacion_predicciones_unificada.png
del 05_scatter_predicho_vs_real.png 2>nul && echo [OK] 05_scatter_predicho_vs_real.png
del reporte_metricas_modelos.txt 2>nul && echo [OK] reporte_metricas_modelos.txt

echo.
echo ======================================================================
echo LIMPIEZA COMPLETADA
echo ======================================================================
echo.
echo Archivos restantes (solo con datos reales):
echo.
dir *.png *.txt *.csv *.json /b

echo.
echo ======================================================================
pause
