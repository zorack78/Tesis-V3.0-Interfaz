# Script de monitoreo - Verifica cada 10 segundos si terminó
Write-Host "🔍 Monitoreando generación de gráficas..." -ForegroundColor Cyan

$targetFiles = @(
    "01_comparacion_metricas_completa_REAL.png",
    "04_comparacion_predicciones_unificada_REAL.png",
    "05_scatter_predicho_vs_real_REAL.png",
    "reporte_metricas_modelos_REAL.txt"
)

$outputDir = "outputs\metricas_ML"
$allFound = $false

while (-not $allFound) {
    Clear-Host
    Write-Host "=" * 70 -ForegroundColor Yellow
    Write-Host "🔍 MONITOR DE GENERACIÓN DE GRÁFICAS ML" -ForegroundColor Cyan
    Write-Host "=" * 70 -ForegroundColor Yellow
    Write-Host ""
    
    $foundCount = 0
    foreach ($file in $targetFiles) {
        $path = Join-Path $outputDir $file
        if (Test-Path $path) {
            $fileInfo = Get-Item $path
            Write-Host "✅ $file" -ForegroundColor Green
            Write-Host "   Tamaño: $([math]::Round($fileInfo.Length/1KB,2)) KB" -ForegroundColor Gray
            Write-Host "   Modificado: $($fileInfo.LastWriteTime)" -ForegroundColor Gray
            $foundCount++
        } else {
            Write-Host "⏳ $file (esperando...)" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
    Write-Host "Progreso: $foundCount / $($targetFiles.Count) archivos completados" -ForegroundColor Cyan
    
    if ($foundCount -eq $targetFiles.Count) {
        $allFound = $true
        Write-Host ""
        Write-Host "=" * 70 -ForegroundColor Green
        Write-Host "✅ ¡GENERACIÓN COMPLETADA!" -ForegroundColor Green
        Write-Host "=" * 70 -ForegroundColor Green
        Write-Host ""
        Write-Host "📁 Archivos generados en: $outputDir" -ForegroundColor Cyan
        break
    }
    
    Write-Host ""
    Write-Host "Próxima verificación en 10 segundos... (Ctrl+C para cancelar)" -ForegroundColor Gray
    Start-Sleep -Seconds 10
}

Read-Host "Presiona Enter para salir"
