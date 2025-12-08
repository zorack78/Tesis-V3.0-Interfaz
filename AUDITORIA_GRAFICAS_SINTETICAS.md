# 🔍 AUDITORÍA DE GRÁFICAS - DATOS SINTÉTICOS vs REALES

## Fecha de Auditoría
**05 de Diciembre, 2025**

---

## 📊 Resumen Ejecutivo

Durante la revisión del proyecto, se identificó que varios scripts de generación de gráficas utilizaban **datos hardcodeados y predicciones sintéticas** en lugar de predicciones reales de los modelos. Esto compromete el rigor científico necesario para una tesis.

### ⚠️ Problema Principal
- **Script**: `generar_graficas_metricas_rapido.py`
- **Método**: `METRICAS_EJEMPLO` con valores hardcodeados (línea 29-47)
- **Método**: `generar_datos_sinteticos_predicciones()` usando `np.random.normal()` (línea 180-210)
- **Impacto**: 9 archivos de gráficas generadas con datos NO reales

---

## 📋 Tabla Comparativa de Scripts

| Script | Tipo de Datos | Método | Archivos Generados | Estado |
|--------|---------------|--------|-------------------|--------|
| `generar_graficas_metricas_rapido.py` | ❌ Sintéticos | `np.random.normal()` + hardcode | 9 archivos (.png) | ⚠️ NO USAR |
| `generar_graficas_metricas_simple.py` | ✅ Reales (métricas) | Lee de interfaz Gradio | 2 archivos (_REAL) | ✅ APTO |
| `generar_graficas_metricas_reales.py` | ❌ No funcional | Intento fallido entrenar | Ninguno | ⚠️ ABANDONADO |
| `generar_graficas_metricas_ml.py` | ⚠️ Por verificar | Carga modelos | 2+ archivos | ⚠️ VERIFICAR |
| `generar_scatter_plot_real.py` | ✅ Reales (predicciones) | Lee CSV exportado | 2 archivos (_REAL) | ✅ APTO |

---

## 🎯 Archivos por Estado

### ✅ APTOS PARA TESIS (Datos Reales)

| Archivo | Fuente | Contenido | N° Registros |
|---------|--------|-----------|--------------|
| `01_comparacion_metricas_completa_REAL.png` | Interfaz Gradio | Métricas R², RMSE, MAE, MAPE | N/A (resumen) |
| `05_scatter_predicho_vs_real_REAL.png` | predicciones_reales.csv | Scatter plots 3 modelos | 2,256 |
| `reporte_metricas_modelos_REAL.txt` | Interfaz Gradio | Tabla de métricas | N/A |
| `reporte_scatter_plot_REAL.txt` | predicciones_reales.csv | Estadísticas descriptivas | 2,256 |
| `predicciones_reales.csv` | Interfaz comparación ML | Predicciones test set | 2,256 × 5 cols |
| `metricas_reales.json` | Interfaz comparación ML | Metadata y métricas | N/A |

**Total: 6 archivos** ✅

---

### ❌ NO APTOS PARA TESIS (Datos Sintéticos)

| Archivo | Problema | Script Origen |
|---------|----------|---------------|
| `01_comparacion_metricas_completa.png` | Métricas hardcodeadas incorrectas | generar_graficas_metricas_rapido.py |
| `02_1_r2_ranking.png` | Métricas hardcodeadas | generar_graficas_metricas_rapido.py |
| `02_2_rmse_ranking.png` | Métricas hardcodeadas | generar_graficas_metricas_rapido.py |
| `02_3_mae_ranking.png` | Métricas hardcodeadas | generar_graficas_metricas_rapido.py |
| `02_4_mape_ranking.png` | Métricas hardcodeadas | generar_graficas_metricas_rapido.py |
| `03_prediccion_vs_real_XGBoost.png` | Predicciones sintéticas (random) | generar_graficas_metricas_rapido.py |
| `03_prediccion_vs_real_RandomForest.png` | Predicciones sintéticas (random) | generar_graficas_metricas_rapido.py |
| `03_prediccion_vs_real_LightGBM.png` | Predicciones sintéticas (random) | generar_graficas_metricas_rapido.py |
| `04_comparacion_predicciones_unificada.png` | Predicciones sintéticas (random) | generar_graficas_metricas_rapido.py |
| `05_scatter_predicho_vs_real.png` | Predicciones sintéticas (random) | generar_graficas_metricas_rapido.py |
| `reporte_metricas_modelos.txt` | Métricas hardcodeadas | generar_graficas_metricas_rapido.py |

**Total: 11 archivos** ❌

---

## 🔬 Detalle de Métricas: Real vs Hardcodeado

### RandomForest - DISCREPANCIA DETECTADA

| Fuente | R² | RMSE | MAE | MAPE |
|--------|----|----|-----|------|
| **Interfaz Gradio (REAL)** | 0.9842 | 548 | **295** | 3.15% |
| **Hardcodeado (FAKE)** | 0.9897 | 442 | **278** | 2.75% |
| **Diferencia** | +0.0055 | -106 | **-17** | -0.40% |

❌ **Problema**: El hardcoded muestra un modelo MEJOR de lo que realmente es.

### XGBoost V3.0 - CORRECTO

| Fuente | R² | MAE |
|--------|----|----|
| Interfaz Gradio (REAL) | 0.9905 | 269 |
| Hardcodeado | 0.9905 | 269 |
| Diferencia | 0.0000 | 0 |

✅ **Estado**: Coincide

### LightGBM - CORRECTO

| Fuente | R² | MAE |
|--------|----|----|
| Interfaz Gradio (REAL) | 0.9923 | 261 |
| Hardcodeado | 0.9923 | 261 |
| Diferencia | 0.0000 | 0 |

✅ **Estado**: Coincide

---

## 📍 Ubicación de Problemas en Código

### generar_graficas_metricas_rapido.py

```python
# Línea 29-47: Valores hardcodeados
METRICAS_EJEMPLO = {
    'RandomForest': {
        'r2': 0.9897,      # ❌ INCORRECTO - Real: 0.9842
        'mae': 278,        # ❌ INCORRECTO - Real: 295
        # ...
    }
}

# Línea 180-210: Generación de datos sintéticos
def generar_datos_sinteticos_predicciones(n_points=168):
    """Genera datos sintéticos de predicciones para demostración"""
    
    # ❌ PROBLEMA: Usa np.random.normal() en lugar de predicciones reales
    error_xgb = np.random.normal(0, 269, n_points)
    predicciones['XGBoost V3.0'] = y_test + error_xgb
    
    error_rf = np.random.normal(0, 278, n_points)  # ❌ MAE incorrecto
    predicciones['RandomForest'] = y_test + error_rf
    # ...
```

---

## 🛠️ Solución Implementada

### 1. Modificación de Interfaz
**Archivo**: `interfaz_planificacion_qin_v1.py`
**Cambio**: Agregado método `_exportar_predicciones_reales()` (línea ~1868)

```python
def _exportar_predicciones_reales(self, predicciones, y_real, 
                                  timestamps, metricas):
    """
    Exporta predicciones reales de los modelos para análisis científico.
    """
    # Exporta a CSV: predicciones_reales.csv
    # Exporta a JSON: metricas_reales.json
```

**Resultado**: Cada ejecución de comparación exporta automáticamente:
- `predicciones_reales.csv` (2,256 registros × 5 columnas)
- `metricas_reales.json` (metadata completo)

### 2. Nuevo Script de Scatter Plot
**Archivo**: `generar_scatter_plot_real.py`
**Función**: Lee CSV exportado y genera scatter plots con datos reales

**Características**:
- Busca columnas dinámicamente (maneja sufijos como "(Actual)")
- Calcula R² desde métricas JSON o recalcula si falta
- Genera estadísticas descriptivas
- 100% datos reales del test set

---

## 📦 Scripts de Utilidad Creados

| Script | Función | Uso |
|--------|---------|-----|
| `auditoria_graficas.py` | Analiza todos los scripts y archivos | Auditoría completa |
| `limpiar_graficas_sinteticas.bat` | Elimina archivos sintéticos | Limpieza de directorio |
| `monitor_y_generar.py` | Monitorea CSV y genera scatter auto | Automatización |
| `ejecutar_scatter_plot.bat` | Ejecuta generación de scatter | Acceso rápido |

---

## 🎯 Procedimiento para Gráficas Reales

### Flujo Completo

```
1. python interfaz_planificacion_qin_v1.py
   └─> Inicia interfaz en http://127.0.0.1:7867

2. Interfaz Gradio → Tab "🔬 Comparación Modelos ML"
   └─> Click "Ejecutar Comparación" (2-3 minutos)
   └─> Exporta automáticamente:
       • predicciones_reales.csv
       • metricas_reales.json

3. python generar_scatter_plot_real.py
   └─> Lee CSV exportado
   └─> Genera:
       • 05_scatter_predicho_vs_real_REAL.png
       • reporte_scatter_plot_REAL.txt
```

### Tiempo Total
- Comparación: 2-3 minutos
- Generación scatter: 5-10 segundos
- **Total: ~3 minutos**

---

## 📊 Información del Test Set

| Propiedad | Valor |
|-----------|-------|
| **Periodo** | 26 Jun 2025 22:00 → 30 Sep 2025 23:00 |
| **N° Registros** | 2,256 |
| **Frecuencia** | Horaria |
| **Distribución** | Jun: 95 (4.2%), Jul: 733 (32.5%), Ago: 723 (32.0%), Sep: 705 (31.3%) |
| **División** | Train: 70%, Val: 15%, Test: 15% |
| **Estación** | Invierno/Primavera (hemisferio sur) |

---

## ✅ Checklist de Validación

Para asegurar rigor científico en gráficas de tesis:

- [ ] Archivo tiene sufijo `_REAL` en el nombre
- [ ] Proviene de `predicciones_reales.csv` o interfaz Gradio
- [ ] Métricas coinciden con tab "🔬 Comparación Modelos ML"
- [ ] Test set: 2,256 registros (Jun 26 - Sep 30, 2025)
- [ ] R² de modelos:
  - [ ] XGBoost: 0.9905
  - [ ] RandomForest: 0.9842
  - [ ] LightGBM: 0.9923
- [ ] No usa `np.random` ni `METRICAS_EJEMPLO`
- [ ] Archivo incluido en sección "✅ CONSERVAR" de auditoría

---

## 📎 Referencias Cruzadas

### Archivos Relacionados
- `COMO_PROBAR.md` - Instrucciones de prueba
- `SUGERENCIAS_PRE_GRADIO.md` - Consideraciones de interfaz
- `README.md` - Documentación general

### Issues Resueltos
1. ✅ Scatter plot mostraba R²=0.0000 para XGBoost
   - **Causa**: Nombre del modelo en JSON tenía sufijo "(Actual)"
   - **Solución**: Búsqueda por coincidencia parcial
   
2. ✅ Métricas inconsistentes entre fuentes
   - **Causa**: Scripts usaban hardcoded en lugar de datos reales
   - **Solución**: Export automático desde interfaz + nuevo script

3. ✅ Fechas incorrectas en gráficas (Nov-Dec en lugar de Jun-Sep)
   - **Causa**: `datetime.now()` en lugar de fechas del test set
   - **Solución**: Usar `pd.Timestamp('2025-09-30 23:00:00')`

---

## 🔒 Conclusión

**Estado Actual**: ✅ RESUELTO

- **Archivos aptos para tesis**: 6 archivos con datos reales
- **Archivos a eliminar**: 11 archivos con datos sintéticos
- **Flujo automatizado**: Implementado y verificado
- **Rigor científico**: Garantizado con predicciones reales

**Recomendación**: Ejecutar `limpiar_graficas_sinteticas.bat` para eliminar archivos obsoletos y conservar solo los aptos para tesis.

---

**Documento generado**: 05/12/2025
**Autor**: Sistema de Planificación Qin
**Versión**: 1.0
