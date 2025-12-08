# Análisis de Hipótesis - Modelo Predictivo Demanda Agua

Este directorio contiene los scripts y resultados del análisis de hipótesis para validar la eficacia del modelo predictivo de demanda de agua potable en Gran Valparaíso.

## 📁 Estructura

```
analisis_hipotesis/
├── 01_analisis_correlaciones.py      # Análisis de correlaciones entre variables
├── 02_metricas_modelo.py              # Métricas del modelo (RMSE, MAE, MAPE, R²)
├── 03_importancia_variables.py        # Importancia de features del modelo
├── ejecutar_todos.py                  # Script maestro que ejecuta todos los análisis
├── outputs/                           # Resultados generados
│   ├── 01_mapa_calor_correlaciones.png
│   ├── 01_correlaciones_completas.csv
│   ├── 02_metricas_visualizacion.png
│   ├── 02_metricas_modelo.csv
│   ├── 03_top20_variables.png
│   ├── 03_importancia_completa.csv
│   └── reporte_ejecucion.txt
└── README.md
```

## 🚀 Ejecución

### Ejecutar todos los análisis:
```bash
python analisis_hipotesis/ejecutar_todos.py
```

### Ejecutar análisis individuales:
```bash
python analisis_hipotesis/01_analisis_correlaciones.py
python analisis_hipotesis/02_metricas_modelo.py
python analisis_hipotesis/03_importancia_variables.py
```

## 📊 Resultados Principales

### 1. Análisis de Correlaciones

**Variables analizadas:**
- `Qin_m3_hr`: Caudal de entrada (producción)
- `Q_net_m3h`: Demanda real (balance neto)
- `Volumen_Total_m3`: Almacenamiento total en estanques
- `temp`: Temperatura ambiente (°C)
- `HR`: Humedad relativa (%)
- `mmhr`: Precipitación (mm/hr)

**Correlaciones más fuertes:**
1. **temp ↔ HR**: -0.7021 (Fuerte negativa)
   - A mayor temperatura, menor humedad relativa
2. **Qin_m3_hr ↔ temp**: 0.5141 (Moderada positiva)
   - La producción aumenta con la temperatura
3. **Q_net_m3h ↔ temp**: -0.3917 (Débil negativa)
   - La demanda disminuye ligeramente con la temperatura

### 2. Métricas del Modelo

**Modelo:** XGBoost V3.0  
**Conjunto de prueba:** 2,240 registros

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **RMSE** | 426.55 m³/hr | Error cuadrático medio |
| **MAE** | 269.61 m³/hr | Error absoluto medio |
| **MAPE** | 30.23% | Error porcentual absoluto medio |
| **R²** | 0.9903 | Coeficiente de determinación |

**Interpretación:**
- **R² = 0.9903**: El modelo explica el 99.03% de la varianza de los datos
- **RMSE/MAE bajos**: Errores absolutos aceptables para el rango de demanda
- **MAPE = 30.23%**: Error porcentual aceptable dado el contexto operacional

### 3. Importancia de Variables

**Top 10 variables más importantes** (explican el 93% de la importancia total):

1. **Q_net_m3h__ema_win_6h** (26.37%)
   - Media exponencial de demanda en ventana de 6 horas
   - Captura patrones recientes de consumo

2. **periodo_dia_Madrugada** (17.64%)
   - Indicador del periodo del día (00:00-06:00)
   - Patrones distintos según hora del día

3. **cal_hour_cos** (17.32%)
   - Componente coseno de la hora (codificación cíclica)
   - Captura periodicidad horaria

4. **hora** (11.33%)
   - Hora del día (0-23)
   - Patrón temporal principal

5. **Q_net_m3h__diff_168h** (7.77%)
   - Diferencia de demanda respecto a hace 1 semana
   - Captura patrones semanales

**Observación clave:**
- Las features temporales y de demanda histórica dominan la importancia
- Las variables climáticas tienen menor peso individual pero contribuyen al ajuste fino
- El modelo aprende principalmente de patrones temporales y auto-regresivos

## 🎯 Conclusiones

### Validación de la Hipótesis

✅ **El modelo predictivo es eficaz** para predecir la demanda de agua potable:
- **Alta precisión**: R² = 0.9903 indica excelente capacidad predictiva
- **Errores controlados**: RMSE y MAE dentro de rangos aceptables operacionales
- **Patrones identificados**: El modelo captura ciclos horarios, diarios y semanales

### Factores Determinantes

1. **Temporales (dominantes)**
   - Hora del día y periodo del día
   - Patrones semanales (día de la semana)
   - Codificación cíclica horaria

2. **Históricos (auto-regresivos)**
   - Demanda reciente (ventanas de 6-12 horas)
   - Diferencias semanales
   - Medias exponenciales móviles

3. **Climáticos (complementarios)**
   - Temperatura: relación moderada con producción
   - Humedad: correlación inversa con temperatura
   - Precipitación: efecto menor pero presente

### Recomendaciones

1. **Para la tesis:**
   - Incluir gráficos de correlaciones en la sección de resultados
   - Destacar el R² = 0.9903 como evidencia de eficacia
   - Explicar la dominancia de features temporales sobre climáticas

2. **Para mejoras futuras:**
   - Considerar features de eventos especiales (cortes, mantenciones)
   - Evaluar modelos ensemble para reducir MAPE
   - Incorporar predicciones de clima más precisas

3. **Para la operación:**
   - El modelo es confiable para planificación operativa
   - Los errores (MAPE 30%) requieren margen de seguridad en decisiones críticas
   - Actualizar periódicamente con datos recientes

## 📄 Archivos Generados

### Imágenes:
- `01_mapa_calor_correlaciones.png`: Matriz de correlaciones
- `02_metricas_visualizacion.png`: Dashboard de métricas
- `03_top20_variables.png`: Ranking de importancia de variables

### Tablas CSV:
- `01_correlaciones_completas.csv`: Matriz completa de correlaciones
- `02_metricas_modelo.csv`: Tabla de métricas
- `03_importancia_completa.csv`: Importancia de todas las features

### Reporte:
- `reporte_ejecucion.txt`: Log completo de la ejecución

## 🔧 Requisitos

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost
- Archivos de datos en `data/processed/`
- Modelo entrenado en `models/forecasting/`

## 📞 Notas

- Los análisis se ejecutan sobre el dataset sincronizado (15,034 registros)
- El período analizado: 2024-01-01 → 2025-09-30
- Los scripts son independientes y pueden ejecutarse por separado
- Los resultados se guardan automáticamente en `outputs/`

---

**Fecha de generación:** Diciembre 2025  
**Modelo:** XGBoost V3.0  
**Dataset:** dataset_completo_sincronizado.csv
