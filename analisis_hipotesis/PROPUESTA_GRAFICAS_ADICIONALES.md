# Propuesta de Gráficas Adicionales para Tesis

## Estado Actual ✓
Ya tenemos 3 análisis completos con 9 gráficas:

### Análisis 1: Correlaciones
- ✓ Mapa de calor de correlaciones

### Análisis 2: Métricas del Modelo
- ✓ Gráfico de barras con métricas

### Análisis 3: Importancia de Variables
- ✓ Top 20 variables más importantes

### Análisis 4: Validación Retrospectiva
- ✓ Serie temporal (30 días)
- ✓ Scatter plot predicción vs real
- ✓ Análisis segmentado (4 paneles)

---

## Gráficas Adicionales Propuestas

### PARA ANÁLISIS DESCRIPTIVO (Capítulo: Análisis de Información)

#### 1. **Estadísticas Descriptivas de Variables Clave**
- **Tipo**: Panel de 6 histogramas con estadísticas
- **Variables**: Qin, Q_net, Volumen, temp, HR, mmhr
- **Info**: Media, mediana, desviación estándar, percentiles
- **Utilidad**: Mostrar distribución de los datos

#### 2. **Patrones Temporales de Demanda**
- **Tipo**: 4 subgráficas
  - Demanda promedio por hora del día (24 horas)
  - Demanda promedio por día de la semana
  - Demanda por mes (boxplot)
  - Tendencia temporal (promedio móvil 7 días)
- **Utilidad**: Identificar patrones estacionales y semanales

#### 3. **Análisis de Temperatura vs Demanda**
- **Tipo**: 3 subgráficas
  - Scatter plot: Temperatura vs Qin (color por hora)
  - Demanda promedio por rango de temperatura
  - Demanda horaria por estación del año
- **Utilidad**: Mostrar relación temp-demanda visualmente

#### 4. **Análisis de Volumen en Estanques**
- **Tipo**: 2 subgráficas
  - Serie temporal: Evolución del volumen (2024-2025)
  - Distribución de volumen por hora del día
- **Utilidad**: Mostrar comportamiento del sistema de almacenamiento

---

### PARA ANÁLISIS INFERENCIAL (Capítulo: Prueba de Hipótesis)

#### 5. **Residuos del Modelo**
- **Tipo**: 4 paneles diagnósticos
  - Histograma de residuos (debe ser normal)
  - Q-Q plot (normalidad de residuos)
  - Residuos vs predicciones (homocedasticidad)
  - Residuos vs tiempo (independencia)
- **Utilidad**: Validar supuestos del modelo estadístico

#### 6. **Análisis de Error por Condiciones**
- **Tipo**: 3 subgráficas
  - Error vs hora del día (boxplot)
  - Error vs temperatura (scatter con regresión)
  - Error vs volumen de estanques (scatter)
- **Utilidad**: Identificar en qué condiciones el modelo falla más

#### 7. **Comparación Train vs Validation vs Test**
- **Tipo**: 2 gráficas
  - Tabla comparativa de métricas (R², RMSE, MAE, MAPE)
  - Gráfico de barras agrupadas por métrica
- **Utilidad**: Mostrar que el modelo generaliza bien

#### 8. **Intervalos de Confianza de Predicciones**
- **Tipo**: Serie temporal con bandas
  - Predicciones con intervalos de confianza 95%
  - Valores reales
  - Período: 7 días representativos
- **Utilidad**: Mostrar incertidumbre de las predicciones

---

### PARA SECCIÓN DE INTERPRETABILIDAD

#### 9. **Análisis de Features Temporales vs Autoregresivos**
- **Tipo**: Gráfico de barras apiladas
  - Importancia agrupada por tipo:
    - Features temporales (hora, día, mes)
    - Features autoregresivos (lags, diffs, EMAs)
    - Features climáticos (temp, HR, mmhr)
    - Features de calendario (feriados, eventos)
- **Utilidad**: Mostrar qué tipo de features dominan

#### 10. **SHAP Values (Análisis de Contribución)**
- **Tipo**: 2 gráficas
  - SHAP Summary Plot (top 15 variables)
  - SHAP Dependence Plot (Q_net_m3h__ema_win_6h vs temp)
- **Utilidad**: Mostrar cómo cada variable influye en las predicciones
- **Nota**: Requiere instalar `shap` library

---

## Priorización Recomendada

### ALTA PRIORIDAD (Para tesis inmediata):
1. ✅ **Estadísticas Descriptivas** (#1)
2. ✅ **Patrones Temporales** (#2)
3. ✅ **Residuos del Modelo** (#5)
4. ✅ **Comparación Train/Val/Test** (#7)

### MEDIA PRIORIDAD (Complementarias):
5. **Temperatura vs Demanda** (#3)
6. **Error por Condiciones** (#6)
7. **Features por Tipo** (#9)

### BAJA PRIORIDAD (Opcional):
8. **Volumen en Estanques** (#4)
9. **Intervalos de Confianza** (#8)
10. **SHAP Values** (#10) - requiere librería adicional

---

## Estructura de Carpetas Propuesta

```
analisis_hipotesis/
├── outputs/
│   ├── 01_mapa_calor_correlaciones.png
│   ├── 02_metricas_visualizacion.png
│   ├── 03_top20_variables.png
│   ├── validacion_retrospectiva/
│   │   ├── 01_serie_temporal_validacion.png
│   │   ├── 02_scatter_prediccion_real.png
│   │   └── 03_analisis_segmentos.png
│   ├── descriptivo/  [NUEVO]
│   │   ├── 01_estadisticas_descriptivas.png
│   │   ├── 02_patrones_temporales.png
│   │   ├── 03_temperatura_demanda.png
│   │   └── 04_volumen_estanques.png
│   └── inferencial/  [NUEVO]
│       ├── 01_residuos_modelo.png
│       ├── 02_error_condiciones.png
│       ├── 03_comparacion_datasets.png
│       └── 04_intervalos_confianza.png
```

---

## ¿Cuáles quieres que implemente primero?

Dime qué gráficas necesitas con más urgencia y las creo. Recomiendo:
- **Para análisis descriptivo**: #1, #2, #3
- **Para prueba de hipótesis**: #5, #7

Estas 5 gráficas darían una cobertura completa de ambos capítulos de tu tesis.
