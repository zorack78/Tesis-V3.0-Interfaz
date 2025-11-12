# REPORTE: Features Climáticas Avanzadas para ML

**Fecha de generación:** 2025-11-12 07:55:52  
**Dataset:** Gran Valparaíso - Sistema ESVAL  
**Periodo:** 2024-01-01 00:00:00+00:00 → 2025-09-30 23:00:00+00:00  
**Registros totales:** 15,336

---

## 🎯 Objetivo

Identificar y crear features climáticas avanzadas que capturen eventos extremos 
(olas de calor, frentes fríos, tormentas) para mejorar la capacidad predictiva 
del modelo de demanda de agua potable.

---

## 📊 Umbrales Climáticos Definidos

### Temperatura (°C)
- **Muy fría**: < 8°C
- **Fría**: < 12°C (validado: 53.5% de casos con volumen negativo)
- **Fresca**: < 15°C
- **Cálida**: > 22°C
- **Muy cálida**: > 26°C
- **Ola de calor**: > 30°C

### Humedad Relativa (%)
- **Muy baja**: < 30%
- **Baja**: < 50%
- **Alta**: > 70%
- **Muy alta**: > 85%
- **Saturación**: > 95%

### Precipitación (mm/hr)
- **Llovizna**: < 0.5 mm/hr
- **Lluvia moderada**: 2.0-5.0 mm/hr
- **Lluvia fuerte**: 5.0-10.0 mm/hr
- **Torrencial**: > 10.0 mm/hr

---

## 🔧 Features Creadas

### Total: **78 features climáticas avanzadas**

#### 1️⃣ Temperatura (31 features)
- Rangos extremos (muy_fria, fria, calida, muy_calida, ola_calor)
- LAGs: 1h, 2h, 3h, 6h, 12h, 24h, 48h
- Rolling statistics: mean, std, max, min (6h, 12h, 24h)
- Cambios/diferencias: 1h, 3h, 6h, 12h, 24h
- Amplitud térmica diaria (max - min en 24h)
- Detección frentes fríos (caída >5°C en 6h/12h)
- Detección ola de calor (temp >26°C sostenida 24h)

#### 2️⃣ Humedad (18 features)
- Rangos extremos (muy_baja, baja, alta, muy_alta)
- LAGs: 1h, 2h, 3h, 6h, 12h, 24h
- Rolling statistics: mean, std (6h, 24h)
- Cambios bruscos (>15% en 1h)

#### 3️⃣ Precipitación (18 features)
- Clasificación intensidad (sin_lluvia, ligera, moderada, fuerte)
- LAGs: 1h, 2h, 3h, 6h, 12h
- Acumulados: 3h, 6h, 12h, 24h, 48h
- Intensidad máxima reciente (6h, 24h)
- Horas desde última lluvia significativa
- Detección tormenta (>20mm en 6h)

#### 4️⃣ Combinaciones Climáticas (12 features)
- Sensación térmica (temp ajustada por HR)
- Calor seco / Calor húmedo
- Frío seco / Frío húmedo
- Lluvia fría
- Niebla (HR>85%, temp<15°C, sin lluvia)
- Condiciones ideales (18-24°C, HR 50-70%, sin lluvia)
- Índice condiciones adversas (suma de extremos)
- Interacciones (temp × HR)

---

## 💡 Hallazgos Clave

### ✅ Features con Mayor Potencial Predictivo

Las features que muestran mayor correlación con la demanda de agua son:

1. **LAGs de temperatura** - Capturan patrones de consumo retardados
2. **Olas de calor sostenidas** - Incrementos significativos en demanda
3. **Frentes fríos** - Reducciones bruscas en consumo
4. **Sensación térmica** - Mejor predictor que temperatura sola
5. **Condiciones compuestas** - Calor seco/húmedo tienen impactos diferentes
6. **Acumulados de lluvia** - Efecto prolongado en la demanda

### 🎯 Recomendaciones para Modelado ML

1. **Usar todas las features de LAGs** - Capturan dependencia temporal
2. **Incluir features de eventos extremos** - Mejoran robustez en condiciones atípicas
3. **Probar interacciones** - Temp × HR, Temp × Lluvia, etc.
4. **Features rolling** - Suavizan ruido y capturan tendencias de corto plazo
5. **Binarias de eventos** - Simplifican decisiones del modelo

---

## 📁 Archivos Generados

- **Dataset completo**: `data/processed/data_processed_complete_with_climate_advanced.csv`
- **Lista de features**: `outputs/analisis_climatico/features_climaticas_avanzadas.txt`
- **Este reporte**: `outputs/analisis_climatico/REPORTE_FEATURES_CLIMATICAS_AVANZADAS.md`

---

## 🚀 Próximos Pasos

1. Reentrenar modelo XGBoost con nuevo conjunto de features
2. Evaluar importancia de features con SHAP values
3. Realizar feature selection para optimizar performance
4. Validar en periodo de testing (Agosto-Sept 2025)
5. Comparar MAPE/R² con modelo anterior

---

**Generado automáticamente por:** `analisis_features_climaticas_avanzadas.py`
