# Guía Rápida: Uso del Modelo Forecasting V3.0

## 🚀 Inicio Rápido

### 1. Cargar Modelo
```python
import joblib
import pandas as pd
import json

# Cargar modelo y metadatos
modelo = joblib.load('models/forecasting/modelo_forecasting_xgboost.pkl')

# Cargar lista de features (ORDEN IMPORTANTE)
with open('models/forecasting/features.txt') as f:
    features = [line.strip() for line in f]

# Cargar métricas
with open('models/forecasting/metricas.json') as f:
    metricas = json.load(f)
    print(f"RMSE: {metricas['rmse']:.2f} m³/hr")
    print(f"R²: {metricas['r2']:.4f}")

# Cargar umbrales
umbrales = joblib.load('models/forecasting/umbrales.pkl')
print(f"Temp frío: {umbrales['temp_frio']:.1f}°C")
print(f"Temp calor: {umbrales['temp_calor']:.1f}°C")
```

### 2. Preparar Datos para Predicción
```python
# Cargar dataset completo (ya procesado con todas las features)
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Seleccionar últimas 168 horas (para features LAG)
df_reciente = df.tail(168).copy()

# Crear features adicionales (umbrales, períodos)
from entrenar_modelos_forecasting_explicativo import ingenieria_features_adicionales

df_pred = ingenieria_features_adicionales(
    df_reciente, 
    umbrales['temp_frio'], 
    umbrales['temp_calor'],
    umbrales['qin_bajo'],
    umbrales['qin_alto']
)

# Seleccionar features del modelo (47 features)
X_pred = df_pred[features].tail(1)  # Última hora para predecir siguiente

# Predecir
prediccion = modelo.predict(X_pred)[0]
print(f"Predicción Q_net: {prediccion:.2f} m³/hr")
```

---

## 📋 Features Requeridas (47 Total)

### 1. Features Prometedoras Base (31)
Estas vienen del dataset_features_completo.csv y tienen |correlación| > 0.2:

**Target derivado (5):**
- `Q_net_m3h__ema_win_6h` ⭐ **26.4% importancia**
- `Q_net_m3h__lag_168h`
- `Q_net_m3h__diff_168h` ⭐ **7.8% importancia**
- `Q_net_m3h__rolling_6h`
- `Q_net_m3h__rolling_24h`

**Clima y temperatura (21):**
- `clima_temp_c__slope_lin_win_48h` ⭐ (tendencia temperatura)
- `clima_temp_c__max_win_12h`
- `clima_temp_c__min_win_12h`
- `clima_temp_c__std_win_12h`
- `clima_temp_c__ema_win_6h`
- `clima_temp_c__diff_6h`
- `clima_temp_c__diff_12h`
- `clima_temp_c__diff_24h`
- `clima_HR_pct__max_win_12h`
- `clima_HR_pct__min_win_12h`
- `clima_HR_pct__std_win_12h`
- ... (y más rolling windows de temp y HR)

**Calendario (5):**
- `cal_hour_cos` ⭐ **17.3% importancia**
- `cal_hour_sin`
- `cal_day_of_week_cos`
- `cal_day_of_week_sin`
- `cal_month_sin`

### 2. Features Temporales (4)
Generadas en ingenieria_features_adicionales():
- `hora` ⭐ **11.3% importancia**
- `dia_semana`
- `mes`
- `es_fin_de_semana`

### 3. Umbrales Categóricos (8 after one-hot)
One-hot encoding de 3 categorías:
- `temp_nivel_Frio` (temp < 10.5°C)
- `temp_nivel_Normal` (10.5°C ≤ temp ≤ 16.3°C)
- `temp_nivel_Calor` (temp > 16.3°C)
- `periodo_dia_Madrugada` ⭐ **17.6% importancia**
- `periodo_dia_Manana_critica`
- `periodo_dia_Dia`
- `periodo_dia_Noche`
- `periodo_dia_Noche_tardia`

### 4. Interacciones (3)
- `temp_x_hora`
- `delta_temp_6h_x_hora`
- `temp_x_finde`

### 5. Regímenes (1)
- `regimen_equilibrio` (Qin estable 24h)

---

## 🔧 Ejemplos de Uso

### Ejemplo 1: Predicción Próxima Hora
```python
import joblib
import pandas as pd

# Cargar
modelo = joblib.load('models/forecasting/modelo_forecasting_xgboost.pkl')
features = open('models/forecasting/features.txt').read().splitlines()

# Datos últimos registros (necesitas al menos 168h para LAG_168h)
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Última hora
X_ultima = df[features].iloc[-1:]

# Predecir
pred = modelo.predict(X_ultima)[0]

if pred < 0:
    print(f"Predicción: DEMANDA de {abs(pred):.0f} m³/hr")
else:
    print(f"Predicción: RECUPERACIÓN de {pred:.0f} m³/hr")
```

### Ejemplo 2: Predicción con Temperatura Manual
```python
# Supón que tienes pronóstico de temperatura para próxima hora
temp_pronosticada = 18.5  # °C
hr_pronosticada = 65      # %

# Cargar última hora
df_base = df.iloc[-1:].copy()

# Actualizar temperatura y HR
df_base['clima_temp_c'] = temp_pronosticada
df_base['clima_HR_pct'] = hr_pronosticada

# Recalcular features derivadas de temperatura
# (esto requiere historia de 48h para slopes, 12h para max/min, etc.)
# En producción, necesitas pipeline completo de feature engineering

# Por ahora, usar última hora con temp actualizada
X_pred = df_base[features]
pred = modelo.predict(X_pred)[0]
print(f"Con temp {temp_pronosticada}°C: {pred:.0f} m³/hr")
```

### Ejemplo 3: Forecast 24h
```python
import numpy as np

# Para forecast 24h, necesitas:
# 1. Pronóstico meteorológico 24h (temp, HR)
# 2. Propagar features de persistencia (EMA, LAG)

predicciones_24h = []

for hora in range(24):
    # Última observación + hora actual
    X_hora = df[features].iloc[-1:].copy()
    
    # Actualizar hora (circular)
    hora_siguiente = (df['hora'].iloc[-1] + hora + 1) % 24
    X_hora['hora'] = hora_siguiente
    
    # Actualizar temp (de pronóstico meteorológico)
    # temp_forecast = obtener_temp_forecast(hora)
    # X_hora['clima_temp_c'] = temp_forecast
    # ... recalcular features derivadas
    
    # Predecir
    pred = modelo.predict(X_hora)[0]
    predicciones_24h.append(pred)
    
    # Actualizar features de persistencia para siguiente iteración
    # df = pd.concat([df, nueva_prediccion])
    # df = recalcular_features(df)

print(f"Predicciones 24h: {len(predicciones_24h)} horas")
print(f"Demanda total predicha: {abs(sum(predicciones_24h)):.0f} m³")
```

---

## ⚠️ Notas Importantes

### Feature Engineering Completo
Para producción, necesitas el pipeline completo:

```python
from entrenar_modelos_forecasting_explicativo import (
    cargar_features_prometedoras,
    ingenieria_features_adicionales,
    preparar_features_modelo_a
)

# 1. Cargar features prometedoras
features_prometedoras = cargar_features_prometedoras()

# 2. Aplicar ingeniería de features
df_eng = ingenieria_features_adicionales(
    df_raw,
    temp_frio=10.5,
    temp_calor=16.3,
    qin_bajo=10998,
    qin_alto=12614
)

# 3. Preparar features para modelo A
df_modelo, features_finales = preparar_features_modelo_a(
    df_eng,
    features_prometedoras
)

# 4. Predecir
X = df_modelo[features_finales]
predicciones = modelo.predict(X)
```

### Umbrales Críticos
```python
# Temperatura
TEMP_FRIO = 10.5   # Q25 - comportamiento irregular
TEMP_CALOR = 16.3  # Q75 - patrones predecibles

# Hora
HORA_MAX_DEMANDA = 12  # -5,222 m³/hr promedio
HORA_MAX_RECUP = 4     # +5,149 m³/hr promedio

# Períodos
PERIODO_MADRUGADA = (0, 6)      # Estable, baja demanda
PERIODO_MANANA = (7, 9)          # CRÍTICO - alta variabilidad
PERIODO_DIA = (9, 18)            # Más predecible
PERIODO_NOCHE = (18, 22)         # Transición
PERIODO_NOCHE_TARDIA = (22, 24)  # Recuperación
```

### Interpretación Predicciones
```python
def interpretar_prediccion(pred_m3h):
    """Interpreta valor predicho Q_net"""
    if pred_m3h < -3000:
        return "🔴 DEMANDA MUY ALTA - Riesgo déficit"
    elif pred_m3h < -1000:
        return "🟡 DEMANDA ALTA - Monitorear"
    elif pred_m3h < 0:
        return "🟢 DEMANDA NORMAL"
    elif pred_m3h < 1000:
        return "🟢 RECUPERACIÓN LEVE"
    elif pred_m3h < 3000:
        return "🔵 RECUPERACIÓN MODERADA"
    else:
        return "🔵 RECUPERACIÓN FUERTE - Sistema cargando"

# Uso
pred = modelo.predict(X)[0]
print(f"Predicción: {pred:.0f} m³/hr")
print(interpretar_prediccion(pred))
```

---

## 📊 Validación de Predicciones

### Métricas Esperadas
```python
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Comparar predicciones vs reales
mae = mean_absolute_error(y_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
r2 = r2_score(y_real, y_pred)

# Benchmarks del modelo
print(f"MAE actual: {mae:.2f} (esperado ~270 m³/hr)")
print(f"RMSE actual: {rmse:.2f} (esperado ~427 m³/hr)")
print(f"R² actual: {r2:.4f} (esperado ~0.99)")

# Alertas
if mae > 270 * 1.15:  # +15% sobre esperado
    print("⚠️ ALERTA: Error aumentó - considerar reentrenamiento")
if r2 < 0.95:
    print("⚠️ ALERTA: R² bajó - verificar calidad datos")
```

### Monitoreo Continuo
```python
# Guardar predicciones vs reales
log_predicciones = []

for timestamp, pred, real in zip(timestamps, predicciones, reales):
    error = abs(pred - real)
    log_predicciones.append({
        'timestamp': timestamp,
        'prediccion': pred,
        'real': real,
        'error_abs': error,
        'error_pct': error / abs(real) * 100 if real != 0 else 0
    })

df_log = pd.DataFrame(log_predicciones)
df_log.to_csv('logs/predicciones_monitoring.csv', index=False)

# Análisis semanal
error_promedio_semanal = df_log.groupby(
    pd.Grouper(key='timestamp', freq='W')
)['error_abs'].mean()

if error_promedio_semanal.iloc[-1] > 270 * 1.15:
    print("⚠️ Error semanal elevado - revisar")
```

---

## 🔄 Reentrenamiento

### Cuándo Reentrenar
- ✅ **Mensual**: Incorporar datos nuevos del mes anterior
- ⚠️ **Urgente** si error aumenta >15%
- ⚠️ **Urgente** si R² cae <0.95
- ⚠️ **Urgente** tras eventos excepcionales (terremotos, cortes prolongados)

### Script de Reentrenamiento
```bash
# Actualizar datos
python analyze_data.py --update

# Generar features
python src/feature_engineering.py --all

# Reentrenar
python entrenar_modelos_forecasting_explicativo.py

# Validar
python comparar_modelo_baseline.py

# Desplegar (si mejora)
cp models/forecasting/modelo_forecasting_xgboost.pkl models/produccion/
```

---

## 📞 Soporte

**Modelo:** Forecasting V3.0  
**Archivo:** `models/forecasting/modelo_forecasting_xgboost.pkl`  
**Documentación completa:** `RESUMEN_MODELO_FORECASTING_V3.md`  
**Scripts:** `entrenar_modelos_forecasting_explicativo.py`

**Performance:**
- RMSE: 426.55 m³/hr
- MAE: 269.61 m³/hr
- R²: 0.9903
- MAPE: 30.23%

🚀 **Estado: PRODUCCIÓN**
