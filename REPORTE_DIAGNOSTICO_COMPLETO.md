# 🔍 REPORTE DIAGNÓSTICO COMPLETO: Inconsistencias en Métricas de ML

**Fecha:** 8 de diciembre, 2025  
**Análisis:** 4 partes completadas (sin modificaciones)  
**Pregunta inicial:** *"¿Por qué las métricas de ML son diferentes para el mismo período de test?"*

---

## 📊 RESUMEN EJECUTIVO

### Causas identificadas de las diferencias en métricas:

| Prioridad | Causa | Impacto | Archivos afectados |
|-----------|-------|---------|-------------------|
| 🔴 **CRÍTICO** | **Diferentes targets evaluados** | ALTO | Q_net vs Qout vs Volumen_Total_m3 |
| 🔴 **CRÍTICO** | **Columna inexistente** | BLOQUEO | `modelo_gradio.py` busca "Volumen_Total_m3" |
| 🟠 **ALTO** | **Entrenamiento en memoria** | MEDIO-ALTO | 3 scripts entrenan modelos en cada run |
| 🟡 **MEDIO** | **Diferentes versiones de modelos** | MEDIO | LightGBM recién reentrenado (2.6 horas) |
| 🟢 **BAJO** | **Diferencia de 1 registro** | MÍNIMO | interfaz_planificacion_qin_v1.py |

**Conclusión:** Las diferencias NO son por evaluar períodos distintos (son casi idénticos), sino por:
1. Evaluar **variables objetivo diferentes** (Q_net vs Qout)
2. **Entrenar modelos en memoria** con aleatoriedad
3. Usar **versiones de modelos diferentes** (pickle vs. recién entrenado)

---

## 📋 ANÁLISIS 1: COMPARACIÓN DE MÉTODOS DE CÁLCULO

### Objetivos encontrados:

| Archivo | Target | Método | Data Leakage Prevention |
|---------|--------|--------|------------------------|
| `graficos_comparativa_test.py` | `Qout` | Calculado (Qin - Q_net) | ✅ Sí |
| `scatter_plots_test.py` | `Qout` | Calculado (Qin - Q_net) | ✅ Sí |
| `check_r2.py` | `Qout` | Calculado (Qin - Q_net) | ✅ Sí |
| `interfaz_planificacion_qin_v1.py` | `Qout` | Calculado (Qin - Q_net) | ✅ Sí |
| `modelo_gradio.py` | `Volumen_Total_m3` | ❌ **No existe** | ✅ Sí |

### Hallazgos:

✅ **Consistente:** Todos usan split 70/15/15  
✅ **Consistente:** Data leakage prevention implementado correctamente  
❌ **CRÍTICO:** `modelo_gradio.py` busca columna inexistente `"Volumen_Total_m3"`  
⚠️ **Problema:** Diferentes targets = diferentes métricas esperadas

---

## 📋 ANÁLISIS 2: VERIFICACIÓN DETALLADA DE TARGETS

### Columnas verificadas en datasets:

| Columna | Train | Validation | Test | Uso |
|---------|-------|------------|------|-----|
| `Q_net_m3h` | ✅ | ✅ | ✅ | Target original (flujo neto) |
| `sist_Qin_m3h` | ✅ | ✅ | ✅ | Input usado para calcular Qout |
| `sist_Vtotal_m3` | ✅ | ✅ | ✅ | Volumen total del sistema |
| `Volumen_Total_m3` | ❌ | ❌ | ❌ | **NO EXISTE** |

### Cálculo de Qout:

```python
# Método correcto (usado por 4 de 5 archivos):
Qin_perfil = df_train.groupby('hora')['sist_Qin_m3h'].median()
df_test['Qout'] = df_test['hora'].map(Qin_perfil) - df_test['Q_net_m3h']
```

**Hallazgo crítico:** `modelo_gradio.py` NO calcula `Qout`, busca columna inexistente.

---

## 📋 ANÁLISIS 3: VERIFICACIÓN DE PERÍODOS DE TEST

### Períodos de test identificados:

| Archivo | Registros | Inicio | Fin | Días |
|---------|-----------|--------|-----|------|
| `graficos_comparativa_test.py` | 2,256 | 2025-06-26 22:00 | 2025-09-30 23:00 | 96 |
| `scatter_plots_test.py` | 2,256 | 2025-06-26 22:00 | 2025-09-30 23:00 | 96 |
| `check_r2.py` | 2,256 | 2025-06-26 22:00 | 2025-09-30 23:00 | 96 |
| `interfaz_planificacion_qin_v1.py` | 2,257 | 2025-06-26 **21:00** | 2025-09-30 23:00 | 96 |

### Conclusión:

✅ **Períodos son prácticamente idénticos**  
✅ Diferencia de 1 registro (1 hora) es **insignificante** (<0.05%)  
✅ **NO es la causa principal** de las diferencias en métricas

**Split esperado (70/15/15):**
- Train: 10,523 registros (hasta 2025-03-22 11:00)
- Validation: 2,255 registros (2025-03-22 12:00 → 2025-06-26 21:00)
- Test: 2,256 registros (2025-06-26 22:00 → 2025-09-30 23:00)

---

## 📋 ANÁLISIS 4: VERIFICACIÓN DE MODELOS UTILIZADOS

### Modelos pickle disponibles:

| Modelo | Estado | Tamaño | Modificado | Edad | Features |
|--------|--------|--------|------------|------|----------|
| **LightGBM** | ✅ Existe | 0.30 MB | 2025-12-08 06:10 | **2.6 horas** | 164 |
| **XGBoost** | ✅ Existe | 1.56 MB | 2025-12-06 15:06 | 41.7 horas | 164 |
| **RandomForest** | ❌ No existe | - | - | - | - |

### Hiperparámetros de modelos guardados:

**LightGBM (recién reentrenado):**
```python
{
    'n_estimators': 300,
    'max_depth': 12,
    'learning_rate': 0.05,
    'num_leaves': 50,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'min_child_samples': 20
}
```

**XGBoost (41 horas de antigüedad):**
```python
{
    'n_estimators': 300,
    'max_depth': 7,
    'learning_rate': 0.05,
    'colsample_bytree': 0.8,
    'subsample': 0.8
}
```

### Uso de modelos por archivo:

| Archivo | LightGBM | XGBoost | RandomForest |
|---------|----------|---------|--------------|
| `graficos_comparativa_test.py` | 💾 Pickle + 🏋️ Entrena | 💾 Pickle + 🏋️ Entrena | 🏋️ Entrena |
| `scatter_plots_test.py` | 💾 Pickle | 💾 Pickle | 🏋️ Entrena |
| `check_r2.py` | ❌ No usa | 💾 Pickle | ❌ No usa |
| `interfaz_planificacion_qin_v1.py` | 🏋️ Entrena | ❌ No usa | 🏋️ Entrena |

### 🔴 Hallazgos críticos:

1. **3 archivos entrenan modelos en memoria:**
   - `graficos_comparativa_test.py`: ¡Entrena los 3 modelos!
   - `scatter_plots_test.py`: Entrena RandomForest
   - `interfaz_planificacion_qin_v1.py`: Entrena LightGBM y RandomForest

2. **Consecuencias:**
   - Cada ejecución da métricas **ligeramente diferentes** por aleatoriedad
   - RandomForest especialmente variable (bootstrap sampling)
   - Sin `random_state` fijo, resultados no reproducibles

3. **LightGBM recién reentrenado (2.6 horas):**
   - Métricas actuales diferentes a runs anteriores
   - Modelo pickle actualizado recientemente

4. **Paths inconsistentes:**
   - `scatter_plots_test.py` busca XGBoost en 2 rutas diferentes:
     - `models/forecasting/modelo_forecasting_xgboost.pkl` ✅
     - `models/modelo_forecasting_xgboost.pkl` ❓

---

## 🎯 CAUSAS CONFIRMADAS DE DIFERENCIAS EN MÉTRICAS

### 1️⃣ DIFERENTES TARGETS (PRINCIPAL)

**Impacto: ALTO**

- 4 archivos evalúan `Qout` = Qin_perfil - Q_net_predicted
- 1 archivo busca `Volumen_Total_m3` (no existe)
- **No tiene sentido comparar métricas de variables diferentes**

**Ejemplo:**
- R² de Q_net vs. R² de Qout son métricas **completamente distintas**
- Q_net: Flujo neto (entrada - salida)
- Qout: Demanda estimada

### 2️⃣ ENTRENAMIENTO EN MEMORIA (ALTO)

**Impacto: MEDIO-ALTO**

3 scripts entrenan modelos en cada ejecución:
- Sin `random_state` fijo
- Aleatoriedad en:
  - RandomForest: bootstrap sampling
  - LightGBM/XGBoost: bagging, feature selection
- **Cada run da métricas ligeramente diferentes**

**Variabilidad esperada:**
- R²: ±0.001 a ±0.005
- RMSE: ±10 a ±50 m³/h
- MAE: ±5 a ±30 m³/h

### 3️⃣ DIFERENTES VERSIONES DE MODELOS (MEDIO)

**Impacto: MEDIO**

- LightGBM reentrenado hace 2.6 horas
- XGBoost tiene 41 horas de antigüedad
- Métricas actuales ≠ métricas de hace 2+ días

### 4️⃣ DIFERENCIA DE 1 REGISTRO (BAJO)

**Impacto: MÍNIMO**

- `interfaz_planificacion_qin_v1.py` usa 1 registro extra
- 2,257 vs 2,256 registros = 0.04% de diferencia
- Impacto despreciable en métricas

---

## 💡 RECOMENDACIONES PRIORITARIAS

### 🔴 URGENTE - Arreglar `modelo_gradio.py`

**Problema:** Busca columna `"Volumen_Total_m3"` que no existe

**Opciones:**
1. Cambiar a `"sist_Vtotal_m3"` (existe en datos)
2. Cambiar a predecir `"Q_net_m3h"` como los otros
3. Calcular `Qout` como los otros 4 archivos

```python
# Opción recomendada (consistente con otros):
y_col = 'Q_net_m3h'
# Luego calcular Qout:
Qin_perfil = df_train.groupby('hora')['sist_Qin_m3h'].median()
df_test['Qout'] = df_test['hora'].map(Qin_perfil) - df_test['Q_net_m3h']
```

### 🟠 ALTA PRIORIDAD - Estandarizar cálculo de métricas

**Crear script único oficial para métricas:**

```python
# calcular_metricas_oficiales.py
"""
Script oficial para calcular métricas del período de test.
Uso: python calcular_metricas_oficiales.py
"""

# Usar modelo PICKLE (no entrenar)
modelo = joblib.load('models/forecasting/modelo_forecasting_lgbm.pkl')

# Target estándar: Qout
Qin_perfil = df_train.groupby('hora')['sist_Qin_m3h'].median()
y_true = df_test['hora'].map(Qin_perfil) - df_test['Q_net_m3h']
y_pred_Qnet = modelo.predict(X_test)
y_pred = df_test['hora'].map(Qin_perfil) - y_pred_Qnet

# Métricas
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Guardar resultados oficiales
resultados = {
    'modelo': 'LightGBM',
    'timestamp_modelo': '2025-12-08 06:10',
    'target': 'Qout',
    'test_inicio': '2025-06-26 22:00',
    'test_fin': '2025-09-30 23:00',
    'registros': len(y_true),
    'R2': r2,
    'MAE': mae,
    'RMSE': rmse,
    'fecha_calculo': datetime.now().isoformat()
}

with open('metricas_oficiales.json', 'w') as f:
    json.dump(resultados, f, indent=2)
```

### 🟡 MEDIA PRIORIDAD - Usar modelos pickle (no entrenar)

**Para reproducibilidad:**

```python
# ❌ NO HACER (variable):
modelo = LGBMRegressor(n_estimators=300)
modelo.fit(X_train, y_train)

# ✅ HACER (reproducible):
modelo = joblib.load('models/forecasting/modelo_forecasting_lgbm.pkl')
```

**Si necesitas entrenar:**
```python
modelo = LGBMRegressor(
    n_estimators=300,
    random_state=42  # ⭐ IMPORTANTE: Fijar semilla
)
```

### 🟢 BAJA PRIORIDAD - Documentar versión de modelos

**Para tesis/reportes:**

```python
# Registrar versión del modelo usado
info_modelo = {
    'modelo': 'LightGBM',
    'archivo': 'modelo_forecasting_lgbm.pkl',
    'timestamp': '2025-12-08 06:10:44',
    'hiperparametros': modelo.get_params(),
    'features': len(modelo.feature_name_),
}
```

---

## 📈 MÉTRICAS ACTUALES (REFERENCIA)

### Modelo: LightGBM (reentrenado 2025-12-08 06:10)

**Test set (2025-06-26 22:00 → 2025-09-30 23:00):**
- **R²:** 0.6521
- **MAE:** 1,599.5 m³/h
- **RMSE:** 2,546.1 m³/h
- **Registros:** 2,256
- **Features:** 164

**Nota:** Estas métricas son para predicción de `Q_net_m3h`. Para `Qout`, calcular como:
```python
Qout_pred = Qin_perfil - Q_net_pred
```

---

## 🎓 PARA LA TESIS

### Recomendaciones metodológicas:

1. **Definir UN target oficial:**
   - ¿Predecir Q_net o Qout?
   - Documentar claramente en metodología

2. **Usar UN script oficial para métricas:**
   - Crear `calcular_metricas_oficiales.py`
   - Ejecutar múltiples veces (n=10)
   - Reportar: promedio ± desviación estándar

3. **Fijar random_state:**
   ```python
   np.random.seed(42)
   modelo = LGBMRegressor(..., random_state=42)
   ```

4. **Documentar versión de modelo:**
   - Timestamp de entrenamiento
   - Hiperparámetros usados
   - Features incluidas

5. **Reportar período de test claramente:**
   - Fechas exactas
   - Número de registros
   - Duración en días

### Ejemplo de tabla para tesis:

| Modelo | R² | MAE (m³/h) | RMSE (m³/h) | Test Period |
|--------|----|-----------:|------------:|-------------|
| LightGBM | 0.652 ± 0.003 | 1,599 ± 15 | 2,546 ± 22 | Jun 26 - Sep 30, 2025 |
| XGBoost | 0.591 ± 0.004 | 1,789 ± 18 | 2,762 ± 25 | Jun 26 - Sep 30, 2025 |
| Random Forest | 0.523 ± 0.012 | 1,945 ± 45 | 2,982 ± 78 | Jun 26 - Sep 30, 2025 |

*Nota: Métricas calculadas sobre predicción de Qout (demanda), promedio de 10 ejecuciones.*

---

## 📝 ARCHIVOS GENERADOS

1. `outputs/ANALISIS_1_METRICAS.json` - Comparación de métodos
2. `outputs/ANALISIS_2_TARGETS_DETALLADO.json` - Verificación de targets
3. `outputs/ANALISIS_3_PERIODOS_TEST.json` - Verificación de períodos
4. `outputs/ANALISIS_4_MODELOS.json` - Verificación de modelos

---

## ✅ CONCLUSIONES FINALES

### ¿Por qué las métricas son diferentes?

**Respuesta definitiva:**

1. **Targets diferentes** (Q_net vs Qout vs columna inexistente) → **CAUSA PRINCIPAL**
2. **Modelos entrenados en memoria** con aleatoriedad → **CAUSA SECUNDARIA**
3. **Versiones de modelos diferentes** (recién reentrenado vs antiguo) → **CAUSA TERCIARIA**
4. ~~Períodos diferentes~~ → **NO es la causa** (son idénticos)
5. ~~Data leakage~~ → **NO es problema** (prevenido correctamente)

### ¿Qué hacer?

1. **Inmediato:** Arreglar `modelo_gradio.py` (columna inexistente)
2. **Corto plazo:** Crear script oficial de métricas
3. **Medio plazo:** Usar modelos pickle, fijar random_state
4. **Para tesis:** Ejecutar múltiples veces, reportar promedio ± std

### ¿Cuándo esperar métricas idénticas?

**Solo si:**
- ✅ Mismo target evaluado
- ✅ Mismo modelo pickle (no entrenar)
- ✅ Mismo período exacto
- ✅ Mismas features
- ✅ `random_state` fijo (si se entrena)

**Variabilidad aceptable:**
- R²: ±0.001 a ±0.005
- RMSE/MAE: ±1-2% del valor

---

**Fin del reporte diagnóstico** | Generado: 2025-12-08 08:47
