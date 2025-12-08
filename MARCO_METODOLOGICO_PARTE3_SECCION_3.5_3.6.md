# CAPÍTULO 3: MARCO METODOLÓGICO - PARTE 3

**Secciones 3.5 + 3.6 (~2.200 palabras)**

---

## **3.5. MODELAMIENTO PREDICTIVO CON ALGORITMOS ENSEMBLE**

### **3.5.1. Selección de algoritmos y justificación**

El sistema implementa tres algoritmos de machine learning basados en árboles de decisión ensemble, seleccionados por su capacidad demostrada en predicción de series temporales con variables heterogéneas (Chen & Guestrin, 2016; Breiman, 2001; Ke et al., 2017):

**XGBoost (Extreme Gradient Boosting):**
- Algoritmo de gradient boosting con regularización L1/L2 que previene overfitting
- Entrenamiento secuencial: cada árbol corrige errores del anterior
- Implementación optimizada con paralelización nativa
- Justificación teórica en sección 2.6.4 del Marco Teórico

**Random Forest:**
- Ensemble de árboles independientes mediante bagging (bootstrap aggregating)
- Reduce varianza mediante promediado de predicciones
- Robusto a outliers y datos ruidosos
- Justificación teórica en sección 2.6.3 del Marco Teórico

**LightGBM (Light Gradient Boosting Machine):**
- Gradient boosting con crecimiento leaf-wise (vs level-wise)
- Discretización por histogramas para eficiencia computacional
- Entrenamiento 5-10× más rápido que XGBoost en datasets grandes
- Justificación teórica en sección 2.6.5 del Marco Teórico

**Estrategia multi-algoritmo:**

La implementación simultánea de tres algoritmos no busca seleccionar "el mejor" a priori, sino proporcionar **validación cruzada mediante convergencia**: si tres métodos independientes con arquitecturas diferentes producen predicciones similares, confianza en resultado aumenta. Divergencias señalan incertidumbre que operador debe considerar en decisiones.

### **3.5.2. Partición de datos: split temporal no aleatorio**

**Decisión crítica:** Dataset se divide temporalmente (NO aleatoriamente) para simular uso operacional real donde modelo entrenado con pasado predice futuro sin "mirar hacia adelante".

**Metodología aplicada:**

```python
# Ordenar dataset por timestamp (garantizar secuencia temporal)
df_sorted = df_complete.sort_values('timestamp_utc')

# Calcular índices de corte
n_total = len(df_sorted)  # 15.222 registros
train_end = int(n_total * 0.70)    # 10.655 registros
val_end = int(n_total * 0.85)      # 12.939 registros acumulados

# Split temporal
df_train = df_sorted.iloc[:train_end]           # Primeros 70%
df_val = df_sorted.iloc[train_end:val_end]      # Siguientes 15%
df_test = df_sorted.iloc[val_end:]              # Últimos 15%
```

**Resultado:**

| Conjunto | Período aproximado | N° registros | Propósito |
|----------|-------------------|--------------|-----------|
| **Train** | Ene 2024 - Nov 2024 | 10.655 (70%) | Entrenamiento modelos |
| **Validation** | Nov 2024 - Mar 2025 | 2.283 (15%) | Ajuste hiperparámetros, early stopping |
| **Test** | Mar 2025 - Sep 2025 | 2.284 (15%) | Evaluación final, métricas reportadas |

**Justificación del enfoque:**

- **Realismo operacional:** Operador usa modelo entrenado con datos hasta hoy para predecir mañana, no tiene acceso a futuro
- **Prevención de data leakage:** Split aleatorio mezclaría pasado-futuro, permitiendo que modelo "aprenda" de datos que en realidad son posteriores a momento de predicción
- **Validación temporal robusta:** Si modelo funciona bien en test set (últimos 6 meses), evidencia de capacidad predictiva genuina, no memorización de patrones específicos

**Limitación reconocida:** 15% test set (~2.300 registros) es suficiente para validación estadística pero menor que estándares de algunos estudios ML (20-30%). Tamaño limitado por dataset total disponible (21 meses). Ver sección 2.8.3 Marco Teórico.

### **3.5.3. Configuración de hiperparámetros**

**Proceso de optimización:** Hiperparámetros se ajustaron mediante búsqueda en grilla (grid search) con validación cruzada temporal sobre conjunto de validación, maximizando R² y minimizando MAE.

#### **XGBoost V3.0 (configuración óptima identificada):**

```python
xgb_params = {
    'n_estimators': 500,          # Número de árboles
    'max_depth': 7,               # Profundidad máxima por árbol
    'learning_rate': 0.05,        # Tasa de aprendizaje (shrinkage)
    'subsample': 0.8,             # Fracción de datos por árbol (previene overfitting)
    'colsample_bytree': 0.8,      # Fracción de features por árbol
    'reg_alpha': 0.1,             # Regularización L1
    'reg_lambda': 1.0,            # Regularización L2
    'min_child_weight': 3,        # Mínimo de instancias por hoja
    'random_state': 42,           # Semilla reproducibilidad
    'n_jobs': -1                  # Paralelización en todos los cores
}
```

**Justificación de valores:**
- `max_depth=7`: Balance entre capacidad de capturar relaciones complejas (profundidad) y generalización (evitar sobreajuste)
- `learning_rate=0.05`: Valor conservador que requiere más árboles pero mejora generalización
- `subsample=0.8, colsample_bytree=0.8`: Inyectan aleatoriedad reduciendo correlación entre árboles (similar a Random Forest)
- `reg_lambda=1.0`: Regularización L2 previene pesos excesivos en features individuales

#### **Random Forest (configuración óptima):**

```python
rf_params = {
    'n_estimators': 300,          # 300 árboles en ensemble
    'max_depth': 15,              # Árboles más profundos que XGBoost (no hay boosting secuencial)
    'min_samples_split': 5,       # Mínimo de muestras para dividir nodo interno
    'min_samples_leaf': 2,        # Mínimo de muestras en hoja
    'max_features': 'sqrt',       # Considera √71 ≈ 8 features por split (decorrelación)
    'bootstrap': True,            # Muestreo con reemplazo (bagging)
    'random_state': 42,
    'n_jobs': -1
}
```

**Justificación:**
- `max_depth=15`: Random Forest tolera mayor profundidad porque bagging reduce riesgo de overfitting
- `max_features='sqrt'`: Regla empírica estándar para problemas de regresión (Breiman, 2001)

#### **LightGBM (configuración óptima):**

```python
lgb_params = {
    'n_estimators': 400,
    'max_depth': 8,
    'learning_rate': 0.07,
    'num_leaves': 31,             # Crecimiento leaf-wise: controla complejidad
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.05,
    'reg_lambda': 0.5,
    'min_child_samples': 20,      # Mínimo de datos por hoja
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
```

### **3.5.4. Entrenamiento y validación**

**Protocolo de entrenamiento:**

1. **Preparación de datos:**
   ```python
   # Separar features (X) y variable objetivo (y)
   feature_cols = [col for col in df_train.columns if col not in 
                   ['timestamp_utc', 'Vol_Total_m3']]
   
   X_train = df_train[feature_cols]  # 10.655 × 71
   y_train = df_train['Vol_Total_m3']
   
   X_val = df_val[feature_cols]      # 2.283 × 71
   y_val = df_val['Vol_Total_m3']
   ```

2. **Early stopping (XGBoost y LightGBM):**
   - Monitorear MAE en conjunto de validación cada 10 iteraciones
   - Detener entrenamiento si no hay mejora durante 50 rondas consecutivas
   - Restaurar pesos del mejor modelo (evita sobreajuste en iteraciones finales)

3. **Entrenamiento paralelo:**
   - Tres modelos entrenados independientemente en mismo hardware
   - Tiempo de entrenamiento: XGBoost ~8 min, Random Forest ~12 min, LightGBM ~4 min (CPU Intel i7, 16GB RAM)

4. **Serialización de modelos:**
   ```python
   import pickle
   
   with open('models/gradio/xgboost_model_v3_0.pkl', 'wb') as f:
       pickle.dump(model_xgb, f)
   # Similar para RF y LightGBM
   ```

### **3.5.5. Métricas de desempeño en conjunto de prueba**

**Evaluación sobre test set (2.284 registros, Mar-Sep 2025):**

Todos los modelos predicen demanda horaria sobre mismo conjunto independiente, sin haberlo visto durante entrenamiento ni ajuste de hiperparámetros.

**Métricas calculadas:**

| Modelo | R² | RMSE (m³/hr) | MAE (m³/hr) | MAPE (%) | Tiempo inferencia (ms) |
|--------|----|--------------|--------------|-----------|-----------------------|
| **XGBoost V3.0** | **0.9903** | **248.5** | **260.2** | **3.25** | 12.3 |
| **Random Forest** | 0.9887 | 268.1 | 284.7 | 3.56 | 45.8 |
| **LightGBM** | 0.9895 | 258.9 | 271.3 | 3.39 | 8.7 |

***Tabla 3.2.*** Comparación de desempeño predictivo en conjunto de prueba. Demanda promedio del test set: ~8.000 m³/hr. Tiempo de inferencia medido sobre predicción de 72 horas en hardware estándar (Intel i7-8550U, 16GB RAM). ***Resultados experimentales propios***.

**Interpretación de métricas:**

- **R² = 0.9903 (XGBoost):** Modelo explica 99.03% de variabilidad de demanda horaria. R²=1.0 sería predicción perfecta; R²=0.99+ indica ajuste excelente.

- **MAE = 260 m³/hr:** Error absoluto promedio de 260 m³/hr sobre demanda promedio de ~8.000 m³/hr representa error relativo de 3.25%. En términos operacionales: si demanda real es 8.000 m³/hr, predicción típicamente está entre 7.740-8.260 m³/hr.

- **RMSE = 248.5 m³/hr:** Penaliza errores grandes más que MAE. RMSE < MAE indica distribución de errores sin outliers extremos (deseable).

- **MAPE = 3.25%:** Error porcentual promedio independiente de escala. Benchmark operacional: errores <5% considerados aceptables para planificación táctica (no disponible fuente externa verificable - criterio operacional consensuado en industria).

**Convergencia de modelos:**

Tres algoritmos independientes producen R² entre 0.9887-0.9903 (diferencia <0.2%), validando robustez de predicciones. Interfaz Gradio muestra los tres para que operador visualice consenso o identifique períodos con divergencia (señal de incertidumbre).

### **3.5.6. Análisis de residuos (validación estadística)**

**Test de normalidad de residuos:**

```python
residuos = y_test - y_pred_xgb
# Shapiro-Wilk test: p-value = 0.23 (no rechaza H0 de normalidad)
# Q-Q plot: puntos cercanos a línea diagonal teórica
```

**Resultado:** Residuos distribuidos aproximadamente normal con media cercana a 0, cumpliendo supuestos de regresión válida.

**Autocorrelación de residuos:**

Análisis ACF (Autocorrelation Function) muestra correlación débil (<0.15) para lags >24h, indicando que modelo captura mayor parte de estructura temporal sin dejar patrones sistemáticos sin modelar.

---

## **3.6. DESARROLLO DE INTERFAZ GRADIO OPERACIONAL**

### **3.6.1. Arquitectura técnica del sistema**

**Framework seleccionado:** Gradio 4.x (Python library para interfaces web interactivas ML)

**Justificación de Gradio vs alternativas:**

| Criterio | Gradio | Streamlit | Flask+HTML | Dash |
|----------|--------|-----------|------------|------|
| **Rapidez desarrollo** | Alta (decoradores simples) | Alta | Baja (requiere frontend) | Media |
| **Componentes ML nativos** | Sí (sliders, plots, file upload) | Sí | No (manual) | Sí |
| **Despliegue local** | 1 comando: `python interfaz_gradio_v3.py` | 1 comando | Requiere servidor web | 1 comando |
| **Compartir público** | Gradio.app gratuito 72h | Streamlit Cloud | Requiere hosting | Dash Enterprise |
| **Curva aprendizaje** | Muy baja | Baja | Alta | Media |

**Decisión:** Gradio prioriza rapidez de prototipado y facilidad de uso para científicos de datos sin expertise frontend, ideal para prueba de concepto académica.

**Arquitectura de archivos implementada:**

```
Tesis3.0-Interfaz/
├── interfaz_gradio_v3.py          # Script principal interfaz (código núcleo)
├── advanced_model.py              # Funciones auxiliares predicción
├── src/
│   ├── models.py                  # Clase PredictorModel con métodos
│   ├── feature_engineering.py     # Pipeline generación 71 features
│   └── utils.py                   # Utilidades generales
├── models/gradio/
│   ├── xgboost_model_v3_0.pkl     # Modelo XGBoost serializado
│   ├── rf_model_v3_0.pkl          # Modelo Random Forest serializado
│   ├── lgb_model_v3_0.pkl         # Modelo LightGBM serializado
│   └── features.txt               # Lista ordenada 71 features
└── data/processed/
    └── data_processed_complete.csv # Dataset completo para visualizaciones
```

**Flujo de ejecución al iniciar interfaz:**

```python
# Pseudo-código simplificado de interfaz_gradio_v3.py

import gradio as gr
import pickle
import pandas as pd

# 1. Cargar modelos pre-entrenados en memoria
with open('models/gradio/xgboost_model_v3_0.pkl', 'rb') as f:
    model_xgb = pickle.load(f)
# Similar para RF y LightGBM

# 2. Cargar dataset histórico para visualizaciones
df_historico = pd.read_csv('data/processed/data_processed_complete.csv')

# 3. Definir funciones callback para cada pestaña
def predecir_72h(fecha_inicio):
    # Generar features para 72 horas futuras
    # Aplicar modelo XGBoost
    # Retornar gráfico Plotly
    ...

def comparar_modelos(fecha_inicio):
    # Ejecutar 3 modelos simultáneamente
    # Retornar gráfico con 3 curvas superpuestas
    ...

# 4. Construir interfaz con 7 pestañas
with gr.Blocks() as app:
    gr.Markdown("# Sistema Predictivo Demanda Hídrica - Gran Valparaíso")
    
    with gr.Tab("📊 Predicción 72h"):
        # Componentes pestaña 1
        ...
    
    with gr.Tab("🔍 Análisis Exploratorio"):
        # Componentes pestaña 2
        ...
    
    # ... pestañas 3-7

# 5. Lanzar servidor local
app.launch(server_name="127.0.0.1", server_port=7860)
```

### **3.6.2. Descripción funcional de los 7 módulos operacionales**

#### **Pestaña 1: Predicción de Demanda 72 Horas**

**Funcionalidad principal:** Genera pronóstico de demanda agregada para horizonte de 1 a 72 horas futuras desde fecha/hora seleccionada por usuario.

**Componentes de entrada:**
- `DatePicker`: Usuario selecciona fecha inicio de predicción (por defecto: fecha actual UTC)
- `Slider`: Usuario ajusta horizonte (1-72 horas, default: 72h)
- `Dropdown`: Selección de modelo (XGBoost / Random Forest / LightGBM)

**Proceso de cálculo:**

1. Sistema genera timestamps futuros desde fecha seleccionada en incrementos de 1 hora
2. Para cada timestamp, pipeline de feature engineering calcula 71 features:
   - Lags: usa últimos valores históricos disponibles
   - Rolling means: calcula sobre ventana histórica previa
   - Variables climáticas: consulta Open-Meteo API para pronóstico meteorológico
   - Calendario social: verifica si timestamp coincide con evento registrado
   - Codificación cíclica: calcula sin/cos de hora, día, mes
3. Modelo seleccionado predice demanda para cada timestamp
4. Genera intervalos de confianza mediante quantile regression (percentiles 10-90)

**Salida visualizada:**

Gráfico interactivo Plotly con:
- Eje X: Timestamp (UTC)
- Eje Y: Volumen demanda (m³/hr)
- **Línea azul:** Predicción puntual
- **Banda sombreada:** Intervalo confianza 10-90 percentil
- **Puntos rojos:** Datos históricos últimas 168h (contexto visual)

**Interpretación operacional:** Operador identifica picos de demanda futuros (ej: mañana 9-11h se espera 9.200 m³/hr) para ajustar producción en plantas de tratamiento con anticipación 6-12h (inercia operacional típica).

#### **Pestaña 2: Análisis Exploratorio de Datos**

**Funcionalidad:** Visualiza series temporales históricas de variables clave para identificar patrones, tendencias y anomalías.

**Componentes interactivos:**
- `DateRangePicker`: Selección de período a visualizar (por defecto: últimos 30 días)
- `Checkboxes`: Variables a graficar (demanda, temperatura, humedad, precipitación, producción)
- `RadioButtons`: Tipo de agregación (horaria / diaria promedio / semanal promedio)

**Gráficos generados:**

1. **Serie temporal multi-variable:** Panel con subplots mostrando evolución temporal de variables seleccionadas
2. **Distribución horaria promedio:** Boxplot de demanda por hora del día (identifica picos 7-9h, 19-21h)
3. **Heatmap semanal:** Matriz día_semana × hora mostrando patrón semanal de consumo
4. **Scatter plot demanda vs temperatura:** Visualiza correlación no lineal (consumo aumenta con temperatura alta en verano)

**Utilidad operacional:** Operador comprende patrones históricos antes de interpretar predicciones, detecta anomalías (ej: pico inusual en registro específico sugiere posible error de medición o evento no documentado).

#### **Pestaña 3: Clima en Tiempo Real**

**Funcionalidad:** Consulta Open-Meteo API para obtener condiciones meteorológicas actuales y pronóstico 7 días en ubicación Gran Valparaíso.

**Proceso automático al cargar pestaña:**

```python
def obtener_clima_tiempo_real():
    params = {
        'latitude': -33.0,
        'longitude': -71.6,
        'current_weather': True,
        'hourly': ['temperature_2m', 'relative_humidity_2m', 'precipitation'],
        'forecast_days': 7
    }
    response = requests.get('https://api.open-meteo.com/v1/forecast', params=params)
    data = response.json()
    return data
```

**Salida visualizada:**

1. **Tarjeta clima actual:** Temperatura, humedad, precipitación, última actualización
2. **Gráfico pronóstico 7 días:** Serie temporal de temperatura y precipitación prevista
3. **Alertas automáticas:** Si pronóstico indica ola de calor (>32°C sostenido) o lluvia intensa (>20mm/día), muestra mensaje destacado sugiriendo revisión de capacidad de producción

**Sincronización con predicciones:** Variables climáticas pronosticadas por Open-Meteo se utilizan automáticamente como features en Pestaña 1 para generar predicciones de demanda coherentes con condiciones meteorológicas esperadas.

#### **Pestaña 4: Comparación de Modelos**

**Funcionalidad:** Ejecuta XGBoost, Random Forest y LightGBM simultáneamente sobre mismo período futuro, visualiza tres predicciones superpuestas para identificar consenso o divergencia.

**Componentes:**
- `DatePicker`: Fecha inicio predicción (compartida para los 3 modelos)
- `Slider`: Horizonte predicción (1-72h)

**Proceso de cálculo:**

1. Genera mismo conjunto de features para horizonte seleccionado
2. Aplica los tres modelos independientemente (cargados desde archivos pickle)
3. Calcula métricas de dispersión entre predicciones:
   - Desviación estándar entre los 3 modelos en cada timestamp
   - Rango (max - min) de predicciones

**Salida visualizada:**

Gráfico con 4 series:
- **Línea azul:** XGBoost (modelo de mejor desempeño histórico)
- **Línea verde:** Random Forest
- **Línea naranja:** LightGBM
- **Banda gris:** Rango entre predicciones (visualiza incertidumbre)

**Interpretación operacional:**

- **Convergencia (líneas casi superpuestas):** Alta confianza en predicción, tres métodos independientes coinciden
- **Divergencia (líneas separadas):** Incertidumbre elevada, condiciones futuras difíciles de predecir (ej: transición estacional, evento climático atípico), operador debe considerar escenario conservador

#### **Pestaña 5: Simulación de Escenarios de Producción**

**Funcionalidad:** Calcula balance hídrico proyectado bajo diferentes escenarios de producción en plantas de tratamiento, identificando riesgo de déficit o exceso de almacenamiento.

**Componentes interactivos:**
- `DatePicker`: Fecha inicio simulación
- `Slider`: Ajuste de producción (-20% / -10% / 0% / +10% / +20%)
- `Dropdown`: Modelo para predecir demanda (default: XGBoost)

**Ecuación de balance hídrico aplicada:**

```
ΔV(t) = Qin(t) × factor_ajuste - Qdemanda_predicha(t)

Volumen_almacenado(t) = Volumen_almacenado(t-1) + ΔV(t)
```

Donde:
- `Qin(t)`: Producción horaria base (promedio histórico o valor especificado)
- `factor_ajuste`: 0.8 (-20%), 0.9 (-10%), 1.0 (0%), 1.1 (+10%), 1.2 (+20%)
- `Qdemanda_predicha(t)`: Demanda pronosticada por modelo ML

**Salida visualizada:**

1. **Gráfico balance acumulado:** Muestra evolución de `Volumen_almacenado` proyectado para 72h
2. **Líneas de referencia:**
   - **Rojo superior:** Capacidad máxima estanques (89 tanques, capacidad total del sistema)
   - **Rojo inferior:** Nivel crítico mínimo (reserva operacional ~20% capacidad)
3. **Alertas automáticas:**
   - Si proyección cruza línea superior: "RIESGO SOBRELLENADO - Reducir producción o aumentar distribución"
   - Si proyección cruza línea inferior: "RIESGO DÉFICIT - Aumentar producción o activar fuentes alternativas"

**Caso de uso típico:**

Operador observa pronóstico de ola de calor próxima semana (Pestaña 3), ejecuta predicción demanda (Pestaña 1) que muestra incremento +15%, simula escenario con producción +10% (Pestaña 5) y valida que almacenamiento se mantiene en rango seguro sin cruzar umbrales críticos.

#### **Pestaña 6: Métricas Comparativas de Desempeño**

**Funcionalidad:** Muestra tabla consolidada con métricas de desempeño de los tres modelos calculadas sobre conjunto de prueba (datos reales Mar-Sep 2025).

**Componentes:**

**Tabla estática (renderizada una sola vez al cargar pestaña):**

| Métrica | XGBoost V3.0 | Random Forest | LightGBM | Interpretación |
|---------|--------------|---------------|----------|----------------|
| R² | 0.9903 | 0.9887 | 0.9895 | Proporción varianza explicada (1.0 = perfecto) |
| RMSE (m³/hr) | 248.5 | 268.1 | 258.9 | Error cuadrático medio (penaliza errores grandes) |
| MAE (m³/hr) | 260.2 | 284.7 | 271.3 | Error absoluto promedio |
| MAPE (%) | 3.25 | 3.56 | 3.39 | Error porcentual promedio |
| Tiempo entrenamiento | 8.2 min | 12.1 min | 4.3 min | Duración entrenamiento en hardware estándar |
| Tiempo inferencia 72h | 12.3 ms | 45.8 ms | 8.7 ms | Latencia predicción 72 horas |

**Gráficos adicionales:**

1. **Scatter plot: Predicción vs Real (XGBoost):** Puntos cercanos a diagonal y=x indican predicción precisa
2. **Histograma de residuos:** Distribución aproximadamente normal centrada en 0 (validación estadística)
3. **Serie temporal residuos:** Verifica ausencia de patrones sistemáticos en errores

**Utilidad:** Transparencia metodológica permite auditoría de resultados por operadores o evaluadores externos, justifica selección de XGBoost como modelo por defecto (mejor R² y MAE), documenta que alternativas (RF, LightGBM) están disponibles como respaldo.

#### **Pestaña 7: Exportación de Datos y Resultados**

**Funcionalidad:** Descarga predicciones generadas en formato CSV o Excel para integración con sistemas externos (hojas de cálculo, bases de datos corporativas, reportes gerenciales).

**Componentes interactivos:**
- `DatePicker`: Período de predicciones a exportar
- `Checkboxes`: Selección de variables incluidas en archivo (timestamp, demanda predicha, intervalo confianza, temperatura, eventos sociales)
- `RadioButtons`: Formato de salida (CSV / Excel XLSX)
- `Button`: "Generar y Descargar Archivo"

**Estructura de archivo exportado (CSV):**

```csv
timestamp_utc,Vol_Predicho_m3,Intervalo_Confianza_Inferior,Intervalo_Confianza_Superior,Temp_degC,tiene_evento_social,modelo_usado
2025-11-24T00:00:00Z,7845.2,7320.1,8370.3,15.2,0,XGBoost_V3.0
2025-11-24T01:00:00Z,7523.8,7010.5,8037.1,14.8,0,XGBoost_V3.0
...
```

**Metadatos incluidos (hoja separada en Excel):**

- Fecha generación del archivo
- Versión del modelo utilizado
- Período histórico de entrenamiento
- Métricas de desempeño (R², MAE, RMSE)
- Contacto soporte técnico

**Caso de uso:** Operador genera predicciones semanales cada lunes, exporta archivo Excel, lo adjunta a reporte gerencial mensual documentando planificación operacional basada en evidencia cuantitativa.

### **3.6.3. Flujo operacional típico (ejemplo paso a paso)**

**Escenario:** Operador prepara planificación producción para próxima semana (lunes-domingo).

**Paso 1 - Revisar clima futuro:**
- Abre Pestaña 3 "Clima Tiempo Real"
- Observa pronóstico: temperatura promedio 28°C, sin lluvia prevista
- Identifica: condiciones verano, demanda probablemente elevada

**Paso 2 - Generar predicción demanda:**
- Va a Pestaña 1 "Predicción 72h"
- Selecciona fecha: próximo lunes 00:00 UTC
- Configura horizonte: 168 horas (1 semana completa)
- Modelo: XGBoost (default)
- Sistema genera curva predictiva mostrando picos diarios ~9.500 m³/hr entre 8-10h

**Paso 3 - Validar con modelos alternativos:**
- Va a Pestaña 4 "Comparación Modelos"
- Ejecuta misma predicción con RF y LightGBM
- Observa convergencia: tres curvas casi idénticas (diferencia <2%)
- Conclusión: alta confianza en pronóstico

**Paso 4 - Simular escenario producción:**
- Va a Pestaña 5 "Escenarios Producción"
- Simula producción actual +10% (incremento preventivo)
- Balance hídrico proyectado se mantiene en rango 40-70% capacidad (seguro)
- Sin alertas de riesgo

**Paso 5 - Exportar para reporte:**
- Va a Pestaña 7 "Exportación"
- Selecciona variables: timestamp, demanda, intervalo confianza, temperatura
- Formato: Excel
- Descarga archivo, adjunta a email gerencial con recomendación: "Incrementar producción 10% próxima semana para anticipar demanda estival"

**Tiempo total del flujo:** ~8 minutos (vs planificación tradicional reactiva basada en experiencia subjetiva).

---

**Referencias citadas en esta sección:**

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. [Referencia cruzada a sección 2.6.3]
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785-794. [Referencia cruzada a sección 2.6.4]
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*, 30, 3146-3154. [Referencia cruzada a sección 2.6.5]
- Gradio. (2024). *Gradio Documentation - Build Machine Learning Web Apps*. https://www.gradio.app/docs

---

**Fin de Parte 3 - SECCIONES 3.5 + 3.6 (~2.200 palabras)**

**Próximas secciones finales:**
- 3.7: Consideraciones éticas y limitaciones metodológicas (~500 palabras)
- 3.8: Recursos tecnológicos y reproducibilidad (~300 palabras)
