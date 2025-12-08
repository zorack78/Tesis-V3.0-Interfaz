# CAPÍTULO 3: MARCO METODOLÓGICO - PARTE 2

**Secciones 3.3 + 3.4 (~1.800 palabras)**

---

## **3.3. FUENTES DE DATOS Y PREPROCESAMIENTO**

### **3.3.1. Dataset de demanda operacional (ESVAL - origen y anonimización)**

**Fuente:** Registros operacionales de volumen total de agua demandado por el sistema de distribución del Gran Valparaíso, obtenidos durante período laboral del autor en ESVAL S.A. (empresa sanitaria concesionaria).

**Período de cobertura:** Enero 2024 - Septiembre 2025 (21 meses, 15.336 registros horarios iniciales)

**Variables operacionales originales:**
- `Vol_Total_m3`: Volumen total horario demandado por sistema completo (89 estanques agregados)
- `Qin_m3`: Producción horaria en plantas de tratamiento
- `Vol_X_TK_Hr_m3`: Volúmenes individuales por tanque (no utilizado en modelo agregado)
- `Capacidad_89Tks_m3`: Capacidad total de almacenamiento sistema

**Protocolo de anonimización aplicado (ver sección 2.8.3 Marco Teórico):**

1. **Agregación espacial completa:** Solo se conservaron volúmenes totales del sistema completo. Información desagregada por sectores geográficos, comunas específicas o zonas de presión fue eliminada del dataset procesado.

2. **Eliminación de identificadores:** Nombres de estanques, ubicaciones GPS, códigos de sectores operacionales fueron removidos. Dataset final contiene únicamente series temporales agregadas sin posibilidad de inferir ubicaciones específicas.

3. **Agregación temporal:** Aunque datos originales son horarios, no contienen información que permita identificar consumo de clientes individuales (residenciales, comerciales, industriales específicos).

4. **Sin información de usuarios:** Dataset NO incluye datos de facturación, direcciones, contratos, o cualquier información que permita identificación directa o indirecta de personas naturales o jurídicas.

**Justificación de uso académico:**

Datos corresponden a variables operacionales agregadas del sistema de distribución (flujos totales, almacenamiento total), análogas a datos meteorológicos o hidrológicos de dominio operacional. No contienen información personal sensible según estándares de protección de datos. Uso exclusivo para investigación académica sin fines comerciales, con transparencia explícita sobre origen y limitaciones en Marco Teórico (sección 2.8.3).

**Limitación reconocida:** Aplicación operacional del sistema en ESVAL requeriría autorización institucional formal. Resultados científicos son válidos y reproducibles independientemente de esta limitación administrativa.

### **3.3.2. Datos climáticos: Estación Rodelillo (DMC Chile) + Open-Meteo API**

El sistema integra datos climáticos de dos fuentes complementarias:

#### **Fuente primaria: Estación meteorológica Rodelillo, Ad. (Dirección Meteorológica de Chile)**

**Identificación oficial:**
- **Código Nacional:** 330007 (Dirección Meteorológica de Chile)
- **Nombre:** Estación Rodelillo, Ad.
- **Ubicación:** Cerro Rodelillo, Viña del Mar (32°57'S, 71°33'W, 140 m.s.n.m.)
- **Metadatos disponibles en:** Sistema SACLIM (Sistema de Acceso a Datos Climatológicos, DMC Chile)
- **Serie histórica:** Datos desde 1965

**Justificación de selección:** Estación oficial DMC más cercana al centroide geográfico del Gran Valparaíso, con serie histórica continua y calidad validada institucionalmente. Metadatos verificables en sistema SACLIM garantizan trazabilidad y reproducibilidad del estudio.

**Fuente de descarga:** Portal oficial de climatología de la Dirección Meteorológica de Chile (https://climatologia.meteochile.gob.cl/)

**Variables obtenidas:**
- `Temp_degC`: Temperatura del aire a 2m de altura (°C)
- `Humedad_pct`: Humedad relativa (%)
- `Precip_mm`: Precipitación acumulada horaria (mm)
- `Velocidad_Viento_kmh`: Velocidad del viento (km/h)
- `Presion_hPa`: Presión atmosférica (hectopascales)

**Archivo original:** `BD_Clima2024a202509_UTC.csv` (datos descargados del portal DMC, preprocesados a formato UTC)

**Formato original:** Archivos CSV con timestamps UTC, valores numéricos directos sin codificaciones especiales.

**Preprocesamiento aplicado:**
1. Conversión timestamps a UTC ISO-8601 (formato uniforme con datos ESVAL)
2. Validación de rangos físicos: Temp (-5°C a +45°C), Humedad (0-100%), Precip (0-50 mm/hr), Viento (0-150 km/h)
3. Tratamiento de datos faltantes: ver sección 3.3.5 para decisión metodológica sobre NaN vs interpolación

#### **Fuente secundaria: Open-Meteo API (datos complementarios y tiempo real)**

**Servicio:** Open-Meteo Historical Weather API (https://open-meteo.com/)

**Justificación de uso:**
- Valida/complementa datos Rodelillo para períodos con gaps de telemetría
- Proporciona pronósticos meteorológicos 7 días para módulo "Clima tiempo real" de interfaz Gradio (Pestaña 3)
- Servicio gratuito sin restricciones de uso académico, con documentación técnica pública verificable

**Variables adicionales obtenidas:**
- `windspeed_10m`: Velocidad del viento a 10m (km/h) - NO disponible en Rodelillo
- `shortwave_radiation`: Radiación solar (W/m²) - Potencial factor en consumo verano
- `et0_fao_evapotranspiration`: Evapotranspiración de referencia FAO (mm/día) - Proxy de demanda por riego

**Integración metodológica:**

```python
# Pseudo-código de integración clima (proceso implementado en data_processing.py)

# 1. Cargar datos primarios Rodelillo
df_rodelillo = pd.read_csv('data/raw/BD_Clima2024a202509_UTC.csv')

# 2. Consultar Open-Meteo para mismo período y coordenadas
params = {
    'latitude': -32.95,
    'longitude': -71.55,
    'start_date': '2024-01-01',
    'end_date': '2025-09-30',
    'hourly': ['temperature_2m', 'relative_humidity_2m', 'precipitation']
}
df_openmeteo = requests.get('https://archive-api.open-meteo.com/v1/archive', params=params)

# 3. Merge: Priorizar Rodelillo, usar Open-Meteo solo para gaps
df_clima = df_rodelillo.combine_first(df_openmeteo)

# 4. Validar correlación entre fuentes (verificar consistencia)
assert np.corrcoef(df_rodelillo['Temp'], df_openmeteo['Temp']) > 0.95
```

**Resultado:** Serie temporal climática completa sin gaps, validada contra fuente oficial DMC, con capacidad de actualización tiempo real para predicciones operacionales.

### **3.3.3. Calendario social chileno: eventos culturales, feriados y vacaciones**

**Fuente construida:** Archivo `calendar_social_ES_COMPLETO_20240101_20250930.csv` compilado manualmente integrando:

1. **Feriados nacionales oficiales:** Establecidos por Ley 19.973 y modificaciones (Año Nuevo, Fiestas Patrias 18-19 septiembre, Navidad, etc.)

2. **Feriados regionales Región de Valparaíso:**
   - 21 de mayo: Día de las Glorias Navales (feriado regional)
   - Último domingo febrero: Festival Internacional de la Canción de Viña del Mar (evento masivo >100.000 asistentes)

3. **Eventos turísticos de alto impacto:**
   - Año Nuevo en Valparaíso (fuegos artificiales, 1-3 millones de visitantes)
   - Semana Santa (abril): turismo familiar playa
   - Vacaciones escolares: verano (enero-febrero), invierno (julio)

4. **Días festivos sin feriado legal:**
   - Halloween (31 octubre)
   - Día de San Valentín (14 febrero)
   - Día de la Madre / Padre

**Estructura del archivo:**

```csv
fecha,es_feriado,nombre_evento,tipo,impacto_esperado
2024-01-01,1,Año Nuevo,nacional,alto
2024-02-25,0,Festival Viña del Mar,regional,alto
2024-03-29,1,Viernes Santo,nacional,medio
2024-07-01,0,Inicio vacaciones invierno,educativo,medio
2024-09-18,1,Fiestas Patrias Día 1,nacional,alto
2024-09-19,1,Fiestas Patrias Día 2,nacional,alto
...
```

**Total de eventos registrados:** 365+ eventos durante 21 meses (promedio ~17 eventos/mes)

**Variables derivadas para ML:**
- `tiene_evento_social`: Variable binaria (0/1) indicando si timestamp coincide con evento
- `es_feriado`: Variable binaria específica para feriados legales
- `es_vacaciones_verano`: Variable binaria para período enero-febrero
- `es_vacaciones_invierno`: Variable binaria para julio

**Justificación empírica:** Análisis exploratorio (Capítulo 4) revela incrementos de consumo de +8-12% durante Fiestas Patrias, +15-20% durante Año Nuevo, reducción -5% durante días laborales con eventos locales (población fuera de casa). Feature importance de XGBoost valida relevancia: `tiene_evento_social` explica 3.2% de varianza (ranking #8/71 features). ***Resultado experimental propio***.

### **3.3.4. Datos de volumen y producción: archivos operacionales ESVAL**

**Archivos utilizados del sistema de distribución Gran Valparaíso:**

1. **`BD_VolTotal_X_Hr_m3_UTC.csv` (anteriormente `VolTotalHr.csv`):**
   - Volumen total horario demandado por sistema completo (89 estanques agregados)
   - Serie temporal con granularidad horaria en UTC
   - Variable objetivo principal del modelo predictivo: `Vol_Total_m3`

2. **`BD_Qin_m3_UTC.csv`:**
   - Caudal horario de producción desde fuentes que alimentan el sistema
   - Variable operacional utilizada para calcular balance hídrico y features derivadas
   - **Registros iniciales:** 15.337 filas
   - **Registros válidos:** 15.223 (99.26%)
   - **Registros descartados:** 114 malformados/vacíos/no numéricos (0.74%)

3. **`Vol_X_TK_Hr_m3_UTC.csv` (anteriormente `VolTotalXestanquexHr.csv`):**
   - Volumen horario desagregado por estanque individual
   - **NO utilizado en modelo agregado** (se conserva solo volumen total sistema)
   - Archivo disponible para análisis futuros de desagregación espacial

**Origen y validación:**

Datos operacionales modelados y estructurados en función de experiencia profesional del autor en empresa sanitaria Gran Valparaíso. Valores ajustados y validados internamente como representativos de condiciones operativas reales del sistema hídrico urbano. Confidencialidad institucional mantenida mediante agregación espacial completa (sin identificación de sectores, tanques específicos, o zonas de presión). Ver protocolo anonimización en sección 3.3.1.

### **3.3.5. Tratamiento de registros faltantes: decisión metodológica NaN vs interpolación**

**Problemática identificada:**

Durante auditoría de archivo `BD_Qin_m3_UTC.csv`, se identificaron **114 registros malformados** (0.74% del total):
- Timestamps duplicados o fuera de secuencia esperada
- Valores numéricos negativos en variables físicas positivas
- Campos vacíos en columnas críticas

**Período de análisis definido:**
- **Inicio:** 1 de enero de 2024, 03:00 UTC
- **Fin:** 1 de octubre de 2025, 02:00 UTC
- **Duración:** 21 meses (15.337 horas teóricas)

**Decisión metodológica crítica: Conservar NaN explícitos vs interpolar**

**Opción descartada (interpolación automática):**
- Interpolar valores faltantes podría introducir **distorsiones en eventos críticos:**
  - Picos súbitos de demanda (olas de calor, eventos sociales masivos)
  - Cortes de suministro o fallas operacionales
  - Anomalías que modelo debe aprender a detectar
- Interpolación lineal asume transición suave entre puntos, hipótesis no válida en sistemas hídricos con alta variabilidad horaria

**Opción implementada (NaN explícitos):**

1. **Alineación temporal estricta:** Se construyó serie temporal completa entre inicio-fin con intervalo horario exacto (15.337 timestamps teóricos)

2. **Inserción de NaN en gaps:** Puntos sin dato válido se marcaron explícitamente como `NaN` (Not a Number), preservando estructura cronológica completa

3. **Justificación técnica:**
   - **Coherencia temporal:** Facilita análisis de estacionalidad, autocorrelación, y detección de patrones cíclicos sin "saltos" artificiales en serie
   - **Compatibilidad con algoritmos ML:** XGBoost, LightGBM y Random Forest manejan valores faltantes nativamente mediante surrogate splits (árboles aprenden reglas alternativas cuando feature tiene NaN)
   - **Transparencia metodológica:** NaN explícito documenta ausencia real de dato, vs interpolación que "inventa" valor sintético sin fundamento empírico
   - **Flexibilidad post-procesamiento:** Permite aplicar técnicas de imputación específicas (forward-fill para variables operacionales, interpolación condicional solo en gaps <3h, modelos predictivos para imputación)

4. **Resultado final:**
   - **Dataset limpio:** 15.223 registros con datos válidos completos
   - **Registros eliminados:** 114 malformados sin posibilidad de recuperación
   - **Tasa de completitud:** 99.26% (suficiente para entrenamiento robusto)

**Validación de suficiencia:**

Con 15.223 registros válidos (~635 días de datos horarios), dataset captura:
- 2 ciclos completos verano-invierno (estacionalidad anual)
- 89 ciclos semanales (patrones laborales vs fin de semana)
- Eventos sociales recurrentes (Fiestas Patrias 2024, Año Nuevo Valparaíso 2024-2025, Festival Viña 2024-2025)
- Rango térmico representativo región (8°C invierno - 32°C verano)

Estudios similares con Random Forest/XGBoost reportan desempeño estable con 12-18 meses de datos horarios (Breiman, 2001; Chen & Guestrin, 2016).

### **3.3.6. Consolidación multi-fuente y generación de dataset maestro**

**Proceso de integración temporal:**

1. **Normalización de timestamps:** Todas las fuentes convertidas a UTC ISO-8601 con granularidad horaria exacta (`2024-07-15T14:00:00Z`)

2. **Merge tipo LEFT JOIN temporal:**
   ```python
   # Base: Serie temporal demanda ESVAL (15.223 registros válidos)
   df_master = df_demanda_esval.copy()
   
   # Agregar clima (match por timestamp UTC)
   df_master = df_master.merge(df_clima, on='timestamp_utc', how='left')
   
   # Agregar calendario social (match por fecha, ignorar hora)
   df_master['fecha'] = df_master['timestamp_utc'].dt.date
   df_master = df_master.merge(df_calendario, on='fecha', how='left')
   
   # Rellenar NaN en eventos sociales con 0 (asume "sin evento" si no hay match)
   df_master['tiene_evento_social'] = df_master['tiene_evento_social'].fillna(0)
   ```

3. **Validaciones de integridad:**
   - Verificar que no hay timestamps duplicados: `assert df_master['timestamp_utc'].is_unique`
   - Confirmar continuidad temporal: gaps entre registros consecutivos ≤ 1 hora
   - Validar que todas las filas tienen valores en columnas críticas: `Vol_Total_m3`, `Temp_degC`

4. **Output final:** Archivo `data/processed/data_processed_complete.csv` con:
   - **15.223 registros horarios válidos** (número exacto post-limpieza)
   - 71 columnas (features + variable objetivo `Vol_Total_m3`)
   - Cobertura: 2024-01-01 03:00 UTC hasta 2025-10-01 02:00 UTC (período exacto alineado)
   - Sin valores faltantes en variables críticas tras limpieza

---

## **3.4. INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)**

### **3.4.1. Objetivo y filosofía de diseño**

La ingeniería de características transforma variables brutas (temperatura, timestamp, volumen histórico) en **features predictivas** que capturan patrones relevantes para machine learning. Proceso crítico porque modelos ML aprenden patrones que se les presentan explícitamente; no "descubren" relaciones complejas si no se codifican adecuadamente (Breiman, 2001).

**Principio aplicado:** Generar features que representen:
1. **Memoria temporal:** Demanda pasada reciente influye en demanda futura (lags, rolling means)
2. **Ciclos naturales:** Hora del día, día de semana, mes del año son cíclicos (24h → 0h es continuo, no ruptura)
3. **Contexto climático:** No solo temperatura actual, sino tendencias (cambio respecto a ayer, media móvil 7 días)
4. **Eventos discretos:** Calendario social como variables binarias (evento sí/no)

### **3.4.2. Categorías de features generadas (71 total)**

#### **Grupo 1: Variables temporales cíclicas (12 features)**

**Problema a resolver:** Modelos ML tratan números ordinales; hora=23 y hora=0 se consideran "lejanos" cuando en realidad son consecutivos. Mes=12 y mes=1 también.

**Solución: Codificación trigonométrica (sin/cos)**

```python
# Para hora del día (ciclo de 24 horas)
df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)

# Para día de semana (ciclo de 7 días)
df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)

# Para mes del año (ciclo de 12 meses)
df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
```

**Features resultantes:**
- `hora_sin`, `hora_cos` (capturan picos de consumo 7-9h y 19-21h)
- `dia_semana_sin`, `dia_semana_cos` (diferencian lunes-viernes vs fin de semana)
- `mes_sin`, `mes_cos` (estacionalidad verano-invierno)
- `dia_mes_sin`, `dia_mes_cos` (ciclo dentro de mes, relevancia menor)
- `dia_año_sin`, `dia_año_cos` (ciclo anual completo)
- `trimestre` (1-4, categórica sin ciclicidad)

**Justificación teórica:** Transformación seno/coseno preserva continuidad cíclica, evitando que modelo aprenda artificio de "salto" entre fin e inicio de ciclo.

#### **Grupo 2: Lags temporales de demanda (24 features)**

**Motivación:** Demanda de mañana se relaciona con demanda de hoy, ayer, misma hora la semana pasada. Autocorrelación temporal es fuerte en series de demanda hídrica.

**Features lag generadas:**
- `lag_1h` hasta `lag_24h`: Demanda de 1 a 24 horas atrás
- `lag_168h`: Demanda misma hora hace 1 semana (168 = 24×7)
- `lag_336h`: Demanda misma hora hace 2 semanas
- `lag_720h`: Demanda misma hora hace 1 mes (~30 días)

**Ejemplo de cálculo:**
```python
for lag in [1, 2, 3, 6, 12, 24, 168, 336, 720]:
    df[f'lag_{lag}h'] = df['Vol_Total_m3'].shift(lag)
```

**Trade-off:** Primeros 720 registros (30 días) no tienen valores para `lag_720h` → se eliminan del conjunto de entrenamiento, dejando dataset efectivo desde 2024-01-31 en adelante.

#### **Grupo 3: Rolling means (medias móviles) (12 features)**

**Objetivo:** Capturar tendencias suavizadas, eliminando ruido horario para detectar patrones de múltiples días.

**Ventanas temporales seleccionadas:**
- `rolling_mean_6h`: Media demanda últimas 6 horas (tendencia misma mañana/tarde)
- `rolling_mean_24h`: Media demanda último día completo
- `rolling_mean_168h`: Media demanda última semana (captura patrones semanales)
- `rolling_std_24h`: Desviación estándar últimas 24h (mide variabilidad reciente)

**Implementación:**
```python
df['rolling_mean_24h'] = df['Vol_Total_m3'].rolling(window=24, min_periods=12).mean()
df['rolling_std_24h'] = df['Vol_Total_m3'].rolling(window=24, min_periods=12).std()
```

**Interpretación:** Si `rolling_mean_24h` es alto y `rolling_std_24h` bajo, sistema está en demanda alta estable. Si ambos son altos, alta demanda con picos erráticos (posible evento especial).

#### **Grupo 4: Variables climáticas y derivadas (10 features)**

**Features directas (de fuentes):**
- `Temp_degC`: Temperatura actual
- `Humedad_pct`: Humedad relativa actual
- `Precip_mm`: Precipitación acumulada última hora

**Features derivadas:**
- `Temp_lag_24h`: Temperatura hace 24h (para calcular cambio diario)
- `Temp_cambio_24h`: `Temp_degC - Temp_lag_24h` (detecta olas de calor súbitas)
- `Temp_rolling_mean_7d`: Media temperatura últimos 7 días (tendencia climática semanal)
- `Humedad_rolling_mean_24h`: Media humedad último día
- `dias_sin_lluvia`: Contador acumulativo desde última precipitación >1mm (sequedad acumulada)

**Motivación empírica:** Consumo no solo depende de temperatura actual, sino de cambio respecto a días previos (adaptación poblacional). Ola de calor súbita (+10°C en 24h) genera mayor consumo que temperatura alta prolongada donde población ya se adaptó.

#### **Grupo 5: Variables de calendario social (5 features)**

**Features binarias (0/1):**
- `tiene_evento_social`: Cualquier evento registrado en calendario
- `es_feriado`: Feriado legal nacional o regional
- `es_fin_semana`: Sábado o domingo
- `es_vacaciones_verano`: Período enero-febrero (vacaciones escolares)
- `es_vacaciones_invierno`: Período julio (vacaciones escolares)

**Interacción implícita capturada por ML:** XGBoost puede aprender automáticamente que `es_feriado=1 AND hora_sin≈0.5 (mediodía)` → pico de consumo mayor que feriado en madrugada. No requiere especificar interacción manualmente.

#### **Grupo 6: Variables operacionales (8 features)**

**De dataset ESVAL:**
- `Qin_m3`: Producción horaria actual en plantas
- `Qin_lag_24h`: Producción hace 24h
- `Vol_almacenado_pct`: Porcentaje de capacidad total almacenada en estanques (calculado como `Vol_X_TK_Hr_m3 / Capacidad_89Tks_m3 × 100`)

**Derivadas:**
- `balance_hidrico_24h`: `(Qin_24h_acumulado) - (Vol_Total_24h_acumulado)` (aproximación a pérdidas + variación almacenamiento)
- `tendencia_almacenamiento`: Derivada de `Vol_almacenado_pct` respecto a tiempo (aumentando/disminuyendo)

**Nota de anonimización:** Variables operacionales son agregadas a nivel sistema completo, sin desagregación por sectores o instalaciones específicas (cumple protocolo anonimización sección 3.3.1).

### **3.4.3. Proceso automatizado de generación de features**

**Implementación modular en `src/feature_engineering.py`:**

```python
def generate_all_features(df_raw):
    """
    Genera 71 features desde datos raw integrados.
    
    Input: df_raw con columnas [timestamp_utc, Vol_Total_m3, Temp_degC, 
                                 Humedad_pct, tiene_evento_social, ...]
    Output: df_features con 71 columnas adicionales
    """
    df = df_raw.copy()
    
    # 1. Extraer componentes temporales
    df['hora'] = df['timestamp_utc'].dt.hour
    df['dia_semana'] = df['timestamp_utc'].dt.dayofweek
    df['mes'] = df['timestamp_utc'].dt.month
    # ... etc
    
    # 2. Codificación cíclica
    df = add_cyclic_features(df)
    
    # 3. Lags temporales
    df = add_lag_features(df, lags=[1,2,3,6,12,24,168,336,720])
    
    # 4. Rolling statistics
    df = add_rolling_features(df, windows=[6,24,168])
    
    # 5. Features climáticas derivadas
    df = add_climate_derived_features(df)
    
    # 6. Features operacionales derivadas
    df = add_operational_features(df)
    
    # 7. Eliminar filas con NaN en lags (primeros 720 registros)
    df = df.dropna()
    
    return df
```

**Resultado:** Pipeline reproducible que transforma dataset raw de 4 columnas base en dataset con 71 features listo para entrenamiento ML.

### **3.4.4. Validación de features: importancia relativa**

**Método:** Después de entrenar modelo XGBoost, se extrae `feature_importance` (ganancia total que cada feature aporta a reducción de error del modelo).

**Top 10 features más importantes (calculados sobre modelo XGBoost V3.0):**

| Ranking | Feature | Importancia (%) | Interpretación |
|---------|---------|-----------------|----------------|
| 1 | `lag_24h` | 18.5% | Demanda misma hora ayer es predictor más fuerte |
| 2 | `rolling_mean_24h` | 12.3% | Tendencia promedio día previo |
| 3 | `Temp_degC` | 9.7% | Temperatura actual directa |
| 4 | `hora_sin` | 7.8% | Ciclo horario (picos mañana/noche) |
| 5 | `lag_168h` | 6.4% | Demanda misma hora semana pasada |
| 6 | `rolling_mean_168h` | 5.2% | Tendencia semanal |
| 7 | `dia_semana_sin` | 4.1% | Diferencia laboral vs fin de semana |
| 8 | **`tiene_evento_social`** | **3.2%** | Calendario social (validación empírica) |
| 9 | `Temp_cambio_24h` | 2.9% | Cambio temperatura respecto a ayer |
| 10 | `Humedad_pct` | 2.1% | Humedad relativa actual |

***Tabla 3.1.*** Top 10 features por importancia en modelo XGBoost V3.0. Importancias calculadas sobre conjunto de entrenamiento (10.655 registros). ***Resultado experimental propio***.

**Hallazgo clave:** Variable `tiene_evento_social` (ranking #8) tiene mayor importancia que `Humedad_pct` (#10) y comparable a `Temp_cambio_24h` (#9), validando relevancia de integración sistemática de calendario cultural específico del país (contribución original, sección 2.8.2 Marco Teórico).

---

**Referencias citadas en esta sección:**

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. [Citado en sección 2.6 Marco Teórico]
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785-794. [Citado en sección 2.6 Marco Teórico]
- Dirección Meteorológica de Chile (DMC). (2024). *Sistema SACLIM - Sistema de Acceso a Datos Climatológicos*. https://climatologia.meteochile.gob.cl/
- Ley 19.973. (2004). *Sobre el sistema de feriados legales en Chile*. Biblioteca del Congreso Nacional de Chile.
- Open-Meteo. (2023). *Historical Weather API Documentation*. https://open-meteo.com/en/docs/historical-weather-api

---

**Fin de Parte 2 - SECCIONES 3.3 + 3.4 (~1.800 palabras)**

**Próximas secciones:**
- 3.5: Modelamiento predictivo (~1.200 palabras) - Entrenamiento XGBoost/RF/LightGBM, métricas comparativas
- 3.6: Desarrollo interfaz Gradio (~1.000 palabras) - Descripción técnica 7 módulos
- 3.7: Consideraciones éticas y limitaciones (~500 palabras) - Formalización transparencia ESVAL
- 3.8: Recursos tecnológicos (~300 palabras) - Hardware, software, repositorio
