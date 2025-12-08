# CAPÍTULO 3: MARCO METODOLÓGICO - PARTE 1

**Secciones 3.1 + 3.2 (~1.700 palabras)**

---

## **3.1. ENFOQUE METODOLÓGICO GENERAL**

### **3.1.1. Paradigma de investigación**

Este trabajo adopta enfoque **cuantitativo aplicado orientado al desarrollo de soluciones tecnológicas** (data-driven engineering), donde el objetivo central no es teorización abstracta sino creación de herramienta operacional funcional que resuelva problema concreto de gestión hídrica urbana.

**Características del paradigma adoptado:**

1. **Cuantitativo:** Toda validación se basa en métricas numéricas objetivas (R², MAE, RMSE, MAPE) calculadas sobre conjuntos de datos independientes, sin interpretaciones subjetivas de desempeño.

2. **Aplicado:** Solución desarrollada (interfaz Gradio) está diseñada para uso operacional real por personal técnico de empresas sanitarias, no como ejercicio académico teórico.

3. **Data-driven:** Decisiones de diseño (selección de algoritmos, ingeniería de características, arquitectura de datos) se fundamentan en análisis empírico de patrones detectados en dataset de 15.222 registros horarios.

4. **Experimental comparativo:** Sistema implementa tres algoritmos ML simultáneamente (XGBoost, Random Forest, LightGBM) para validación cruzada, siguiendo principio de robustez mediante convergencia de métodos independientes.

### **3.1.2. Diseño metodológico: exploratorio + correlacional + experimental**

El trabajo integra tres diseños complementarios:

**Fase exploratoria (Capítulo 4 - Análisis de información):**
- Visualización de series temporales para identificar patrones estacionales, ciclos horarios/semanales/mensuales
- Detección de anomalías y eventos atípicos (picos de consumo, caídas súbitas)
- Análisis de correlaciones bivariadas entre variables climáticas y demanda
- Caracterización de distribuciones estadísticas (normalidad, estacionalidad, autocorrelación)

**Fase correlacional (Capítulo 3 - Ingeniería de características):**
- Cuantificación de relaciones no lineales mediante análisis de importancia de features (feature importance) calculado por algoritmos ensemble
- Identificación de variables predictoras más relevantes (temperatura, humedad, calendario social, lags temporales)
- Validación empírica de hipótesis sobre factores que influyen en demanda urbana (secciones 2.4-2.5 del Marco Teórico)

**Fase experimental (Capítulo 3 - Modelamiento predictivo):**
- Entrenamiento controlado de tres algoritmos ML con misma partición de datos (train/validation/test 70/15/15)
- Comparación rigurosa de desempeño mediante métricas estandarizadas
- Validación temporal (no aleatoria) para simular uso operacional real: modelos entrenados con datos pasados predicen futuro sin "mirar hacia adelante"
- Prueba de concepto mediante interfaz Gradio operando sobre dataset completo

### **3.1.3. Justificación del enfoque ML supervisado**

La elección de machine learning supervisado basado en árboles de decisión ensemble (XGBoost, Random Forest, LightGBM) se fundamenta en:

**Superioridad sobre métodos estadísticos clásicos:**
- ARIMA/SARIMA requieren estacionaridad (transformaciones previas), identificación manual de órdenes (p,d,q), suponen linealidad en relaciones
- ML captura automáticamente relaciones no lineales complejas (temperatura vs demanda no es lineal), interacciones entre variables (evento social + fin de semana + temperatura alta), sin especificación manual de forma funcional

**Capacidad multi-variable nativa:**
- Modelos tradicionales univariables (ARIMA) modelan solo demanda histórica
- ML integra simultáneamente 71 features de dominios heterogéneos (clima, calendario social, operación, patrones temporales) sin requerir homogeneización de escalas

**Robustez a datos faltantes y outliers:**
- Árboles de decisión manejan naturalmente registros con valores faltantes mediante surrogate splits
- Ensemble methods reducen impacto de observaciones atípicas mediante promediado/votación

**Justificación de horizonte 72 horas:**
- Ventana crítica para decisiones operacionales tácticas (ajuste producción plantas, programación bombeos, gestión almacenamiento)
- Balance entre utilidad práctica y fiabilidad predictiva: horizontes >7 días requieren proyecciones climáticas de menor precisión con utilidad marginal decreciente para gestión diaria

### **3.1.4. Alineación con objetivos de investigación**

El enfoque metodológico responde directamente a los objetivos planteados:

**Objetivo general:** "Desarrollar sistema predictivo de demanda hídrica urbana mediante machine learning para planificación táctica en Gran Valparaíso"
- **Metodología:** Entrenamiento de modelos ML sobre datos operacionales reales, implementación en interfaz web interactiva Gradio, validación mediante métricas cuantitativas sobre conjunto de prueba independiente

**Objetivos específicos:**
1. Integrar variables multi-dominio (clima, calendario social, operación) → **Ingeniería de 71 features** documentada en sección 3.4
2. Comparar algoritmos ML para identificar óptimo → **Validación experimental** con RF/XGBoost/LightGBM en sección 3.5
3. Desarrollar herramienta accesible para operadores → **Interfaz Gradio 7 módulos** descrita en sección 3.6
4. Validar precisión en caso real Gran Valparaíso → **Dataset 21 meses, 15.222 registros**, resultados en Capítulo 4

---

## **3.2. ARQUITECTURA DE DATOS Y PIPELINE MODULAR**

### **3.2.1. Principios de Data-Oriented Design (DOD)**

El sistema implementa arquitectura orientada a datos (Data-Oriented Design, Zaharia et al., 2016) que prioriza **calidad, trazabilidad y reproducibilidad** del procesamiento sobre optimizaciones prematuras de código. Este enfoque es estándar en ingeniería de datos moderna para proyectos de machine learning (Wilkinson et al., 2016).

**Principios DOD aplicados al sistema desarrollado:**

1. **Separación clara de capas:** Datos crudos (Raw), procesados (Curated), y features ML (Feature Store) residen en directorios independientes con responsabilidades diferenciadas.

2. **Inmutabilidad de fuentes originales:** Archivos en capa Raw nunca se modifican; toda transformación genera nuevo archivo en capa superior, preservando trazabilidad completa hacia fuentes originales.

3. **Idempotencia de transformaciones:** Ejecutar pipeline múltiples veces con mismos inputs produce siempre mismo output, requisito fundamental para reproducibilidad científica.

4. **Metadatos explícitos:** Archivo `sistema_info_v3.json` documenta proveniencia de datos, fecha procesamiento, registros válidos/descartados, permitiendo auditoría completa.

### **3.2.2. Estructura de capas Raw → Curated → Feature Store**

El pipeline implementa arquitectura de tres capas inspirada en medallion architecture (Databricks) y principios de Tidy Data (Wickham, 2014):

```
Tesis3.0-Interfaz/
├── data/
│   ├── raw/                                    # CAPA BRONZE: Datos originales inmutables
│   │   ├── BD_VolTotal_X_Hr_m3_UTC.csv        # Volumen total sistema (ESVAL)
│   │   ├── BD_Clima2024a202509_UTC.csv        # Clima Rodelillo (DMC Chile)
│   │   ├── calendar_social_ES_COMPLETO_...    # Calendario social Chile 2024-2025
│   │   ├── BD_Qin_m3_UTC.csv                  # Producción horaria (ESVAL)
│   │   └── Vol_X_TK_Hr_m3_UTC.csv             # Volúmenes por tanque (ESVAL)
│   │
│   ├── processed/                              # CAPA SILVER: Datos limpios estandarizados
│   │   ├── data_processed_complete.csv        # Dataset integrado con 71 features
│   │   ├── data_train.csv                     # Conjunto entrenamiento (70%)
│   │   ├── data_validation.csv                # Conjunto validación (15%)
│   │   ├── data_test.csv                      # Conjunto prueba (15%)
│   │   ├── volumen_total_chile_v3.csv         # Serie temporal demanda limpia
│   │   ├── clima_chile_v3.csv                 # Serie temporal clima sincronizada
│   │   ├── produccion_chile_v3.csv            # Serie temporal producción
│   │   ├── metricas_diarias_v3.json           # Estadísticas descriptivas diarias
│   │   └── sistema_info_v3.json               # Metadatos pipeline completo
│   │
│   └── models/                                 # CAPA GOLD: Artefactos ML listos producción
│       └── gradio/
│           ├── xgboost_model_v3_0.pkl         # Modelo XGBoost serializado
│           ├── rf_model_v3_0.pkl              # Modelo Random Forest serializado
│           ├── lgb_model_v3_0.pkl             # Modelo LightGBM serializado
│           └── features.txt                    # Lista ordenada 71 features
```

**Figura 3.1.** Estructura de directorios del pipeline de datos implementado.

**Justificación de la arquitectura:**

- **Capa Raw (Bronze):** Archivos CSV originales con timestamps UTC, sin transformaciones. Permite replicar análisis completo desde fuentes primarias.
  
- **Capa Processed (Silver):** Integración multi-fuente mediante joins temporales (UTC ISO-8601), generación de 71 features, partición train/val/test temporal (no aleatoria), limpieza de 114 registros malformados.
  
- **Capa Models (Gold):** Modelos entrenados serializados en formato pickle, listos para inferencia en producción sin reentrenamiento.

### **3.2.3. Estandarización temporal UTC ISO-8601 / RFC 3339**

**Decisión crítica:** Todo el sistema opera con timestamps en **UTC (Coordinated Universal Time)** siguiendo estándares ISO 8601:2019 y RFC 3339 (Klyne & Newman, 2002).

**Motivación:**

Datos originales provienen de tres fuentes con posibles inconsistencias de zona horaria:
1. ESVAL (operación local): Potencial uso de Chile Continental (UTC-3 verano, UTC-4 invierno con cambio horario)
2. DMC Chile (clima Rodelillo): Registros oficiales meteorológicos en UTC
3. Calendario social: Eventos en hora local Chile

**Solución implementada:**

```python
# Ejemplo de conversión aplicada en pipeline (pseudo-código)
import pandas as pd
from datetime import timezone

# 1. Parsear timestamp original asumiendo UTC si no especifica zona
df['timestamp'] = pd.to_datetime(df['timestamp_str'], utc=True)

# 2. Convertir a UTC explícitamente si viene en otra zona
df['timestamp_utc'] = df['timestamp'].dt.tz_convert('UTC')

# 3. Formatear salida RFC 3339 con 'Z' (Zulu = UTC)
# Ejemplo: "2024-07-15T14:30:00Z"
df['timestamp_str'] = df['timestamp_utc'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
```

**Ventajas de UTC:**

1. **Elimina ambigüedad de cambios horarios:** Chile tiene cambio horario verano/invierno; UTC es constante todo el año, evita duplicaciones/saltos en serie temporal.

2. **Facilita integración multi-fuente:** Join temporal entre datasets requiere referencia temporal común; UTC es estándar universal.

3. **Simplifica sincronización con APIs externas:** Open-Meteo API devuelve datos en UTC; no requiere conversiones adicionales.

4. **Compatibilidad con herramientas modernas:** Pandas, NumPy, scikit-learn, XGBoost operan nativamente con timezone-aware datetime en UTC.

**Trade-off aceptado:** Visualizaciones en interfaz Gradio muestran UTC, no hora local Chile. Operadores deben aplicar conversión mental UTC-3/-4. Alternativa futura: agregar toggle "mostrar hora local" en interfaz (no implementado en prototipo).

### **3.2.4. Principios FAIR y Tidy Data**

El pipeline sigue dos frameworks metodológicos complementarios para calidad de datos:

#### **Principios FAIR (Wilkinson et al., 2016):**

- **Findable (Encontrable):** Todos los archivos en `data/raw/` tienen nombres descriptivos con sufijos de versión (`_v3`, `_UTC`), metadatos en `sistema_info_v3.json` documentan proveniencia.

- **Accessible (Accesible):** Formato CSV universal legible por cualquier herramienta (Excel, Python, R), sin dependencias de software propietario.

- **Interoperable (Interoperable):** Timestamps ISO-8601, unidades explícitas en nombres de columnas (`Vol_Total_m3`, `Temp_degC`), sin códigos ambiguos.

- **Reusable (Reutilizable):** Licencia MIT en repositorio GitHub permite reutilización académica/comercial, documentación en README.md explica estructura.

#### **Tidy Data (Wickham, 2014):**

Todos los datasets en capa Processed siguen formato tidy:

1. **Cada variable es una columna:** `timestamp`, `Vol_Total_m3`, `Temp_degC`, `tiene_evento_social`, `hora_sin`, `lag_24h`, etc.

2. **Cada observación es una fila:** Registro horario único identificado por timestamp.

3. **Cada tipo de unidad observacional es una tabla:** `data_train.csv` contiene observaciones de entrenamiento, `data_test.csv` observaciones de prueba, sin mezcla.

**Ejemplo de transformación a Tidy Data:**

**Formato NO tidy (común en Excel):**
```
Fecha       | Hora | Vol_m3 | Temp | Humedad
2024-07-15  | 00:00| 8234   | 12.5 | 78
2024-07-15  | 01:00| 7892   | 12.1 | 80
```

**Formato Tidy implementado:**
```
timestamp_utc        | Vol_Total_m3 | Temp_degC | Humedad_pct | tiene_evento_social | hora_sin | lag_24h
2024-07-15T00:00:00Z | 8234         | 12.5      | 78          | 0                   | 0.0      | 8456
2024-07-15T01:00:00Z | 7892         | 12.1      | 80          | 0                   | 0.2588   | 8123
```

**Ventajas operacionales:**

- Filtrado directo: `df[df['tiene_evento_social'] == 1]` obtiene todos los registros con eventos sociales
- Agregación sencilla: `df.groupby('hora')['Vol_Total_m3'].mean()` calcula demanda promedio por hora del día
- Joins automáticos: pandas merge por `timestamp_utc` sin transformaciones previas

### **3.2.5. Tratamiento de datos faltantes y registros malformados**

**Problemática identificada:** Durante carga de archivos Raw, se detectaron **114 registros malformados** (0.74% del dataset total de 15.336 registros iniciales) con:
- Timestamps duplicados o fuera de secuencia temporal esperada
- Valores numéricos negativos en variables físicas positivas (volumen, temperatura)
- Campos vacíos en columnas críticas (`Vol_Total_m3`, `Temp_degC`)

**Protocolo de limpieza implementado:**

1. **Detección automática:** Scripts Python validan rangos físicos razonables (volumen 0-50.000 m³, temperatura -5°C a +45°C, humedad 0-100%)

2. **Registro de anomalías:** Archivo `sistema_info_v3.json` documenta timestamps específicos descartados con razón de eliminación

3. **Eliminación sin imputación:** Registros malformados se descartan completamente (no se imputan valores sintéticos), manteniendo principio de integridad de datos reales

4. **Interpolación limitada para gaps menores:** Si faltan 1-2 horas consecutivas (NO 114 registros dispersos), se aplica interpolación lineal simple solo si gap <3 horas

**Resultado final:** Dataset limpio con **15.222 registros válidos** (99.26% de datos originales) distribuidos:
- Train: 10.655 registros (70%)
- Validation: 2.283 registros (15%)
- Test: 2.284 registros (15%)

**Transparencia metodológica:** Esta limpieza se documenta explícitamente en sección 2.8.3 (Limitación 3) del Marco Teórico, reconociendo que modelo no fue validado con registros extremos/anómalos.

---

## **3.2.6. Comparación con enfoque alternativo: archivo único vs pipeline modular**

**Justificación de arquitectura multi-archivo:**

| Criterio | Archivo único gigante | Pipeline modular (implementado) |
|----------|----------------------|----------------------------------|
| **Trazabilidad** | Difícil rastrear transformaciones aplicadas | Clara separación Raw→Processed→Models |
| **Reproducibilidad** | Requiere re-procesar todo si cambia un paso | Caché intermedio: solo reejecutar pasos modificados |
| **Colaboración** | Conflictos Git en archivo grande | Archivos pequeños con merge sencillo |
| **Validación** | Difícil inspeccionar datos intermedios | Cada capa validable independientemente |
| **Escalabilidad** | Límites de memoria con datasets >1GB | Procesamiento incremental por capas |

**Costo aceptado:** Mayor número de archivos (17 archivos en `data/processed/` vs potencial 1 archivo). Mitigado mediante documentación clara en README.md y convenciones de nombres descriptivas.

---

**Referencias citadas en esta sección:**

- ISO 8601. (2019). *Date and time — Representations for information interchange*. International Organization for Standardization.
- Klyne, G., & Newman, C. (2002). *RFC 3339: Date and Time on the Internet: Timestamps*. Internet Engineering Task Force (IETF).
- Wickham, H. (2014). Tidy Data. *Journal of Statistical Software*, 59(10), 1-23. https://doi.org/10.18637/jss.v059.i10
- Wilkinson, M. D., Dumontier, M., Aalbersberg, I. J., et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data*, 3, 160018. https://doi.org/10.1038/sdata.2016.18
- Zaharia, M., Xin, R. S., Wendell, P., Das, T., Armbrust, M., Dave, A., ... & Stoica, I. (2016). Apache Spark: A unified engine for big data processing. *Communications of the ACM*, 59(11), 56-65.

---

**Fin de Parte 1 - SECCIONES 3.1 + 3.2 (~1.700 palabras)**

**Próximas secciones:**
- 3.3: Fuentes de datos y preprocesamiento (~1.000 palabras)
- 3.4: Ingeniería de características (~800 palabras)
- 3.5: Modelamiento predictivo (~1.200 palabras)
- 3.6: Desarrollo interfaz Gradio (~1.000 palabras)
- 3.7: Consideraciones éticas y limitaciones (~500 palabras)
- 3.8: Recursos tecnológicos (~300 palabras)
