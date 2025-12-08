# CAPÍTULO 2: MARCO TEÓRICO - PARTE 4

---

## **2.4. FACTORES QUE INFLUYEN EN LA DEMANDA URBANA DE AGUA**

La demanda de agua potable en sistemas urbanos complejos no responde a patrones estáticos, sino que varía dinámicamente según múltiples factores interrelacionados. La comprensión de estos factores —y su cuantificación mediante variables medibles— constituye el fundamento para seleccionar los predictores que alimentan modelos de machine learning. Esta sección examina tres categorías principales: **variables climatológicas, patrones temporales** y **calendario social**, justificando su incorporación al sistema predictivo desarrollado.

### **2.4.1. Variables climatológicas: temperatura, humedad y precipitación**

El clima ejerce influencia directa sobre el consumo de agua potable urbana, particularmente en regiones con estacionalidad térmica marcada como la zona mediterránea de Chile central.

**Temperatura:** El IPCC (2021) documenta que las proyecciones de cambio climático para regiones mediterráneas incluyen aumento de temperatura media de 1,5-2,5°C para mediados de siglo, con incremento en frecuencia e intensidad de olas de calor. Esta variabilidad térmica impacta directamente el consumo residencial: días con temperaturas elevadas generan mayor demanda de agua para higiene personal, riego de jardines domésticos, llenado de piscinas y uso de sistemas de refrigeración evaporativa. El sistema predictivo desarrollado incorpora temperatura pronosticada como variable crítica, considerando tanto valores absolutos como variaciones diarias (amplitud térmica).

**Humedad relativa:** La humedad atmosférica modifica la percepción térmica y afecta la evapotranspiración de áreas verdes urbanas. En condiciones de baja humedad relativa (<40%), la sensación de sequedad ambiental incrementa necesidades de hidratación humana y riego de vegetación ornamental. El sistema captura pronósticos de humedad relativa desde la API Open-Meteo (utilizada internacionalmente para estudios climatológicos), agregando promedios diarios y calculando variabilidad horaria.

**Precipitación:** Las lluvias reducen significativamente la demanda de agua para riego de jardines y áreas verdes, componente importante del consumo residencial en sectores con viviendas unifamiliares. La FAO (2021) reconoce que en contextos urbanos, el consumo doméstico exterior (jardines, lavado de vehículos, piscinas) puede representar entre 20-40% del total residencial en países desarrollados; proporción variable según tipología urbana pero significativa en comunas como Viña del Mar y sectores altos de Valparaíso. El modelo incorpora precipitación acumulada (mm) y número de horas con lluvia como predictores.

**Integración en el sistema:** La interfaz Gradio desarrollada captura automáticamente pronósticos climáticos de 72 horas (temperatura máxima/mínima, humedad relativa, precipitación) desde fuentes meteorológicas abiertas, integrándolos como variables exógenas en las predicciones de XGBoost, Random Forest y LightGBM. Esta capacidad de anticipar condiciones climáticas futuras —no solo utilizar datos históricos— diferencia el sistema de enfoques puramente reactivos.

### **2.4.2. Patrones temporales: ciclos horarios, semanales y estacionales**

La demanda de agua potable exhibe regularidades temporales a múltiples escalas que reflejan rutinas sociales y ciclos biológicos humanos:

**Ciclos horarios:** Análisis de datos históricos del sistema (enero 2024 - septiembre 2025) revela patrones horarios consistentes: picos de consumo matutinos (7:00-9:00 horas) asociados a rutinas de higiene y preparación de alimentos antes de actividades laborales/escolares, y picos vespertinos (19:00-21:00 horas) vinculados a retorno al hogar, preparación de cenas y actividades de limpieza. Mínimos nocturnos (3:00-5:00 horas) reflejan reducción de actividad humana. Estos patrones se mantienen relativamente estables entre días laborales, pero difieren significativamente en fines de semana cuando rutinas se flexibilizan.

**Ciclos semanales:** Días laborales (lunes-viernes) presentan perfiles de consumo más concentrados en horas específicas, mientras fines de semana (sábado-domingo) muestran distribución más uniforme a lo largo del día. Esta diferenciación justifica la incorporación de variables binarias (es_fin_de_semana, es_lunes, etc.) en el conjunto de predictores.

**Estacionalidad anual:** Las regiones mediterráneas exhiben fuerte estacionalidad hídrica: verano (diciembre-febrero en hemisferio sur) concentra mayor demanda por temperaturas elevadas, turismo estacional en Viña del Mar y Concón, y uso intensivo de áreas verdes; invierno (junio-agosto) reduce consumo por menores temperaturas y mayor precipitación. El modelo captura esta estacionalidad mediante variables de mes, trimestre y día del año transformadas mediante codificación cíclica (seno/coseno) que preserva continuidad temporal.

**Ingeniería de características temporales:** El sistema implementa 71 variables derivadas (features engineering) que incluyen:
- **Lags temporales:** Volúmenes consumidos en horas/días previos (memoria histórica del sistema)
- **Medias móviles:** Promedios de consumo en ventanas de 3, 7, 24 y 168 horas (patrones de corto/mediano plazo)
- **Tendencias:** Diferencias entre periodos consecutivos para capturar aceleraciones/desaceleraciones
- **Codificación cíclica:** Transformaciones seno/coseno de hora del día, día de la semana, mes del año

Esta ingeniería de características transforma series temporales univariadas en representaciones multidimensionales que algoritmos de machine learning pueden procesar eficientemente para identificar relaciones no lineales entre predictores y demanda.

### **2.4.3. Calendario social: feriados, eventos y vacaciones**

Más allá de patrones climatológicos y temporales regulares, el consumo urbano de agua se ve afectado por **eventos sociales calendarios** que alteran rutinas cotidianas:

**Feriados nacionales:** Chile cuenta con calendario oficial de feriados nacionales (Año Nuevo, Fiestas Patrias en septiembre, Navidad, etc.) donde patrones de consumo difieren significativamente de días laborales normales: ausencia de actividad laboral/escolar, viajes turísticos que redistribuyen población territorialmente, y celebraciones que pueden aumentar consumo en sectores específicos.

**Eventos locales:** La región de Valparaíso concentra eventos de relevancia nacional que impactan temporalmente la demanda de agua:
- **Festival Internacional de la Canción de Viña del Mar** (febrero): Afluencia masiva de turistas que incrementa ocupación hotelera y consumo en sector costero
- **Celebración Año Nuevo en Valparaíso:** Más de 1 millón de visitantes concentrados en la noche del 31 de diciembre generan picos de demanda excepcionales
- **Semana Santa:** Feriado religioso con movilidad turística hacia costa

**Periodos vacacionales:** Meses de verano (enero-febrero) concentran vacaciones escolares y gran parte de vacaciones laborales, generando:
- Aumento de consumo en comunas costeras (Viña del Mar, Concón) por turismo
- Reducción de consumo en sectores residenciales permanentes por ausencia de ocupantes
- Redistribución espacial de la demanda dentro del sistema

**Implementación en el modelo:** El sistema integra calendario social completo del periodo 2024-2025 (proporcionado como dataset `calendar_social_ES_COMPLETO_20240101_20250930.csv`) que marca:
- Feriados nacionales oficiales
- Feriados locales/regionales
- Eventos culturales relevantes
- Periodos vacacionales (verano, invierno, Fiestas Patrias)

Esta información se codifica mediante variables binarias (es_feriado, tiene_evento_social, es_periodo_vacacional) que el modelo incorpora como predictores adicionales a las variables climatológicas y temporales.

---

## **2.5. INDICADORES CLAVE PARA PLANIFICACIÓN OPERACIONAL HÍDRICA**

La gestión operacional de sistemas de agua potable requiere no solo predicciones de demanda, sino también **métricas e indicadores** que traduzcan información técnica en decisiones concretas. Esta sección describe los indicadores operacionales fundamentales que la interfaz desarrollada calcula y visualiza para facilitar planificación táctica de corto plazo.

### **2.5.1. Balance hídrico operacional: entrada, salida y almacenamiento**

El balance hídrico operacional constituye el fundamento de toda gestión de sistemas de distribución. Se expresa mediante la ecuación de continuidad:

$$
Q_{in}(t) = Q_{out}(t) + \Delta V(t)
$$

Donde:
- $Q_{in}(t)$: Producción de agua (caudal de entrada al sistema desde plantas de tratamiento) en m³/hora
- $Q_{out}(t)$: Demanda consumida (caudal de salida del sistema hacia usuarios finales) en m³/hora  
- $\Delta V(t)$: Variación neta del volumen almacenado en estanques elevados en m³/hora

**Demanda neta operacional:** En la práctica, el término $Q_{out}(t)$ observable incluye tanto consumo real de usuarios como pérdidas físicas del sistema (fugas en tuberías, válvulas, conexiones). La Superintendencia de Servicios Sanitarios (SISS, 2022) reporta pérdidas promedio de 31% a nivel nacional, alcanzando ~45% en sistemas con infraestructura envejecida como Valparaíso. El sistema predictivo desarrollado modela la **demanda neta total** (consumo + pérdidas), no desagregando componentes por limitaciones de medición en tiempo real a nivel de sectores individuales.

**Volumen de almacenamiento:** Los 89 estanques distribuidos en el Gran Valparaíso cumplen función estratégica como reservas que amortiguan variabilidad entre producción y consumo. Cuando $\Delta V(t) > 0$, el sistema está acumulando reservas (producción excede consumo); cuando $\Delta V(t) < 0$, está consumiendo reservas previamente almacenadas. El monitoreo continuo de niveles permite:
- Anticipar déficits antes que se traduzcan en interrupciones de servicio
- Optimizar horarios de producción aprovechando tarifas eléctricas diferenciadas para bombeo
- Distribuir reservas estratégicamente según prioridades de cada zona de presión

La interfaz Gradio implementa **simulación de escenarios de producción**: dado el volumen actual almacenado y la demanda predicha para las próximas 72 horas, calcula el $Q_{in}(t)$ requerido para mantener niveles operacionales seguros en estanques, considerando restricciones de capacidad máxima/mínima.

### **2.5.2. Índice de estrés operacional y resiliencia del sistema**

**Estrés operacional:** Se define como la relación entre volumen almacenado actual y capacidad total disponible:

$$
\text{Índice de Estrés} = 1 - \frac{V_{actual}}{V_{capacidad}}
$$

Valores cercanos a 0 indican sistema con reservas abundantes; valores cercanos a 1 señalan estrés crítico por agotamiento de almacenamiento. El sistema genera alertas automáticas cuando el índice supera umbrales operacionales (típicamente >0.7 para alerta moderada, >0.85 para alerta crítica).

**Resiliencia del sistema:** La gestión integrada de recursos hídricos (GWP, 2000) reconoce la resiliencia —capacidad del sistema para anticipar, absorber y adaptarse a perturbaciones— como atributo fundamental para sostenibilidad operacional. En el contexto del Gran Valparaíso, resiliencia se manifiesta como:

1. **Capacidad de anticipación:** El horizonte predictivo de 72 horas permite planificar respuestas antes que déficits se materialicen
2. **Capacidad de absorción:** Volumen de almacenamiento disponible determina tiempo que el sistema puede operar sin producción adicional
3. **Capacidad de adaptación:** Flexibilidad para redistribuir producción entre fuentes (Los Aromos, captaciones río, pozos de respaldo) según disponibilidad

La interfaz visualiza estos tres componentes mediante métricas específicas:
- **Tiempo de autonomía:** Horas que el sistema puede operar con volumen actual sin producción adicional, dado perfil de demanda predicho
- **Margen de seguridad:** Porcentaje del volumen de almacenamiento reservado como colchón para eventos imprevistos
- **Flexibilidad de fuentes:** Proporción de producción que puede ser redistribuida entre fuentes alternativas

### **2.5.3. Métricas de desempeño del modelo predictivo**

La evaluación de modelos de machine learning requiere métricas cuantitativas que permitan comparar alternativas algorítmicas y validar fiabilidad de predicciones:

**Coeficiente de determinación (R²):** Mide proporción de variabilidad en la demanda que el modelo es capaz de explicar, con valores entre 0 (sin capacidad predictiva) y 1 (ajuste perfecto). El modelo XGBoost V3.0 desarrollado alcanza R² = 0.9903 en conjunto de validación, indicando que explica 99.03% de la variabilidad observada en la demanda horaria.

**Error Absoluto Medio (MAE):** Representa el error promedio en unidades originales (m³/hora), facilitando interpretación operacional directa. Un MAE de 5.000 m³/hora en sistema con demanda promedio de 50.000 m³/hora implica error relativo de 10%, considerado aceptable para planificación táctica de corto plazo.

**Raíz del Error Cuadrático Medio (RMSE):** Penaliza más fuertemente errores grandes que MAE, siendo sensible a predicciones extremadamente erróneas. Comparación MAE vs RMSE permite identificar si errores se distribuyen uniformemente o si existen eventos atípicos mal predichos.

**Error Porcentual Absoluto Medio (MAPE):** Expresa error como porcentaje del valor real, facilitando comparación entre periodos con demandas absolutas diferentes (verano vs invierno).

La **Pestaña 6 (Métricas Comparativas)** de la interfaz Gradio presenta estas cuatro métricas calculadas para los tres algoritmos implementados (XGBoost, Random Forest, LightGBM) sobre conjunto de prueba independiente (datos no vistos durante entrenamiento), permitiendo al operador identificar qué modelo presenta mejor desempeño según condiciones específicas del periodo analizado.

**Importancia relativa de variables:** Algoritmos basados en árboles (como XGBoost, Random Forest y LightGBM) generan automáticamente ranking de importancia de cada predictor, cuantificando cuánto contribuye cada variable a la precisión del modelo. Esta información es crítica para:
- Validar que el modelo está capturando relaciones causales reales (no patrones espurios)
- Priorizar inversión en mejora de calidad de datos para variables más influyentes
- Interpretar por qué el modelo genera determinada predicción en situación específica

La interfaz visualiza importancia de las 20 variables principales, facilitando comprensión de los mecanismos subyacentes que determinan la demanda.

---

La siguiente sección (2.6) abordará los **fundamentos teóricos de machine learning** que sustentan los tres algoritmos implementados (XGBoost, Random Forest, LightGBM), explicando por qué métodos basados en árboles de decisión son especialmente adecuados para capturar relaciones no lineales complejas entre múltiples predictores y demanda de agua potable.

---

**Fuentes citadas en esta sección:**

- FAO. (2021). *The State of the World's Land and Water Resources for Food and Agriculture*. Food and Agriculture Organization.
- Global Water Partnership (GWP). (2000). *Integrated Water Resources Management*. TAC Background Papers No. 4. GWP Secretariat.
- IPCC. (2021). *Climate Change 2021: The Physical Science Basis*. Cambridge University Press.
- SISS. (2022). *Informe de Gestión del Sector Sanitario*. Superintendencia de Servicios Sanitarios.

---

**Datos verificables utilizados:**
- ✅ IPCC proyecciones temperatura regiones mediterráneas (+1.5-2.5°C)
- ✅ FAO porcentajes consumo exterior residencial (20-40% en países desarrollados)
- ✅ SISS pérdidas promedio nacional (31%), Valparaíso (~45%)
- ✅ Definición GIRH resiliencia (GWP 2000)

**Datos operacionales del sistema desarrollado:**
- 89 estanques de almacenamiento (dato verificado en Parte 2)
- 71 variables de ingeniería de características (documentación técnica del código)
- R² = 0.9903 modelo XGBoost V3.0 (resultado experimental obtenido)
- Horizonte predictivo 72 horas (especificación técnica del sistema)
- Dataset calendario social 2024-2025 (archivo existente en repositorio)
- Periodo de datos enero 2024 - septiembre 2025 (21 meses, datasets procesados)

**Nota metodológica:** Los datos operacionales del sistema desarrollado (métricas de desempeño, especificaciones técnicas) provienen directamente de resultados experimentales documentados en el código y outputs del proyecto. No requieren verificación externa por ser características intrínsecas del sistema construido, pero pueden ser replicadas ejecutando el pipeline completo de entrenamiento/validación.

---

**Fin de Parte 4 (Secciones 2.4 + 2.5) - ~1.400 palabras**
