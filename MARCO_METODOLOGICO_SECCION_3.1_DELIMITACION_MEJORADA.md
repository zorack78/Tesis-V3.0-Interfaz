# CAPÍTULO 3: MARCO METODOLÓGICO - SECCIÓN 3.1

## **3.1. DELIMITACIÓN DE LA INVESTIGACIÓN**

Esta investigación desarrolla un **modelo predictivo de demanda horaria de agua potable** mediante técnicas de aprendizaje automático aplicadas a series temporales, con el propósito de apoyar la planificación operativa y táctica de producción y almacenamiento en el sistema urbano de abastecimiento del Gran Valparaíso.

### **3.1.1. Delimitación temática**

**Objeto de estudio:**

Predicción de demanda horaria de agua potable en sistema urbano utilizando algoritmos de machine learning (Random Forest, XGBoost, LightGBM) que integran tres categorías de predictores:

1. **Variables climáticas:** Temperatura, humedad, precipitación, velocidad del viento (Estación DMC Rodelillo, Code 330007)
2. **Variables de calendario social:** Feriados nacionales/regionales, eventos masivos, vacaciones escolares (fuentes oficiales Ley 19.973, MINEDUC)
3. **Variables operacionales del sistema:** Producción horaria, volumen almacenado, balance hídrico (registros SCADA anonimizados)

**Alcance temático:**

El estudio se enfoca en la **predicción de demanda agregada** a nivel sistema completo, no en modelación hidráulica detallada de red ni análisis socioeconómico desagregado.

**Exclusiones explícitas:**

- ❌ Modelación hidráulica de red (presiones, caudales por tramo, pérdidas localizadas, sectorización hidráulica)
- ❌ Demanda no potable (riego agrícola, procesos industriales externos al sistema urbano)
- ❌ Análisis socioeconómico desagregado por estratos, sectores geográficos o barrios específicos
- ❌ Optimización de políticas operacionales (control óptimo de bombas, algoritmos de sectorización dinámica)
- ❌ Evaluación de infraestructura física (diseño de nuevos estanques, ampliación de red, reemplazo de tuberías)

**Justificación del enfoque:**

La agregación espacial a nivel sistema completo permite: (a) cumplir con protocolo de anonimización de datos empresariales (Ley 19.628), (b) desarrollar modelo robusto con datos disponibles sin requerir instrumentación adicional costosa, (c) proporcionar herramienta de planificación táctica 24-72 horas útil para operadores sin complejidad computacional excesiva.

### **3.1.2. Delimitación espacial**

**Área geográfica de estudio:**

Sistema urbano de agua potable del **Gran Valparaíso**, compuesto por:

- **Cobertura municipal:** Valparaíso, Viña del Mar, Concón, Quilpué, Villa Alemana, Casablanca (6 comunas)
- **Población abastecida:** ~1.000.000 habitantes (Censo 2017, INE)
- **Infraestructura:** 89 estanques de regulación con capacidad agregada (dato anonimizado), red de distribución de múltiples zonas de presión
- **Topografía característica:** Sistema costero-cerros con desniveles 0-250 m.s.n.m., 42 cerros urbanos

**Nivel de agregación:**

El análisis trabaja con **variables agregadas a escala sistema completo**:

- ✅ **Volumen total almacenado:** Suma de todos los estanques del sistema
- ✅ **Producción total:** Caudal agregado de plantas de tratamiento que alimentan la red
- ✅ **Demanda total horaria:** Volumen consumido por sistema completo en cada hora

**Exclusiones espaciales:**

- ❌ Desagregación por sectores operacionales, zonas de presión o comunas individuales
- ❌ Análisis de estanques específicos (identificación, ubicación GPS, capacidad individual)
- ❌ Consumo diferenciado por tipo de cliente (residencial, comercial, industrial, municipal)

**Justificación de la agregación:**

La agregación espacial es requisito del protocolo de anonimización aplicado (ver sección 3.3.1) y garantiza que datos publicados no permiten identificar instalaciones específicas, sectores vulnerables o información comercial sensible de la empresa sanitaria.

### **3.1.3. Delimitación temporal**

**Período de análisis:**

- **Inicio:** 1 de enero de 2024, 03:00 UTC
- **Fin:** 30 de septiembre de 2025, 23:00 UTC
- **Duración:** 21 meses (15.223 registros horarios válidos tras limpieza)

**Cobertura estacional:**

El período captura eventos cíclicos relevantes para validación de modelo:

| Dimensión temporal | Cobertura | Justificación |
|-------------------|-----------|---------------|
| **Ciclos anuales** | 2 veranos completos (ene-feb 2024, ene-feb 2025)<br/>2 inviernos completos (jun-ago 2024, jun-ago 2025) | Captura estacionalidad climática y patrones de consumo verano-invierno |
| **Ciclos semanales** | 89 semanas completas | Diferencia patrones laborales (lunes-viernes) vs fin de semana |
| **Eventos sociales** | Fiestas Patrias 2024 (18-19 sep)<br/>Año Nuevo Valparaíso 2024-2025<br/>Festival Viña 2024-2025 | Valida capacidad de modelo para predecir consumo en eventos masivos |
| **Rango térmico** | Temperatura 8°C (invierno) - 32°C (verano) | Representa variabilidad climática típica región mediterránea costera |

**División temporal para modelamiento:**

Los 15.223 registros se dividen respetando **orden cronológico estricto** (split temporal, no aleatorio):

- **Entrenamiento:** 70% (10.655 registros) - ene 2024 a nov 2024 (~11 meses)
- **Validación:** 15% (2.283 registros) - nov 2024 a mar 2025 (~4 meses)
- **Prueba:** 15% (2.284 registros) - mar 2025 a sep 2025 (~6 meses)

**Limitaciones temporales reconocidas:**

- ⚠️ **Eventos extremos no capturados:** Megasequías (última 2010-2021), terremotos >8.0 (último 2010), cortes masivos de energía eléctrica
- ⚠️ **Tendencias de largo plazo:** 21 meses insuficientes para modelar cambios demográficos graduales, modificaciones infraestructura o impacto climático secular (>5 años)
- ⚠️ **Estacionalidad interanual:** No captura variabilidad climática entre años (ej: evento El Niño vs La Niña)

**Justificación del período:**

Pese a limitaciones, 21 meses con 15.223 registros horarios constituyen dataset suficiente para: (a) entrenamiento robusto de modelos ensemble (estudios referentes utilizan 12-18 meses, Breiman 2001; Chen & Guestrin 2016), (b) captura de estacionalidad anual completa, (c) validación con horizonte de prueba independiente de 6 meses que simula uso operacional real.

### **3.1.4. Delimitación operacional**

**Unidad de análisis:**

**Registro horario** del sistema de agua potable, caracterizado por:

- **Timestamp UTC:** Marca temporal ISO-8601 (ej: 2024-07-15T14:00:00Z)
- **Variable objetivo:** `Vol_Total_m3` - Volumen demandado por sistema en esa hora (m³/hora)
- **Variables predictoras:** 71 features generadas mediante ingeniería de características (ver sección 3.4):
  - 12 temporales cíclicas (hora_sin/cos, día_semana_sin/cos, mes_sin/cos)
  - 24 lags de demanda (1h, 2h...720h)
  - 12 rolling statistics (medias móviles 6h, 24h, 168h)
  - 10 climáticas derivadas (Temp_cambio_24h, dias_sin_lluvia)
  - 5 calendario social (tiene_evento_social, es_feriado, es_vacaciones)
  - 8 operacionales (Qin_lag_24h, Vol_almacenado_pct, balance_hidrico)

**Población de interés:**

Conjunto completo de registros horarios del sistema de distribución Gran Valparaíso durante enero 2024 - septiembre 2025, totalizando 15.223 observaciones válidas tras aplicar protocolo de limpieza (ver criterios inclusión/exclusión sección 3.4.1).

**Tipo de muestreo:**

**No aplica muestreo** en sentido estadístico tradicional. Se utiliza **censo completo** del período definido: todos los registros horarios disponibles en sistema SCADA que cumplen criterios de calidad (timestamps válidos, valores dentro de rangos físicos plausibles, sin campos vacíos críticos).

**Justificación:**

- Dataset corresponde a **población completa** de registros operacionales del período, no a muestra probabilística
- Exclusión de 114 registros (0.74%) se debe a problemas de integridad de datos (timestamps duplicados, valores negativos), no a diseño muestral
- Enfoque de **serie temporal completa** es requisito metodológico para modelos ML que aprenden patrones temporales (lags, autocorrelación, estacionalidad)

**Implicación para generalización:**

Resultados son directamente aplicables al sistema Gran Valparaíso durante período estudiado. **Generalización a otros sistemas urbanos** requiere:

1. Reentrenamiento con datos locales (patrones de consumo, clima, calendario social son específicos de cada región)
2. Validación que características del sistema (topografía, tamaño poblacional, infraestructura) sean comparables
3. Adaptación de calendario social a contexto cultural/turístico local

### **3.1.5. Alcance práctico y limitaciones**

**Alcance práctico de la investigación:**

1. **Producto principal:** Modelo predictivo operacional capaz de estimar demanda horaria con horizonte 24-72 horas y error promedio <5% (MAE <400 m³/h sobre demanda media ~8.000 m³/h)

2. **Herramienta de apoyo decisional:** Interfaz web interactiva (Gradio) con 7 módulos funcionales:
   - Predicción 72h con intervalos de confianza
   - Análisis exploratorio de patrones históricos
   - Pronóstico climático en tiempo real (Open-Meteo API)
   - Comparación de modelos (XGBoost, Random Forest, LightGBM)
   - Simulación de escenarios de producción
   - Métricas comparativas de desempeño
   - Exportación de resultados (CSV/Excel)

3. **Contribución metodológica:** Blueprint replicable de arquitectura Data-Oriented Design (Raw → Curated → Feature Store) y pipeline de feature engineering para sistemas urbanos similares

4. **Código abierto:** Repositorio público (post-defensa) con stack tecnológico open-source (Python, XGBoost, Gradio) accesible sin licencias comerciales costosas

**Limitaciones prácticas explícitas:**

| Aspecto | Sistema desarrollado (prueba de concepto) | Requisito producción empresarial |
|---------|-------------------------------------------|----------------------------------|
| **Usuarios simultáneos** | 1-3 (investigador + evaluadores) | 20-50 operadores ESVAL |
| **Disponibilidad** | Bajo demanda (ejecución local) | 24/7 con SLA >99.9% |
| **Integración datos** | Archivos CSV estáticos | API tiempo real SCADA + bases datos corporativas |
| **Auditoría** | Logs básicos Python | Trazabilidad completa ISO 27001, respaldos automáticos |
| **Validación regulatoria** | N/A (académico) | Aprobación SISS (Superintendencia Servicios Sanitarios) |
| **Soporte** | Documentación técnica | Mesa de ayuda, SLA de respuesta |

**Exclusiones del alcance práctico:**

- ❌ **Optimización operacional automática:** Sistema predice demanda pero no calcula automáticamente estrategia óptima de bombeo, llenado de estanques o sectorización
- ❌ **Control en tiempo real:** No reemplaza sistema SCADA existente, no envía comandos directos a válvulas/bombas
- ❌ **Gestión de eventos extremos:** No cubre protocolos de emergencia para terremotos, contaminación masiva, fallas catastróficas de infraestructura
- ❌ **Planificación estratégica de largo plazo:** No proyecta crecimiento demográfico >5 años, no evalúa inversiones en nueva infraestructura, no diseña expansión de red
- ❌ **Instrumentos económicos:** No define tarifas, subsidios, incentivos de ahorro o políticas de equidad social

**Justificación de limitaciones:**

Sistema constituye **prueba de concepto funcional** (proof of concept) que demuestra viabilidad técnica y científica de predicción mediante ML. Transición a sistema empresarial en producción requiere: (a) ingeniería de software adicional (arquitectura cliente-servidor, autenticación SSO, microservicios), (b) validación regulatoria con SISS, (c) plan de contingencia ante fallas, (d) capacitación formal de operadores (ver sección 3.7.3).

**Utilidad operacional inmediata:**

Pese a limitaciones, sistema desarrollado es directamente útil para:

- **Planificación táctica semanal:** Operador genera predicción cada lunes, ajusta cronograma de producción preventivamente
- **Análisis post-operacional:** Comparación entre demanda predicha vs real para detectar anomalías (fugas, medición errónea, eventos no documentados)
- **Validación de hipótesis operacionales:** Simulación de escenarios "what-if" (ej: "¿qué pasa si reduzco producción 10% durante madrugada?")
- **Comunicación gerencial:** Exportación de reportes con evidencia cuantitativa para justificar decisiones operacionales

---

**Referencias citadas en esta sección:**

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785-794.
- INE. (2017). *Censo de Población y Vivienda 2017*. Instituto Nacional de Estadísticas, Chile.
- Ley 19.628. (1999). *Sobre protección de la vida privada*. Biblioteca del Congreso Nacional de Chile.
- Ley 19.973. (2004). *Sobre el sistema de feriados legales en Chile*. Biblioteca del Congreso Nacional de Chile.

---

**Fin de Sección 3.1 - Delimitación de la investigación (~1.200 palabras)**

**Próximas secciones:**
- 3.2: Enfoque metodológico y arquitectura Data-Oriented Design
- 3.3: Fuentes de datos y preprocesamiento
- 3.4: Instrumentos de recopilación y análisis de información
- 3.5: Modelamiento predictivo con algoritmos ensemble
- 3.6: Desarrollo de interfaz Gradio operacional
- 3.7: Consideraciones éticas y limitaciones metodológicas
- 3.8: Recursos tecnológicos y reproducibilidad
