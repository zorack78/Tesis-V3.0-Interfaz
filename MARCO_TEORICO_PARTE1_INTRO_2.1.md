# CAPÍTULO 2: MARCO TEÓRICO

---

## **INTRODUCCIÓN**

La gestión del agua potable en entornos urbanos enfrenta desafíos crecientes debido al cambio climático y la urbanización acelerada. El agua es reconocida como derecho humano fundamental (ONU, Resolución 64/292, 2010) y constituye el eje del Objetivo de Desarrollo Sostenible 6 (ONU, 2015). En Chile, particularmente en la Región de Valparaíso, la sequía prolongada ha reducido drásticamente la disponibilidad del recurso (Escenarios Hídricos 2030, 2019), mientras que los métodos tradicionales de gestión —basados en proyecciones históricas y reglas operacionales estáticas— resultan insuficientes para abordar la alta variabilidad de la demanda.

Esta investigación desarrolla un **sistema predictivo operacional basado en machine learning** para la predicción de demanda de agua potable en el Gran Valparaíso con horizonte de 72 horas. El sistema combina tres modelos de aprendizaje supervisado (XGBoost, Random Forest, LightGBM) implementados en una **interfaz web interactiva Gradio** que integra:

- **Datos operacionales:** Producción horaria (Qin) de dos plantas principales y niveles de almacenamiento de 89 estanques distribuidos en múltiples zonas de presión diferenciadas por topografía
- **Variables climatológicas:** Temperatura, humedad relativa y precipitación
- **Calendario social chileno:** Feriados nacionales, eventos regionales (Festival de Viña del Mar) y períodos de vacaciones escolares
- **Ingeniería de características:** 71 variables predictivas generadas mediante análisis de patrones temporales (hora del día, día de semana, estacionalidad), históricos (lags, medias móviles) y de calendario

El estudio abarca **21 meses de datos continuos** (enero 2024 - septiembre 2025), capturando dos ciclos estacionales casi completos y garantizando consistencia temporal entre todas las fuentes de información (operacional, climatológica y calendario).

Esta investigación constituye una **prueba de concepto** orientada a demostrar la viabilidad técnica de modelos predictivos avanzados en gestión hídrica urbana chilena, con alcance acotado a predicción de corto plazo (72 horas) para planificación operacional táctica. Se reconocen limitaciones metodológicas —especialmente en validación de eventos extremos y simplificación del balance hídrico— que se abordarán como líneas de investigación futura.

El marco teórico se estructura en **ocho secciones** que fundamentan científicamente las decisiones técnicas del sistema: importancia del agua en ciudades (2.1), contexto de escasez hídrica en Chile y Valparaíso (2.2), marco conceptual GIRH (2.3), factores de demanda urbana (2.4), indicadores de planificación hídrica (2.5), fundamentos de machine learning aplicado (2.6), estado del arte internacional (2.7), y síntesis integradora con alcances y limitaciones (2.8). Esta estructura permite comprender no solo *qué* se investigó, sino *por qué* se tomaron las decisiones metodológicas específicas que conectan teoría con implementación práctica.

---

## **2.1. IMPORTANCIA DEL AGUA EN ENTORNOS URBANOS Y NECESIDAD DE HERRAMIENTAS PREDICTIVAS**

La gestión eficiente del agua potable en ciudades es fundamental para la salud pública, el desarrollo económico y la sostenibilidad ambiental. Con el 68% de la población mundial proyectada en áreas urbanas para 2050 (ONU-Habitat, 2022) y la intensificación del cambio climático, la administración del recurso hídrico urbano se ha convertido en un desafío estratégico crítico.

El agua es reconocida como derecho humano fundamental (ONU, Resolución 64/292, 2010) y constituye el Objetivo de Desarrollo Sostenible 6. Su importancia estratégica se manifiesta en cuatro dimensiones: **salud pública** (la OMS estima 485.000 muertes anuales por agua no segura), **desarrollo económico** (el Banco Mundial proyecta que la escasez hídrica podría reducir el PIB chileno en 6% para 2050), **estabilidad social** (la percepción de crisis erosiona confianza institucional, según Escenarios Hídricos 2030, 2019), y **sostenibilidad ambiental** (sobreexplotación de acuíferos con consecuencias irreversibles).

### **2.1.1. Complejidad operacional en sistemas urbanos con topografía irregular**

Los sistemas de distribución de agua potable en ciudades requieren gestionar redes complejas de captación, tratamiento, almacenamiento y distribución, manteniendo parámetros estrictos de calidad, presión y caudal. Esta complejidad se intensifica en ciudades con **topografía irregular** como Valparaíso, donde los 42 cerros con diferencias de cota de hasta 200 metros exigen múltiples zonas hidráulicas independientes, válvulas reguladoras de presión y sistemas de bombeo escalonados (NCh 691).

La demanda de agua urbana presenta **alta variabilidad multitemporal** que dificulta la planificación operacional:

- **Diaria:** Picos matutinos (7-9 hrs) y vespertinos (19-21 hrs), con consumos mínimos nocturnos (2-5 hrs)
- **Semanal:** Mayor consumo en fines de semana por actividades domésticas de limpieza y riego
- **Estacional:** Incrementos estimados de 20-30% en verano (IPCC, 2021)
- **Eventos especiales:** Aumentos abruptos durante feriados (Fiestas Patrias, Año Nuevo) o eventos masivos (Festival de Viña del Mar)

Adicionalmente, existe **variabilidad espacial** significativa: sectores residenciales de altos ingresos presentan consumos per cápita significativamente superiores a sectores vulnerables, mientras que zonas comerciales muestran patrones completamente diferentes con picos concentrados en horarios laborales.

Esta variabilidad obliga a ajustar continuamente la producción, el bombeo (minimizando costos energéticos) y el almacenamiento. En Chile, el promedio nacional de pérdidas físicas es 31% del agua producida (SISS, 2022), lo que hace crítica la optimización operacional en contextos de escasez.

### **2.1.2. Desafíos contemporáneos: cambio climático, urbanización e infraestructura**

Los sistemas de distribución urbana enfrentan presiones crecientes que exacerban su complejidad operacional:

**Cambio climático:** El IPCC (2021) proyecta para regiones mediterráneas como Chile central: reducción de 10-30% en precipitaciones anuales para 2050, incremento de 1,5-2,5°C en temperatura media, y menor acumulación de nieve cordillerana. Esto genera doble efecto: menor disponibilidad de fuentes y mayor demanda urbana. Escenarios Hídricos 2030 (2019) estima déficit hídrico de hasta 50% para 2030 en cuencas que abastecen Valparaíso.

**Urbanización:** Chile presenta 87,8% de población urbana (INE, 2022). El Gran Valparaíso concentra aproximadamente 930.000 habitantes con proyección de crecimiento sostenido. La proliferación de nuevas parcelaciones en zonas rurales periurbanas de la región genera demanda dispersa que compromete la sostenibilidad de acuíferos.

**Infraestructura envejecida:** Sistemas construidos en décadas de 1960-1980 presentan elevadas tasas de falla. La SISS (2022) reporta que 45% de tuberías en la región supera los 40 años de antigüedad.

**Eventos extremos:** La región ha enfrentado sequías severas (37 de 38 comunas bajo decreto de escasez hídrica 2020-2021, con embalses al 6% de capacidad), olas de calor con picos de demanda significativamente superiores al promedio, e incendios forestales que afectan infraestructura y contaminan fuentes superficiales.

### **2.1.3. De reglas estáticas a sistemas predictivos basados en machine learning**

Frente a este panorama —cambio climático, urbanización acelerada, infraestructura envejecida y eventos extremos— los métodos tradicionales de gestión hídrica basados en reglas operacionales estáticas y proyecciones históricas resultan insuficientes. Se requiere transitar hacia **gestión adaptativa e informada por datos**, donde las decisiones operacionales diarias se basen en predicciones actualizadas que integren múltiples fuentes de información.

Los modelos predictivos basados en *machine learning* permiten capturar relaciones no lineales entre variables climatológicas (temperatura, humedad, precipitación), temporales (hora, día, estacionalidad), sociales (feriados, eventos masivos, vacaciones) e históricas (tendencias recientes de consumo), generando pronósticos de demanda con horizonte de corto plazo que los operadores pueden utilizar para:

- **Planificar producción:** Programar activación de plantas anticipando demandas futuras, minimizando costos energéticos
- **Optimizar almacenamiento:** Mantener reservas estratégicas adecuadas evitando pérdidas por rebose o riesgo de desabastecimiento
- **Fortalecer resiliencia:** Generar alertas tempranas ante escenarios de estrés hídrico, activando medidas de contingencia

El **sistema predictivo desarrollado en esta investigación** implementa tres algoritmos de aprendizaje supervisado (XGBoost, Random Forest, LightGBM) que procesan 71 variables predictivas generadas mediante ingeniería de características avanzada. La **interfaz web Gradio** integra 7 módulos funcionales que permiten:

1. **Predicción multi-modelo:** Comparación visual de pronósticos de demanda neta (Q_net) a 72 horas de tres modelos simultáneamente
2. **Simulación de escenarios:** Ajuste interactivo de temperatura para evaluar sensibilidad de la demanda (coeficiente térmico: +2% por cada °C de incremento)
3. **Planificación operacional:** Cálculo automático de producción requerida (Qin) basado en demanda predicha y niveles actuales de almacenamiento
4. **Visualización histórica:** Análisis de patrones de consumo, producción y variación de volumen almacenado
5. **Análisis de características:** Identificación de variables más influyentes mediante importancia de features de cada modelo
6. **Métricas de desempeño:** Evaluación comparativa de modelos (R², MAE, RMSE) en conjuntos de entrenamiento, validación y prueba
7. **Exportación de resultados:** Generación de reportes y datasets para análisis posterior

Este enfoque integrado —fundamento teórico de ML + implementación operacional práctica— permite transitar desde gestión reactiva hacia gestión predictiva y adaptativa, fortaleciendo la capacidad de los sistemas urbanos para anticipar, absorber y adaptarse a perturbaciones climáticas y sociales cada vez más frecuentes e intensas.

La siguiente sección (2.2) contextualiza la crisis hídrica en Chile y particularmente en Valparaíso, donde la convergencia de escasez estructural, topografía compleja y alta vulnerabilidad social justifica la urgencia de herramientas predictivas avanzadas como la desarrollada en esta investigación.

---

**Fuentes citadas:**

- CIREN. (2023). *Balance hídrico de cuencas, Región de Valparaíso*. Centro de Información de Recursos Naturales.
- Escenarios Hídricos 2030. (2019). *Radiografía del agua: Brecha y riesgo hídrico en Chile*. Fundación Chile.
- INE. (2022). *Censo de Población y Vivienda 2017 - Proyecciones 2022*. Instituto Nacional de Estadísticas.
- IPCC. (2021). *Climate Change 2021: The Physical Science Basis*. Cambridge University Press.
- OCDE. (2021). *Managing water for all: An OECD perspective on pricing and financing*. OECD Publishing.
- OMS. (2022). *Guidelines for drinking-water quality*. Organización Mundial de la Salud.
- ONU. (2015). *Agenda 2030 para el Desarrollo Sostenible*. Resolución A/RES/70/1.
- ONU-Habitat. (2022). *World Cities Report 2022*. United Nations Human Settlements Programme.
- SISS. (2022). *Informe de Gestión del Sector Sanitario*. Superintendencia de Servicios Sanitarios.

---

**Fin de Parte 1 (Introducción + Sección 2.1)**

📌 **Próximos pasos:**
- Parte 2: Sección 2.2 (Contexto global y nacional de escasez hídrica)
- Parte 3: Sección 2.3 (GIRH)
- Parte 4: Secciones 2.4-2.5
- Parte 5: Secciones 2.6-2.8

**¿Revisas esta primera parte y confirmo para continuar con la Parte 2?**
