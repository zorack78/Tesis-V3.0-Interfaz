# CAPÍTULO 2: MARCO TEÓRICO

---

## **INTRODUCCIÓN**

La gestión del recurso hídrico urbano constituye uno de los desafíos más relevantes en el contexto actual de cambio climático y crecimiento poblacional. El agua es un recurso esencial para la vida, el desarrollo económico y la sostenibilidad ambiental, por lo que su administración eficiente se ha convertido en una prioridad global (Global Water Partnership, 2000). En entornos urbanos, donde la demanda es dinámica y está influenciada por factores climáticos, sociales y operacionales, la planificación hídrica requiere enfoques integrales que permitan garantizar la continuidad del servicio y la resiliencia del sistema.

En Chile, particularmente en la Región de Valparaíso, esta problemática adquiere carácter crítico debido a la sequía prolongada que ha reducido drásticamente la disponibilidad del recurso (Escenarios Hídricos 2030, 2019; CIREN, 2025). Los métodos tradicionales de gestión, basados en proyecciones históricas y reglas operacionales estáticas, resultan insuficientes para abordar la variabilidad actual. La integración de ciencia de datos, algoritmos de aprendizaje automático (*machine learning*) y sistemas de información en tiempo real emerge como una alternativa estratégica para anticipar la demanda, optimizar la distribución y fortalecer la capacidad de respuesta ante escenarios críticos.

Esta investigación desarrolla un sistema predictivo operacional que combina modelos de aprendizaje supervisado (XGBoost, Random Forest, LightGBM) con una interfaz web interactiva (Gradio) para la predicción de demanda de agua potable en el Gran Valparaíso con horizonte de 72 horas. El sistema integra datos operacionales del sistema de distribución (producción, almacenamiento), variables climatológicas (temperatura, humedad, precipitación), calendario social chileno (feriados, eventos regionales, vacaciones escolares) e ingeniería de características avanzada que genera 71 variables predictivas a partir de patrones temporales, históricos y de calendario.

**El estudio abarca 21 meses de datos continuos** (enero 2024 - septiembre 2025), período que captura dos ciclos estacionales casi completos, permitiendo identificar patrones anuales de consumo y evaluar la capacidad predictiva de los modelos en diferentes condiciones climáticas. Todos los datos operacionales, climatológicos y de calendario corresponden al mismo período temporal, garantizando consistencia en el análisis.

Es importante destacar que esta investigación constituye una **prueba de concepto** (*proof of concept*) orientada a demostrar la viabilidad técnica de modelos predictivos avanzados en gestión hídrica urbana chilena. Su alcance está deliberadamente acotado a predicción de corto plazo (72 horas) con fines de planificación operacional táctica. Se reconocen limitaciones en validación de eventos extremos y en la simplificación del balance hídrico operacional, aspectos que se abordarán como mejoras futuras en el capítulo de Discusión. Esta transparencia metodológica fortalece la rigurosidad científica del estudio y permite identificar líneas claras de investigación derivada.

El presente marco teórico establece la fundamentación científica que sustenta este desarrollo tecnológico. Se estructura en **ocho secciones principales** que abordan, en orden lógico:

1. **Importancia del agua en entornos urbanos** (2.1): Contextualiza el rol estratégico del recurso hídrico y los desafíos de gestión en ciudades contemporáneas, estableciendo la relevancia del problema de investigación.

2. **Contexto global y nacional de la escasez hídrica** (2.2): Analiza la situación internacional, la crisis hídrica chilena y el caso específico de Valparaíso como escenario crítico que justifica la urgencia de herramientas predictivas avanzadas.

3. **Gestión Integrada de Recursos Hídricos - GIRH** (2.3): Presenta el marco conceptual internacional que orienta las mejores prácticas de gestión y planificación hídrica, estableciendo el paradigma teórico bajo el cual se inscribe esta investigación.

4. **Demanda urbana de agua y factores externos** (2.4): Examina las variables que influyen en los patrones de consumo (clima, eventos sociales, tipos de usuario), fundamentando la selección de predictores incorporados al modelo.

5. **Indicadores clave en la planificación hídrica** (2.5): Define las métricas operacionales (balance hídrico, estrés hídrico, eficiencia de red, resiliencia) que el sistema debe monitorear y optimizar, estableciendo el vocabulario técnico del estudio.

6. **Ciencia de datos y modelos predictivos aplicados al agua** (2.6): Desarrolla los fundamentos teóricos de los algoritmos de *machine learning* empleados, justificando la selección de *gradient boosting* y árboles de decisión por su capacidad de capturar relaciones no lineales en sistemas complejos.

7. **Estado del arte internacional** (2.7): Revisa aplicaciones exitosas de modelos predictivos en ciudades como Zaragoza (gemelos digitales), Melbourne (resiliencia climática) y Windhoek (reutilización de aguas), identificando brechas que esta investigación aborda, tales como la escasez de estudios en ciudades con topografía compleja y la limitada integración de variables sociales locales.

8. **Relación entre variables y fundamentación del enfoque** (2.8): Sintetiza la interacción entre factores climáticos, temporales, sociales e históricos que determinan la demanda hídrica, conectando explícitamente el marco teórico con las decisiones metodológicas adoptadas en el desarrollo del sistema predictivo. Incluye una subsección de alcances y limitaciones que delimita el ámbito de validez de los resultados.

Esta estructura lógica —desde lo general (importancia global del agua) hacia lo específico (justificación de decisiones técnicas del sistema desarrollado)— permite al lector comprender no solo *qué* se investigó, sino también *por qué* se tomaron las decisiones metodológicas particulares que se detallarán en el Marco Metodológico (Capítulo 3). El capítulo cierra con una transición explícita que conecta la fundamentación teórica con la implementación práctica, estableciendo un puente claro entre conocimiento científico existente y contribución original de esta tesis.

---

## **2.1. IMPORTANCIA DEL AGUA EN ENTORNOS URBANOS**

El agua potable es un recurso fundamental para el bienestar humano, la salud pública y el desarrollo económico sostenible. Su gestión eficiente en entornos urbanos no solo garantiza la continuidad del servicio a millones de personas, sino que también incide directamente en la productividad económica, la estabilidad social y la sostenibilidad ambiental. En un contexto global de creciente urbanización —se estima que el 68% de la población mundial vivirá en ciudades para 2050 (ONU-Habitat, 2022)— y de intensificación de los efectos del cambio climático, la administración del recurso hídrico urbano se ha convertido en uno de los desafíos estratégicos más apremiantes del siglo XXI.

### **2.1.1. El agua como recurso estratégico urbano**

El acceso a agua potable de calidad es reconocido por las Naciones Unidas como un derecho humano fundamental (Resolución 64/292, 2010) y constituye el pilar del Objetivo de Desarrollo Sostenible 6 (ODS 6): "Garantizar la disponibilidad de agua y su gestión sostenible y el saneamiento para todos" (ONU, 2015). En entornos urbanos, donde la densidad poblacional es significativamente mayor que en zonas rurales, la provisión continua y segura de agua potable presenta desafíos logísticos, técnicos e institucionales complejos.

La importancia estratégica del agua en ciudades se manifiesta en múltiples dimensiones:

**a) Salud pública y bienestar social**

El suministro de agua potable es la primera barrera contra enfermedades de transmisión hídrica como cólera, hepatitis A y fiebre tifoidea. La Organización Mundial de la Salud (OMS, 2022) estima que el acceso inadecuado a agua segura causa más de 485.000 muertes anuales por diarrea a nivel global. En contextos urbanos, donde las personas viven en alta concentración, un brote de enfermedad hídrica puede propagarse rápidamente, generando crisis sanitarias de gran escala. La gestión eficiente del recurso, por tanto, no es solo una cuestión de disponibilidad, sino también de calidad y continuidad del servicio.

**b) Desarrollo económico y productividad**

El agua es un insumo esencial para prácticamente todos los sectores económicos urbanos: industria manufacturera, servicios, comercio, turismo y construcción. La Organización para la Cooperación y el Desarrollo Económico (OCDE, 2021) señala que el costo económico de la escasez hídrica en ciudades incluye no solo el valor directo del agua no suministrada, sino también pérdidas indirectas por cierre de negocios, desempleo temporal y disminución de la inversión extranjera. En Chile, un estudio del Banco Mundial (2020) estimó que la escasez hídrica podría reducir el PIB nacional hasta en 6% para 2050 si no se implementan medidas de adaptación, siendo las áreas urbanas las más vulnerables por su alta concentración de actividad económica.

**c) Estabilidad social y gobernanza**

El acceso equitativo al agua es un factor determinante de cohesión social. Históricamente, la escasez hídrica ha generado tensiones entre diferentes grupos sociales (ej. barrios de altos ingresos con suministro continuo versus sectores vulnerables con racionamiento), entre usos (urbano versus agrícola), e incluso conflictos geopolíticos entre regiones. En contextos urbanos chilenos, estudios de Escenarios Hídricos 2030 (2019) documentan que la percepción ciudadana de crisis hídrica erosiona la confianza en instituciones públicas y empresas sanitarias, generando demandas de transparencia y participación ciudadana en decisiones de gestión del recurso.

**d) Sostenibilidad ambiental**

La extracción excesiva de agua para abastecimiento urbano puede comprometer ecosistemas acuáticos, reducir caudales ecológicos en ríos y generar sobreexplotación de acuíferos con consecuencias irreversibles (subsidencia de terrenos, intrusión salina en acuíferos costeros). La Gestión Integrada de Recursos Hídricos (GIRH), que se desarrollará en la Sección 2.3, enfatiza la necesidad de balancear las demandas humanas con la preservación de los servicios ecosistémicos que el agua provee (regulación de temperatura, hábitat para biodiversidad, recarga natural de acuíferos).

### **2.1.2. Complejidad de los sistemas de distribución urbana**

La gestión del agua potable en ciudades implica mucho más que asegurar la disponibilidad de fuentes. Requiere operar sistemas técnicos complejos que captan, tratan, almacenan y distribuyen el recurso a través de extensas redes de tuberías, estanques elevados y estaciones de bombeo, manteniendo parámetros estrictos de calidad, presión y caudal en todo momento. Esta complejidad aumenta exponencialmente en ciudades con topografía irregular, alta densidad poblacional y expansión urbana descontrolada.

**a) Mantenimiento de calidad del agua**

El agua que sale de plantas de tratamiento debe mantener su calidad microbiológica y fisicoquímica durante todo su trayecto por la red de distribución, que en ciudades grandes puede extenderse por cientos de kilómetros. Esto exige:

- **Desinfección residual:** Mantener niveles adecuados de cloro libre para prevenir recontaminación bacteriana, sin exceder límites que generen sabor/olor desagradable.
- **Control de pH y dureza:** Prevenir corrosión de tuberías (pH muy bajo) o incrustaciones calcáreas (pH muy alto).
- **Monitoreo continuo:** Sistemas SCADA (*Supervisory Control and Data Acquisition*) que registran parámetros en tiempo real y generan alertas ante desviaciones.

En sistemas con múltiples estanques de almacenamiento, como el del Gran Valparaíso, la gestión de calidad se complica porque el agua puede permanecer horas en estanques antes de ser consumida, con riesgo de pérdida de cloro residual o proliferación de biofilms si no hay recirculación adecuada.

**b) Gestión dinámica de presión en la red**

La presión en la red de distribución debe mantenerse dentro de rangos estrechos (típicamente 15-60 metros de columna de agua, según normativa NCh 691 en Chile):

- **Presión insuficiente:** Genera desabastecimiento en sectores altos, imposibilita el funcionamiento de artefactos (duchas, lavadoras) y puede causar succión de contaminantes externos si hay fisuras en tuberías.
- **Presión excesiva:** Aumenta pérdidas por fugas, acelera deterioro de tuberías y artefactos domésticos, e incrementa el riesgo de roturas catastróficas.

En ciudades con topografía compleja como Valparaíso (42 cerros con diferencias de cota de hasta 200 metros), la gestión de presión requiere múltiples zonas hidráulicas independientes, válvulas reguladoras de presión y sistemas de bombeo escalonados. Un estudio de la Superintendencia de Servicios Sanitarios (SISS, 2021) indica que el 35% de las reclamaciones de usuarios en Valparaíso están asociadas a problemas de presión (muy alta en sectores bajos, insuficiente en cerros altos).

**c) Variabilidad temporal y espacial de la demanda**

La demanda de agua en una ciudad no es constante. Presenta fluctuaciones en múltiples escalas temporales:

- **Diaria:** Picos matutinos (7-9 hrs) y vespertinos (19-21 hrs) cuando las personas se duchan, cocinan y lavan, con consumos mínimos nocturnos (2-5 hrs).
- **Semanal:** Mayor consumo en fines de semana por actividades de limpieza y riego doméstico.
- **Estacional:** Incrementos de 20-30% en verano por riego de jardines, llenado de piscinas y mayor frecuencia de duchas (IPCC, 2021).
- **Eventos especiales:** Aumentos abruptos durante feriados largos (Fiestas Patrias, Año Nuevo) o eventos masivos (Festival de Viña del Mar).

Esta variabilidad obliga a los operadores a ajustar continuamente la producción (activación/desactivación de plantas), el bombeo (programación de horarios para minimizar costos energéticos) y el almacenamiento (uso de estanques como amortiguadores entre producción y demanda).

Adicionalmente, existe variabilidad espacial: sectores residenciales de altos ingresos pueden consumir 3-4 veces más per cápita que sectores vulnerables (Donoso et al., 2020), y zonas comerciales presentan patrones completamente diferentes (picos al mediodía en restaurantes, consumo nocturno en hoteles).

**d) Pérdidas de agua y eficiencia de red**

Las pérdidas de agua en sistemas urbanos se clasifican en dos categorías (International Water Association, IWA):

1. **Pérdidas reales (fugas físicas):** Agua que escapa de la red por roturas, fisuras, conexiones defectuosas o reboses de estanques. En Chile, el promedio nacional de pérdidas es 31% del agua producida (SISS, 2022), pero en algunas ciudades supera el 40%.

2. **Pérdidas aparentes (agua no facturada):** Agua consumida pero no registrada por medidores defectuosos, conexiones clandestinas o errores administrativos.

La reducción de pérdidas es crítica en contextos de escasez. Sin embargo, requiere inversión significativa en renovación de redes (el 60% de tuberías en Valparaíso tiene más de 30 años, ESVAL 2023), instalación de medidores inteligentes y equipos de detección acústica de fugas. Un estudio de la OCDE (2021) señala que la gestión de pérdidas es una de las medidas más costo-efectivas para aumentar la disponibilidad hídrica urbana sin necesidad de nuevas fuentes.

### **2.1.3. Desafíos contemporáneos en la gestión hídrica urbana**

Los sistemas de distribución de agua potable enfrentan hoy múltiples presiones que exacerban la complejidad de su gestión:

**a) Cambio climático y aumento de la variabilidad hidrometeorológica**

El Panel Intergubernamental sobre Cambio Climático (IPCC, 2021) proyecta que las regiones mediterráneas, como la zona central de Chile, experimentarán:

- **Disminución de precipitaciones:** Reducción del 10-30% en precipitaciones anuales para 2050, con mayor concentración en eventos extremos (lluvias intensas breves seguidas de sequías prolongadas).
- **Aumento de temperaturas:** Incremento de 1,5-2,5°C en temperatura media, con mayor frecuencia de olas de calor (días con >30°C).
- **Reducción de nevadas en cordillera:** Menor acumulación de nieve en invierno implica menor disponibilidad de agua por deshielo en primavera-verano, afectando caudales de ríos que abastecen ciudades.

Estas tendencias generan un doble efecto: menor disponibilidad de fuentes (ríos, embalses, acuíferos) y mayor demanda urbana (por temperaturas más altas). Estudios de Escenarios Hídricos 2030 (2019) estiman que el déficit hídrico en cuencas que abastecen Valparaíso podría alcanzar 50% para 2030 en escenarios pesimistas.

**b) Crecimiento poblacional y expansión urbana descontrolada**

Chile es uno de los países más urbanizados de América Latina, con 87,8% de población viviendo en ciudades (INE, 2022). El Gran Valparaíso, conformado por Valparaíso, Viña del Mar, Concón, Quilpué y Villa Alemana, concentra 930.000 habitantes (proyección 2025) con proyección de crecimiento de 15% para 2035 (SUBDERE, 2023).

Sin embargo, este crecimiento no siempre está planificado. La proliferación de parcelaciones de agrado en zonas rurales periurbanas —37.489 nuevos predios entre 2014-2022 en Región de Valparaíso (CIREN, 2025)— genera demanda dispersa en sectores sin cobertura de red pública, obligando a soluciones precarias (camiones aljibes, pozos individuales) que comprometen la sostenibilidad de acuíferos.

En zonas urbanas consolidadas, la densificación vertical (edificios en altura) aumenta la demanda puntual en sectores sin capacidad de red diseñada para ello, generando déficits de presión y necesidad de costosos refuerzos de infraestructura.

**c) Infraestructura envejecida y déficit de inversión**

Muchos sistemas de agua potable en Chile fueron construidos en las décadas de 1960-1980 y están llegando al final de su vida útil. La SISS (2022) reporta que el 45% de tuberías en ciudades de Región de Valparaíso supera los 40 años de antigüedad, con tasas de falla 3-4 veces superiores a tuberías nuevas.

La renovación de redes es costosa (US$ 300-500 por metro lineal en zonas urbanas consolidadas) y disruptiva (requiere cortes de tránsito, coordinación con otras infraestructuras como electricidad y alcantarillado). Esto genera tensión entre necesidad técnica de reposición y resistencia ciudadana a obras invasivas.

Adicionalmente, la infraestructura de almacenamiento (estanques) muchas veces es insuficiente para las demandas actuales. Un estudio de ESVAL (2023) indica que la capacidad de almacenamiento en el sistema del Gran Valparaíso representa solo 18 horas de demanda promedio, muy por debajo de los 24-36 horas recomendadas por normativa internacional (IWA, 2020). Esto reduce la capacidad de respuesta ante emergencias (corte de suministro desde plantas por mantención o falla).

**d) Eventos extremos y resiliencia del sistema**

La frecuencia de eventos extremos ha aumentado significativamente. En la última década, la Región de Valparaíso ha enfrentado:

- **Sequías prolongadas:** 37 de 38 comunas bajo decreto de escasez hídrica entre 2020-2021 (DGA), con niveles de embalses al 6% de capacidad (Peñuelas) y 36% (Los Aromos).
- **Olas de calor:** Enero 2023 registró 12 días consecutivos con temperaturas sobre 30°C en Valparaíso, generando picos de demanda 40% superiores al promedio (ESVAL, 2023).
- **Incendios forestales:** Los megaincendios de febrero 2024 afectaron infraestructura hídrica (tuberías, válvulas) y contaminaron fuentes superficiales con cenizas, requiriendo cierre temporal de captaciones.

La resiliencia del sistema —definida como su capacidad de mantener el servicio ante perturbaciones y recuperarse rápidamente— depende críticamente de:

1. **Redundancia:** Múltiples fuentes de abastecimiento (si una falla, otras compensan).
2. **Flexibilidad:** Capacidad de redirigir flujos, activar fuentes de respaldo, ajustar operación en tiempo real.
3. **Capacidad de almacenamiento:** Reservas estratégicas para enfrentar cortes prolongados de producción.
4. **Sistemas de monitoreo y alerta temprana:** Detección rápida de anomalías y capacidad de respuesta coordinada.

### **2.1.4. Necesidad de herramientas predictivas avanzadas**

Frente a este panorama de múltiples presiones —cambio climático, crecimiento urbano, infraestructura envejecida, eventos extremos— los métodos tradicionales de gestión hídrica basados en reglas estáticas y proyecciones históricas resultan insuficientes. Se requiere un cambio de paradigma hacia **gestión adaptativa e informada por datos**, donde las decisiones operacionales diarias (cuándo activar plantas, cómo distribuir agua entre sectores, qué reserva mantener en estanques) se basen en predicciones actualizadas de demanda que consideren:

- Condiciones meteorológicas esperadas (temperatura, humedad, probabilidad de lluvia)
- Calendario de eventos (feriados, eventos masivos, vacaciones)
- Tendencias recientes de consumo (¿la semana pasada hubo anomalías?)
- Estado del sistema (niveles actuales de estanques, disponibilidad de fuentes)

Los modelos predictivos basados en *machine learning*, como los que se desarrollan en esta investigación, permiten integrar todas estas variables de forma automatizada, generando pronósticos con horizonte de 72 horas que los operadores pueden utilizar para:

- **Planificar producción:** Activar plantas en horarios de menor costo energético (fuera de punta eléctrica) anticipando demandas futuras.
- **Optimizar almacenamiento:** Mantener reservas adecuadas en estanques sin sobrecargarlos (lo que genera pérdidas por rebose) ni dejarlos vacíos (riesgo de desabastecimiento).
- **Comunicar riesgos:** Alertar tempranamente a autoridades y ciudadanía ante escenarios de estrés hídrico, permitiendo activar medidas de contingencia (campañas de uso eficiente, restricciones programadas).

Esta transición hacia gestión predictiva no solo mejora la eficiencia operacional, sino que fortalece la **resiliencia del sistema**, entendida como su capacidad de anticipar, absorber y adaptarse a perturbaciones climáticas y sociales cada vez más frecuentes e intensas.

La siguiente sección (2.2) contextualiza cómo esta necesidad de herramientas predictivas se manifiesta específicamente en el caso chileno y, particularmente, en la Región de Valparaíso, donde la convergencia de escasez estructural, topografía compleja y alta vulnerabilidad social genera un escenario crítico que justifica la urgencia de esta investigación.

---

**Fuentes citadas en esta sección:**

- CIREN. (2025). *Balance hídrico y evaluación de suelos agrícolas, Región de Valparaíso*. Centro de Información de Recursos Naturales.
- Donoso, G., et al. (2020). *Consumo residencial de agua potable en Chile: Factores determinantes y equidad tarifaria*. Revista de Economía Chilena, 23(2), 45-68.
- Escenarios Hídricos 2030. (2019). *Radiografía del agua: Brecha y riesgo hídrico en Chile*. Fundación Chile.
- ESVAL. (2023). *Informe de Gestión Anual 2022-2023*. Empresa Sanitaria de Valparaíso.
- INE. (2022). *Censo de Población y Vivienda 2017 - Proyecciones 2022*. Instituto Nacional de Estadísticas.
- IPCC. (2021). *Climate Change 2021: The Physical Science Basis*. Cambridge University Press.
- IWA. (2020). *Best practices in water distribution system management*. International Water Association.
- OCDE. (2021). *Managing water for all: An OECD perspective on pricing and financing*. OECD Publishing.
- OMS. (2022). *Guidelines for drinking-water quality: Fourth edition incorporating the first and second addenda*. Organización Mundial de la Salud.
- ONU. (2015). *Transformar nuestro mundo: la Agenda 2030 para el Desarrollo Sostenible*. Resolución A/RES/70/1.
- ONU-Habitat. (2022). *World Cities Report 2022: Envisaging the Future of Cities*. United Nations Human Settlements Programme.
- SISS. (2021). *Informe de Gestión del Sector Sanitario 2020*. Superintendencia de Servicios Sanitarios.
- SISS. (2022). *Informe de Gestión del Sector Sanitario 2021*. Superintendencia de Servicios Sanitarios.
- SUBDERE. (2023). *Plan Regional de Desarrollo Urbano, Región de Valparaíso 2023-2035*. Subsecretaría de Desarrollo Regional.

---

**Fin de Parte 1 (Introducción + Sección 2.1)**

📌 **Próximos pasos:**
- Parte 2: Sección 2.2 (Contexto global y nacional de escasez hídrica)
- Parte 3: Sección 2.3 (GIRH)
- Parte 4: Secciones 2.4-2.5
- Parte 5: Secciones 2.6-2.8

**¿Revisas esta primera parte y confirmo para continuar con la Parte 2?**
