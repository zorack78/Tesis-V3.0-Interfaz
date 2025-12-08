# CAPÍTULO 2: MARCO TEÓRICO - PARTE 3

---

## **2.3. GESTIÓN INTEGRADA DE RECURSOS HÍDRICOS (GIRH)**

### **2.3.1. Marco conceptual internacional**

La Gestión Integrada de Recursos Hídricos (GIRH) es definida por la Global Water Partnership (GWP, 2000) como "un proceso que promueve la gestión y desarrollo coordinado del agua, la tierra y los recursos relacionados, con el fin de maximizar el bienestar social y económico resultante de manera equitativa, sin comprometer la sostenibilidad de ecosistemas vitales".

Este paradigma surge como respuesta a la gestión fragmentada que históricamente ha caracterizado al sector hídrico, donde diferentes instituciones gestionan independientemente agua potable, riego agrícola, generación hidroeléctrica, saneamiento y protección ambiental, generando ineficiencias, conflictos y sobreexplotación de fuentes compartidas.

**Los cuatro principios de Dublin (1992)** —ratificados en la Cumbre de la Tierra de Río— constituyen los pilares conceptuales de GIRH:

1. **Agua como recurso finito y vulnerable:** Esencial para vida, desarrollo y medio ambiente, requiere gestión holística que vincule desarrollo social-económico con protección de ecosistemas naturales

2. **Enfoque participativo:** Gestión debe involucrar usuarios, planificadores y autoridades en todos los niveles, con decisiones tomadas al nivel más bajo apropiado (principio de subsidiariedad)

3. **Rol central de la mujer:** Reconocimiento de su papel fundamental en provisión, gestión y protección del agua, especialmente en comunidades rurales y urbanas vulnerables

4. **Agua como bien económico:** Tiene valor económico en todos sus usos competitivos, debiendo reconocerse el derecho básico de acceso a agua potable a precio asequible

**Aplicación en sistemas urbanos:** En contextos de distribución de agua potable, GIRH implica:

- **Integración sectorial horizontal:** Coordinación entre abastecimiento urbano, saneamiento, drenaje pluvial, protección de cuencas de captación
- **Integración vertical:** Articulación entre políticas nacionales, planificación regional y operación local
- **Integración de conocimientos:** Combinación de saberes técnicos, científicos, tradicionales y experiencia operacional
- **Gestión adaptativa:** Capacidad de ajustar estrategias basándose en monitoreo continuo y aprendizaje de resultados

### **2.3.2. Desafíos de GIRH en Chile: fragmentación institucional y ausencia de gestión por cuencas**

La implementación de GIRH en Chile enfrenta obstáculos estructurales significativos:

**Fragmentación institucional:** La OCDE (2021) identifica más de 40 instituciones con competencias relacionadas a gestión hídrica en Chile, sin mecanismo efectivo de coordinación. Las principales incluyen:

- **Dirección General de Aguas (DGA):** Otorga y fiscaliza derechos de aprovechamiento, pero con limitadas facultades de gestión activa
- **Superintendencia de Servicios Sanitarios (SISS):** Regula empresas sanitarias urbanas, sin competencia sobre usos agrícolas o industriales del agua
- **Ministerio de Obras Públicas:** Infraestructura hídrica (embalses, canales), pero sin gestión integrada de cuencas
- **Municipalidades:** Responsables de emergencias hídricas locales, con recursos técnicos y financieros limitados
- **Empresas sanitarias privadas:** Operan sistemas de distribución bajo lógica de concesión, con planificación orientada a rentabilidad más que sostenibilidad de cuenca

**Ausencia de gestión por cuencas:** Chile carece de organismos de cuenca con autoridad legal y financiamiento para gestión integrada. Las decisiones sobre agua se toman sectorialmente (urbano, agrícola, minero) sin considerar balance hídrico agregado ni impactos acumulativos sobre fuentes compartidas.

**Reformas recientes (2022):** La Ley 21.435 reformó el Código de Aguas estableciendo:
- Priorización del uso humano sobre otros usos
- Mayor discrecionalidad de DGA para denegar o condicionar derechos en cuencas sobreotorgadas
- Caducidad de derechos no utilizados
- Consideración de cambio climático en otorgamiento de nuevos derechos

Sin embargo, la implementación efectiva enfrenta resistencias de sectores con derechos consolidados (agricultura de exportación, minería) y limitaciones de capacidad técnica de DGA para fiscalización efectiva en todo el territorio nacional.

### **2.3.3. Sistema predictivo ML como herramienta operacional de GIRH urbana**

El sistema predictivo desarrollado en esta investigación se alinea con principios GIRH aplicados a gestión operacional de agua potable urbana:

**Integración de información dispersa:** La interfaz Gradio consolida en plataforma única:
- Datos operacionales del sistema de distribución (producción, almacenamiento)
- Variables climatológicas que afectan oferta (precipitación) y demanda (temperatura, humedad)
- Calendario social chileno que modifica patrones de consumo (feriados, vacaciones, eventos)
- Balance hídrico agregado (Qin - Qout = ΔVolumen) que vincula producción, demanda y almacenamiento estratégico

Esta integración responde al principio GIRH de **gestión holística** que supera fragmentación tradicional donde cada tipo de información se gestiona independientemente.

**Optimización de recurso escaso bajo restricciones múltiples:** Los tres modelos ML (XGBoost, Random Forest, LightGBM) permiten:
- Anticipar demanda considerando simultaneamente restricciones de **oferta** (disponibilidad limitada de fuentes), **infraestructura** (capacidad de plantas y estanques) y **calidad de servicio** (mantener presión adecuada en 89 estanques distribuidos)
- Generar escenarios de producción requerida (Qin) minimizando costos energéticos de bombeo mientras se mantienen reservas estratégicas para contingencias

Esto operacionaliza el principio de **agua como bien económico**, maximizando eficiencia operacional del recurso limitado.

**Facilitación de decisiones participativas e informadas:** La interfaz incluye módulos que:
- Visualizan información histórica de forma accesible para actores no-técnicos (gráficos interactivos de consumo, producción, variación de volumen)
- Permiten simulación de escenarios "qué pasa si" (ej: ¿cómo cambia demanda si temperatura aumenta 3°C?), facilitando discusiones informadas sobre vulnerabilidad climática
- Exportan reportes para comunicación a autoridades y ciudadanía, promoviendo transparencia y participación

Esto se alinea con principios de **gestión participativa** y **subsidiariedad**, donde herramientas técnicas empoderan a operadores locales para tomar decisiones informadas sin depender exclusivamente de autoridades centrales.

**Gestión adaptativa basada en aprendizaje continuo:** El sistema permite:
- Comparar desempeño de tres modelos alternativos (R², MAE, RMSE) e identificar algoritmo más robusto según condiciones específicas del sistema
- Actualizar modelos con nuevos datos operacionales conforme se acumulan, mejorando predicciones progresivamente
- Identificar variables más influyentes (importancia de features) para priorizar inversiones en monitoreo (ej: si temperatura es predictor crítico, justifica densificar red de sensores climáticos)

Esto operacionaliza el principio de **gestión adaptativa** de GIRH, donde estrategias se ajustan basándose en monitoreo continuo y retroalimentación de resultados.

**Limitaciones para GIRH plena:** Es importante reconocer que este sistema aborda gestión operacional táctica de corto plazo (72 horas) en un subsistema urbano específico (Gran Valparaíso). GIRH plena requeriría:
- Integración con gestión de cuenca completa (considerando usos agrícolas aguas arriba, caudales ecológicos, recarga de acuíferos)
- Coordinación interinstitucional formal (DGA, SISS, ESVAL, municipalidades)
- Participación ciudadana estructurada en decisiones estratégicas (no solo consulta técnica)
- Horizonte temporal más largo que capture dinámicas estacionales e interanuales completas

Sin embargo, como **herramienta operacional** que implementa principios GIRH a escala de sistema urbano, el sistema predictivo ML desarrollado representa avance significativo respecto a gestión fragmentada tradicional basada en reglas estáticas y decisiones reactivas.

La siguiente sección (2.4) examina específicamente las variables que influyen en la demanda urbana de agua —climatológicas, temporales, sociales— fundamentando la selección de predictores incorporados al modelo y justificando la ingeniería de características implementada en la interfaz.

---

**Fuentes citadas en esta sección:**

- Global Water Partnership (GWP). (2000). *Integrated Water Resources Management*. TAC Background Papers No. 4. GWP Secretariat.
- International Conference on Water and the Environment (ICWE). (1992). *The Dublin Statement on Water and Sustainable Development*. Dublin, Ireland.
- OCDE. (2021). *Managing water for all: An OECD perspective on pricing and financing*. OECD Publishing.
- República de Chile. (2022). *Ley 21.435: Reforma al Código de Aguas*. Biblioteca del Congreso Nacional.

---

**Fin de Parte 3 (Sección 2.3) - ~950 palabras**

📌 **Siguiente:** Parte 4 - Secciones 2.4 y 2.5 (Factores de demanda + Indicadores operacionales)
