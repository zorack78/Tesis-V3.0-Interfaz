# CAPÍTULO 2: MARCO TEÓRICO - PARTE 7 (FINAL)

---

## **2.8. SÍNTESIS INTEGRADORA: ALCANCES Y LIMITACIONES DEL SISTEMA DESARROLLADO**

Esta sección final articula cómo los conceptos teóricos examinados —GIRH, crisis hídrica regional, factores de demanda, fundamentos ML y estado del arte internacional— convergen en el sistema predictivo desarrollado para el Gran Valparaíso, estableciendo explícitamente sus **alcances** (qué problemas resuelve efectivamente) y **limitaciones** (qué aspectos quedan fuera del ámbito del trabajo) para situar la contribución en contexto realista y metodológicamente transparente.

### **2.8.1. Integración conceptual: de la teoría al sistema operacional**

El marco teórico presentado en las secciones 2.1-2.7 no constituye revisión bibliográfica genérica, sino **justificación estructurada** de decisiones de diseño del sistema desarrollado. La Figura 1 ilustra esta articulación:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  MARCO CONCEPTUAL INTERNACIONAL                     │
│            GIRH (GWP 2000, Dublin 1992, OCDE 2021)                 │
│  Principios: Gestión holística, eficiencia económica, participación│
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              CONTEXTO REGIONAL ESPECÍFICO                           │
│  • Megasequía Chile central (CR2 2021): >30% déficit precipitación│
│  • Valparaíso: 42 cerros, 0-250 m.s.n.m., 89 estanques            │
│  • Marco institucional fragmentado (OCDE 2021): >40 instituciones  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│         VARIABLES QUE DETERMINAN DEMANDA URBANA                     │
│  • Climatológicas: Temperatura, humedad, precipitación (IPCC 2021)│
│  • Temporales: Hora, día, mes (patrones cíclicos)                 │
│  • Sociales: Feriados, eventos, vacaciones (calendario Chile)     │
│  • Operacionales: Volumen almacenado, producción reciente         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│      ALGORITMOS MACHINE LEARNING (Breiman 2001, Chen 2016)        │
│  • XGBoost: Gradient boosting + regularización → R²=0.9903        │
│  • Random Forest: Ensamble robusto → R²=0.9887                    │
│  • LightGBM: Eficiencia computacional → R²=0.9895                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│     INTERFAZ GRADIO: SOLUCIÓN OPERACIONAL 7 MÓDULOS INTEGRADOS     │
│                                                                     │
│  📊 Pestaña 1: Predicción 72h con intervalos de confianza         │
│  🔍 Pestaña 2: Análisis exploratorio datos históricos             │
│  🌡️ Pestaña 3: Clima tiempo real (Open-Meteo API integrada)       │
│  ⚖️ Pestaña 4: Comparación algoritmos (XGBoost/RF/LightGBM)       │
│  🎯 Pestaña 5: Simulación escenarios producción                    │
│  📈 Pestaña 6: Métricas desempeño comparativas                     │
│  💾 Pestaña 7: Exportación resultados CSV/Excel                    │
│                                                                     │
│  → HERRAMIENTA PLANIFICACIÓN TÁCTICA WEB INTERACTIVA              │
└─────────────────────────────────────────────────────────────────────┘
```

**Figura 1.** Arquitectura de la interfaz Gradio como solución operacional integradora.

**Integración conceptual aplicada a la interfaz desarrollada:**

El marco teórico presentado (secciones 2.1-2.7) justifica las decisiones de diseño de la **interfaz Gradio**, que constituye la solución práctica central de este trabajo:

1. **Módulos integrados responden a necesidad identificada:** La crisis hídrica regional (CR2, 2021) y complejidad topográfica (42 cerros) requieren herramientas anticipatorias accesibles. Gradio proporciona interfaz web sin instalación de software complejo, operable desde cualquier navegador.

2. **71 features implementan factores de demanda:** Variables climatológicas (IPCC, 2021), calendario social Chile y patrones operacionales identificados en secciones 2.4-2.5 se materializan en pipeline automático de ingeniería de características que alimenta predicciones.

3. **Tres algoritmos ML validan robustez:** XGBoost (Chen & Guestrin, 2016), Random Forest (Breiman, 2001) y LightGBM (Ke et al., 2017) operan simultáneamente en Pestaña 4, permitiendo validación cruzada y selección del modelo óptimo según condiciones.

4. **Arquitectura open-source democratiza acceso:** Contraste con casos internacionales (sección 2.7) donde software propietario (>USD $50.000) limita replicabilidad. Solución Gradio con costo licencias = USD $0 permite adopción en municipalidades recursos limitados.

### **2.8.2. Alcances del sistema: qué problemas aborda efectivamente**

El sistema desarrollado constituye **herramienta de planificación operacional táctica de corto plazo** (horizonte 72 horas) que resuelve necesidades específicas de gestión diaria en sistemas urbanos complejos:

#### **Alcance 1: Predicción de demanda neta agregada con alta precisión**

**Capacidad:** Predice volumen total de agua demandado (Q_net = consumo real + pérdidas físicas) a nivel agregado del sistema completo (89 estanques) con error promedio de ~260 m³/hr (MAE) sobre demanda promedio de ~8.000 m³/hr (MAPE = 3.25%).

**Limitación reconocida:** No desagrega predicción por sectores geográficos individuales, zonas de presión específicas o tipos de usuario (residencial/comercial/industrial). Modelar demanda espacialmente desagregada requeriría datos de telemetría por sector (no disponibles en dataset utilizado) y multiplicaría complejidad computacional.

**Justificación del enfoque agregado:** Para planificación de producción en plantas de tratamiento (decisión operacional clave), demanda agregada total es variable suficiente; distribución espacial detallada es relevante para gestión de válvulas/bombeos (no cubierto por este sistema).

#### **Alcance 2: Integración de variables multi-dominio en modelo unificado**

**Capacidad:** Captura simultáneamente efectos de:
- Variables climatológicas: Temperatura, humedad, precipitación desde Open-Meteo API (servicio verificado internacionalmente)
- Calendario social: 365+ eventos específicos Chile (Fiestas Patrias, Festival Viña, vacaciones escolares)
- Patrones temporales: Hora, día semana, mes (codificación cíclica que preserva continuidad)
- Estado operacional: Volumen almacenado, producción reciente, tendencias 24h/168h

**Evidencia empírica del sistema desarrollado:** Análisis de importancia de features (feature importance calculado por XGBoost sobre dataset de entrenamiento) revela que variable `tiene_evento_social` explica 3.2% de variabilidad de demanda (ranking #8/71), superior a humedad relativa (2.1%), validando relevancia de calendario social frecuentemente omitido en literatura internacional. ***Resultado experimental propio, no referencia externa***.

**Contribución original:** Escasos estudios documentan integración sistemática de calendario cultural específico del país en modelos operacionales; mayoría se limita a variables climatológicas + patrones temporales genéricos.

#### **Alcance 3: Interfaz Gradio como herramienta operacional multi-módulo**

**Solución desarrollada:** Interfaz web interactiva (framework Gradio) con 7 pestañas especializadas que integran predicción, análisis y simulación en flujo de trabajo unificado:

**Pestaña 1 - Predicción 72 horas:** Genera pronósticos de demanda agregada con horizonte 1-72h, visualiza curva predictiva vs datos históricos, incluye bandas de confianza (percentiles 10-90), permite ajustar fecha inicio de predicción.

**Pestaña 2 - Análisis exploratorio:** Visualiza series temporales históricas (demanda, temperatura, humedad, precipitación), identifica patrones estacionales, detecta anomalías/picos de consumo, facilita comprensión de datos antes de modelamiento.

**Pestaña 3 - Clima tiempo real:** Integra Open-Meteo API para obtener condiciones meteorológicas actuales y pronósticos 7 días (temperatura, humedad, precipitación), sincroniza automáticamente datos climáticos con predicciones de demanda, actualiza cada vez que usuario consulta.

**Pestaña 4 - Comparación modelos:** Ejecuta simultáneamente XGBoost, Random Forest y LightGBM sobre mismo conjunto de datos, visualiza predicciones de los tres algoritmos superpuestas, permite identificar consenso (cuando coinciden) o divergencia (señal de incertidumbre), proporciona validación cruzada sin requerir conocimientos técnicos avanzados.

**Pestaña 5 - Escenarios de producción:** Simula impacto de diferentes niveles de producción en plantas de tratamiento (+10%, +20%, -10%, -20%), calcula balance hídrico proyectado (entrada - demanda = variación almacenamiento), identifica riesgo de déficit o exceso, facilita planificación operacional "qué pasa si...".

**Pestaña 6 - Métricas comparativas:** Muestra tabla con R², RMSE, MAE, MAPE de cada algoritmo calculados sobre conjunto de prueba (2,284 registros = 15% dataset), ordena modelos por desempeño, proporciona transparencia metodológica permitiendo auditoría de resultados.

**Pestaña 7 - Exportación de datos:** Descarga predicciones en formato CSV o Excel para integración con sistemas externos (hojas de cálculo, bases de datos, reportes), incluye timestamp, demanda real, demanda predicha, variables climáticas, metadatos de ejecución.

**Resultado:** XGBoost V3.0 demuestra mejor desempeño consistente (R²=0.9903), pero arquitectura multi-algoritmo previene dependencia de "caja negra" única y facilita auditoría por parte de operadores sin formación especializada en ML.

#### **Alcance 4: Horizonte predictivo operacionalmente relevante (72 horas)**

**Capacidad:** Sistema predice demanda con 1-72 horas de anticipación, ventana crítica para:
- Ajustar producción en plantas de tratamiento (decisión con inercia operacional de 6-12 horas)
- Optimizar uso de almacenamiento en estanques (reservas estratégicas que amortiguan variabilidad)
- Programar bombeos aprovechando tarifas eléctricas diferenciadas por hora (ahorro energético)

**Justificación del horizonte:** 72 horas equilibran utilidad operacional con fiabilidad predictiva; horizontes >7 días requieren proyecciones climáticas de mediano plazo (menor precisión) y utilidad marginal decreciente para decisiones tácticas diarias.

**Limitación temporal:** Sistema NO cubre planificación estratégica de largo plazo (inversiones en infraestructura, ampliaciones de red, nuevas fuentes) que requieren proyecciones multi-anuales bajo escenarios de cambio climático (fuera del alcance de modelos ML de corto plazo).

#### **Alcance 5: Arquitectura replicable para contextos similares**

**Capacidad:** Stack tecnológico open-source (Python, XGBoost, Gradio, Open-Meteo API) con costo total de licencias = USD $0, replicable en ciudades medianas (100.000-1.000.000 habitantes) con recursos limitados.

**Requisitos mínimos:**
- Datos históricos de demanda (mínimo 6 meses, idealmente 12+ meses para capturar estacionalidad)
- Servidor con 16GB RAM (~USD $500-1.000 hardware básico)
- Conocimientos técnicos intermedios Python + machine learning (capacitación ~2 semanas)

**Contraste con software propietario:** Sistemas comerciales (GoAigua, WaterGEMS) cuestan >USD $50.000 anuales, inaccesibles para municipalidades países en desarrollo. Arquitectura open-source democratiza acceso sin sacrificar capacidad técnica (R² > 0.98 comparable a sistemas comerciales).

### **2.8.3. Limitaciones reconocidas: qué aspectos quedan fuera del alcance**

Transparencia metodológica requiere identificar explícitamente limitaciones del sistema para evitar expectativas no realistas y guiar mejoras futuras:

#### **Limitación 1: Prueba de concepto sin validación en eventos extremos**

**Naturaleza:** Dataset utilizado (enero 2024 - septiembre 2025, 21 meses) cubre operación bajo condiciones **normales y sequía moderada**, pero NO incluye:
- Sequías extremas prolongadas (>6 meses con embalses <5% capacidad)
- Terremotos/tsunamis que dañan infraestructura
- Contaminación masiva de fuentes (derrame químico, algas tóxicas)
- Rupturas mayores de red (pérdida >30% producción durante días)

**Implicación:** No se puede garantizar que modelo mantenga precisión bajo condiciones radicalmente diferentes a las observadas en entrenamiento. Machine learning aprende patrones históricos; eventos sin precedentes pueden generar predicciones erróneas.

**Recomendación:** Antes de operación en producción, validar sistema durante período adicional de 12 meses incluyendo estación seca crítica (verano 2025-2026) para evaluar desempeño bajo máximo estrés hídrico.

#### **Limitación 2: Balance hídrico simplificado sin modelamiento hidráulico completo**

**Naturaleza:** Sistema modela demanda agregada (Q_net) mediante ML, pero NO simula:
- Flujos hidráulicos detallados en red de tuberías (presiones, velocidades, pérdidas de carga)
- Optimización de válvulas y bombeos sector por sector
- Calidad de agua en diferentes puntos de red (cloro residual, pH, turbiedad)
- Impacto de operación en costos energéticos desagregados

**Implicación:** Sistema complementa (no reemplaza) software de modelamiento hidráulico tradicional (EPANET, WaterGEMS). Predicción de demanda agregada es INPUT necesario pero no suficiente para gestión completa de red.

**Recomendación:** Integrar predicciones ML con modelo hidráulico EPANET calibrado de la red para simular escenarios operacionales completos (predicción ML → input a EPANET → optimización de bombeos/válvulas).

#### **Limitación 3: Origen de datos sin consentimiento institucional explícito y anonimización implementada**

**Contexto ético:** Datos utilizados provienen de registros personales del autor obtenidos durante período laboral en ESVAL (empresa sanitaria), SIN autorización formal explícita de la empresa para uso académico posterior.

**Protocolo de anonimización riguroso implementado:**
- **Agregación temporal:** Datos consolidados a nivel horario para sistema completo (no se conservan registros individuales de clientes)
- **Agregación espacial:** Volúmenes totales sin desagregación por sectores geográficos específicos, zonas de presión identificables o comunas individuales
- **Eliminación de identificadores:** Ningún registro permite inferir consumo, ubicación o características de usuarios individuales o empresariales específicos
- **Solo volúmenes operacionales agregados:** Variables modeladas (Q_net, volumen almacenado total) representan sistema completo sin trazabilidad a entidades particulares
- **Uso exclusivamente académico:** Sin fines comerciales, sin distribución a terceros

**Implicación:** Datos están completamente anonimizados conforme a estándares de protección de datos (no permiten identificación directa ni indirecta de personas/empresas). Resultados son metodológicamente válidos y reproducibles, pero aplicación operacional en ESVAL requeriría autorización formal institucional.

**Transparencia académica:** Esta limitación se documenta explícitamente cumpliendo con principios de integridad científica, reconociendo que autorización ex-post sería práctica óptima para proyectos futuros.

#### **Limitación 4: Periodo de datos relativamente corto (21 meses)**

**Naturaleza:** Dataset cubre enero 2024 - septiembre 2025 (21 meses), período suficiente para capturar estacionalidad anual (verano/invierno) pero corto para:
- Detectar tendencias de largo plazo (cambios demográficos, expansión urbana)
- Capturar ciclos multi-anuales (El Niño/La Niña con periodo 3-7 años)
- Validar robustez ante cambios estructurales (nuevas fuentes, ampliaciones de red)

**Implicación:** Modelo puede requerir reentrenamiento periódico (recomendado cada 6-12 meses) incorporando datos nuevos para mantener precisión ante evolución del sistema.

**Justificación de suficiencia:** Para sistema ML de corto plazo (72h), 21 meses es adecuado siempre que cubra ciclos estacionales completos (verificado: incluye verano 2024, invierno 2024, verano 2025). Estudios similares con Random Forest/XGBoost reportan desempeño estable con 12-18 meses de datos horarios.

#### **Limitación 5: Interfaz Gradio como prototipo, no sistema empresarial**

**Naturaleza:** Gradio es framework para prototipos rápidos (ideal para demostración académica), pero carece de características de sistemas empresariales robustos:
- Autenticación multi-usuario con roles/permisos
- Auditoría completa de acciones (logs detallados)
- Integración con sistemas SCADA/telemetría existentes
- Alta disponibilidad (redundancia, failover automático)
- Escalabilidad horizontal (múltiples servidores)

**Implicación:** Sistema actual es **prueba de concepto** demostrable en entorno académico/piloto, pero despliegue en producción operacional requeriría desarrollo adicional de arquitectura empresarial.

**Recomendación:** Para operación productiva, migrar backend ML a framework empresarial (FastAPI + PostgreSQL + Docker + Kubernetes) manteniendo lógica de modelos pero agregando capas de seguridad/auditoría.

### **2.8.4. Contribución al conocimiento y líneas futuras**

A pesar de limitaciones reconocidas, el sistema desarrollado aporta:

**Contribuciones metodológicas:**
1. Demuestra viabilidad técnica de ML en sistemas con topografía extrema (42 cerros, 0-250 m.s.n.m.)
2. Valida relevancia empírica de calendario social en modelos operacionales (3.2% explicación de varianza)
3. Proporciona blueprint open-source replicable para ciudades medianas países en desarrollo

**Líneas de investigación futura derivadas:**
1. **Desagregación espacial:** Extender modelo para predecir demanda por zona de presión específica (requiere telemetría sectorial)
2. **Integración hidráulica:** Acoplar predicciones ML con optimización EPANET para gestión integral de red
3. **Proyecciones climáticas:** Incorporar escenarios RCP 4.5/8.5 para evaluar impacto cambio climático en demanda 2030-2050
4. **Validación multi-ciudad:** Replicar metodología en otros sistemas chilenos (Concepción, La Serena, Antofagasta) para evaluar transferibilidad
5. **Aprendizaje continuo:** Implementar modelos que se actualizan automáticamente con datos nuevos (online learning) sin reentrenamiento manual

---

### **2.8.5. Síntesis final: posicionamiento del trabajo**

El sistema predictivo desarrollado para el Gran Valparaíso constituye:

✅ **Herramienta funcional** para planificación táctica 72h con precisión R²=0.9903 (XGBoost validado sobre 2,284 registros)

✅ **Implementación práctica** de principios GIRH (gestión holística, adaptativa, basada en evidencia) mediante tecnología accesible

✅ **Aporte original** en integración de topografía compleja + calendario social + arquitectura open-source (brecha identificada en estado del arte)

✅ **Prueba de concepto validada** en dataset real de 21 meses con métricas comparativas rigurosas

⚠️ **Con limitaciones reconocidas** que requieren abordarse antes de despliegue operacional a gran escala (validación eventos extremos, integración hidráulica, autorización institucional)

El Marco Teórico ha cumplido su función: **justificar teórica y empíricamente** las decisiones de diseño del sistema, situándolo en paradigmas internacionales de GIRH, crisis hídrica regional, fundamentos ML y estado del arte, estableciendo con transparencia qué aporta y qué no pretende resolver.

---

**Fuentes citadas en este capítulo completo (consolidado):**

### **Organismos internacionales:**
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2016). *Time series analysis: Forecasting and control* (5th ed.). Wiley.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
- FAO. (2021). *The State of the World's Land and Water Resources for Food and Agriculture*. Food and Agriculture Organization.
- Global Water Partnership (GWP). (2000). *Integrated Water Resources Management*. TAC Background Papers No. 4.
- Global Water Partnership (GWP). (2023). *Urban water management case studies: Melbourne and Windhoek*. GWP Secretariat, Stockholm. ***VERIFICAR***
- International Conference on Water and the Environment (ICWE). (1992). *The Dublin Statement on Water and Sustainable Development*. Dublin, Ireland.
- IPCC. (2021). *Climate Change 2021: The Physical Science Basis*. Cambridge University Press.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146-3154.
- OCDE. (2021). *Managing water for all: An OECD perspective on pricing and financing*. OECD Publishing.
- OMS. (2022). *Guidelines for drinking-water quality*. Organización Mundial de la Salud.
- ONU. (2015). *Agenda 2030 para el Desarrollo Sostenible*. Resolución A/RES/70/1.
- ONU-Habitat. (2022). *World Cities Report 2022: Envisaging the Future of Cities*. United Nations Human Settlements Programme.

### **Instituciones chilenas:**
- Centro de Ciencia del Clima y la Resiliencia (CR2). (2021). *Informe a las Naciones: La megasequía 2010-2021*. Universidad de Chile.
- CIREN. (2023). *Balance hídrico de cuencas, Región de Valparaíso*. Centro de Información de Recursos Naturales.
- Escenarios Hídricos 2030. (2019). *Radiografía del agua: Brecha y riesgo hídrico en Chile*. Fundación Chile.
- INE. (2022). *Censo de Población y Vivienda 2017 - Proyecciones 2022*. Instituto Nacional de Estadísticas.
- República de Chile. (2022). *Ley 21.435: Reforma al Código de Aguas*. Biblioteca del Congreso Nacional.
- SISS. (2022). *Informe de Gestión del Sector Sanitario*. Superintendencia de Servicios Sanitarios.

### **Fuente que requiere verificación adicional:**
- Idrica. (2025). *Caso Zaragoza: Implementación GoAigua*. ***VERIFICAR*** - Año futuro, buscar reportes técnicos 2020-2024 o fuentes alternativas.

---

**TOTAL MARCO TEÓRICO CAPÍTULO 2:**
- **8 secciones completas:** Introducción + 2.1 a 2.8
- **~8.000 palabras** (extensión estándar para capítulo teórico tesis)
- **18 referencias verificadas** + 2 marcadas para verificación adicional
- **Cobertura completa:** Desde paradigma GIRH hasta limitaciones operacionales
- **Rigor académico:** Transparencia sobre datos verificables vs estimaciones razonables

---

**Fin de Parte 7 - SECCIÓN 2.8 FINAL (~1.200 palabras)**

**FIN DEL CAPÍTULO 2: MARCO TEÓRICO**
