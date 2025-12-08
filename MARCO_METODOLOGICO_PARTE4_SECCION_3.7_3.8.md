# CAPÍTULO 3: MARCO METODOLÓGICO - PARTE 4

**Secciones 3.7 + 3.8 (~800 palabras)**

---

## **3.7. CONSIDERACIONES ÉTICAS Y LIMITACIONES METODOLÓGICAS**

### **3.7.1. Transparencia sobre origen de datos operacionales**

**Protocolo de anonimización aplicado:**

Dataset operacional (volúmenes de demanda y producción) proviene de registros de sistema de distribución Gran Valparaíso durante experiencia profesional del autor. Datos anonimizados mediante 4 niveles de agregación (ver sección 3.3.1):

1. **Agregación espacial completa:** Solo totales sistema, sin desagregación por sectores, comunas o zonas de presión
2. **Eliminación de identificadores:** Sin nombres de instalaciones, coordenadas GPS o códigos operacionales
3. **Agregación temporal:** Datos horarios sin trazabilidad a consumo de usuarios individuales
4. **Sin información personal:** Sin datos de facturación, direcciones, contratos o información que permita identificación directa/indirecta de personas naturales o jurídicas

**Conformidad con estándares de protección de datos:**

Anonimización cumple principios de Ley 19.628 sobre Protección de la Vida Privada (Chile), que establece que datos agregados operacionales sin identificadores personales no constituyen "datos personales" sujetos a consentimiento individual. Análogo a datos meteorológicos o hidrológicos de dominio operacional.

**Limitación reconocida:**

Aplicación operacional del sistema desarrollado en ESVAL requeriría autorización institucional formal y acuerdo de transferencia tecnológica. Este trabajo es prueba de concepto académica que demuestra viabilidad científica, no sistema en producción. Resultados son válidos y reproducibles independientemente de esta limitación administrativa.

### **3.7.2. Limitaciones del período de datos**

**Extensión temporal:** 21 meses (enero 2024 - septiembre 2025, 15.223 registros válidos)

**Suficiencia para el alcance:**

- ✅ Captura 2 ciclos completos verano-invierno (estacionalidad anual validada)
- ✅ Incluye 89 semanas (patrones laborales vs fin de semana)
- ✅ Contiene eventos sociales recurrentes (Año Nuevo 2024-2025, Fiestas Patrias 2024, Festival Viña 2024-2025)
- ✅ Rango térmico representativo región (8°C - 32°C)

**Limitaciones reconocidas:**

1. **Eventos extremos:** Período no incluye megasequías (última 2010-2021), terremotos de gran magnitud (último >8.0 fue 2010), o crisis hídricas severas con racionamiento. Modelo no validado para estas condiciones.

2. **Tendencias de largo plazo:** 21 meses insuficiente para capturar cambios demográficos (crecimiento poblacional), modificaciones infraestructura (expansión red de distribución), o tendencias climáticas seculares (calentamiento gradual >5 años).

3. **Generalización geográfica:** Modelo entrenado específicamente para Gran Valparaíso (clima mediterráneo costero, ~1M habitantes, turismo estacional). Aplicación directa a otras ciudades chilenas requiere reentrenamiento con datos locales.

### **3.7.3. Alcance de la solución desarrollada**

**Clasificación:** Prueba de concepto académica funcional, no sistema empresarial en producción.

**Diferencias con sistema empresarial:**

| Aspecto | Prueba de concepto (este trabajo) | Sistema empresarial producción |
|---------|-----------------------------------|--------------------------------|
| Usuarios | Investigador + evaluadores académicos | Operadores ESVAL (~20-50 usuarios) |
| Disponibilidad | Ejecución local bajo demanda | 24/7 con redundancia |
| Integración datos | Archivos CSV estáticos | API tiempo real SCADA + bases datos corporativas |
| Auditoría | Logs básicos Python | Trazabilidad completa ISO 27001, respaldos automáticos |
| Interfaz | Gradio local (puerto 7860) | Aplicación web empresarial con autenticación SSO |
| Certificación | N/A | Validación normativa NCh 409 (agua potable) |

**Implicación:** Sistema demuestra viabilidad técnica y científica. Implementación operacional requiere: (1) ingeniería de software adicional (arquitectura microservicios, CI/CD), (2) validación regulatoria con Superintendencia de Servicios Sanitarios (SISS), (3) capacitación operadores, (4) plan de contingencia ante fallas.

### **3.7.4. Supuestos metodológicos**

**Supuesto 1: Estacionariedad débil de patrones de consumo**

Modelo asume que patrones horarios/semanales observados en 2024-2025 se mantienen en horizonte predictivo 72h. Válido para operación táctica corto plazo, limitado para planificación estratégica >6 meses.

**Supuesto 2: Integridad de datos climáticos**

Estación Rodelillo (Code 330007, DMC) representa adecuadamente clima del Gran Valparaíso. Validación: correlación r>0.90 entre Rodelillo y Open-Meteo para mismo período (ver sección 3.3.2). Microclimas específicos (sectores costeros vs cerros) no capturados.

**Supuesto 3: Eventos calendario explícitos**

Calendario social incluye eventos planificables (feriados legales, vacaciones escolares, Festival Viña). Eventos imprevistos (cortes de energía masivos, alertas sanitarias, manifestaciones sociales) no modelados. Sistema emite alertas de incertidumbre cuando predicciones divergen entre modelos (Tab 4, Comparación de Modelos).

---

## **3.8. RECURSOS TECNOLÓGICOS Y REPRODUCIBILIDAD**

### **3.8.1. Hardware utilizado**

**Especificaciones equipo de desarrollo:**

- **Procesador:** Intel Core i7-11800H @ 2.30 GHz (8 núcleos, 16 hilos)
- **Memoria RAM:** 16 GB DDR4
- **Almacenamiento:** SSD NVMe 512 GB
- **Sistema operativo:** Windows 11 Pro 64-bit

**Tiempos de ejecución:**

- Preprocesamiento completo (3.3-3.4): ~3 minutos
- Entrenamiento XGBoost (10.655 registros): ~45 segundos
- Entrenamiento Random Forest: ~2 minutos
- Entrenamiento LightGBM: ~30 segundos
- Generación predicción 72h en interfaz Gradio: <2 segundos

**Nota:** Hardware no especializado (sin GPU CUDA). Tiempos permiten reentrenamiento diario si se integra con pipeline automático.

### **3.8.2. Software y dependencias**

**Entorno de desarrollo:**

- **Python:** 3.10.11
- **Gestor de paquetes:** pip 23.3.1
- **Entorno virtual:** venv (aislamiento de dependencias)

**Bibliotecas principales con versiones exactas:**

```txt
pandas==2.1.4          # Manipulación de datos
numpy==1.26.2          # Operaciones numéricas
scikit-learn==1.3.2    # Random Forest, métricas, split temporal
xgboost==2.0.3         # Gradient Boosting optimizado
lightgbm==4.1.0        # Gradient Boosting alternativo
gradio==4.8.0          # Interfaz web interactiva
plotly==5.18.0         # Gráficos interactivos
pyyaml==6.0.1          # Configuración config.yaml
requests==2.31.0       # Consultas Open-Meteo API
```

**Verificación de versiones:** Todas disponibles en repositorio PyPI (https://pypi.org/), instalables mediante `pip install -r requirements.txt`.

**IDE utilizado:** Visual Studio Code 1.85.1 con extensiones Python, Jupyter, Pylance.

### **3.8.3. Estructura del proyecto y reproducibilidad**

**Organización del repositorio:**

```
Tesis3.0-Interfaz/
├── data/
│   ├── raw/                    # Datos originales sin modificar
│   │   ├── BD_Qin_m3_UTC.csv
│   │   ├── BD_VolTotal_X_Hr_m3_UTC.csv
│   │   ├── BD_Clima2024a202509_UTC.csv
│   │   └── calendar_social_ES_COMPLETO_20240101_20250930.csv
│   └── processed/              # Datos procesados (output sección 3.3)
│       └── data_processed_complete.csv
├── models/
│   └── gradio/                 # Modelos entrenados (pickles)
│       ├── xgboost_model.pkl
│       ├── rf_model.pkl
│       └── lgb_model.pkl
├── src/
│   ├── data_processing.py      # Pipeline sección 3.3
│   ├── feature_engineering.py  # Generación 71 features (3.4)
│   ├── models.py               # Entrenamiento RF/XGB/LGB (3.5)
│   └── utils.py                # Funciones auxiliares
├── interfaz_gradio_v3.py       # Interfaz 7 módulos (sección 3.6)
├── config/
│   └── config.yaml             # Parámetros configurables
├── requirements.txt            # Dependencias exactas
└── README.md                   # Instrucciones de reproducción
```

**Pasos de reproducción (documentados en README.md):**

1. Clonar repositorio (disponible bajo solicitud al autor, datos anonimizados incluidos)
2. Crear entorno virtual: `python -m venv venv`
3. Activar entorno: `venv\Scripts\activate` (Windows) o `source venv/bin/activate` (Linux/Mac)
4. Instalar dependencias: `pip install -r requirements.txt`
5. Ejecutar pipeline completo: `python run_pipeline.py` (ejecuta secciones 3.3 → 3.4 → 3.5 secuencialmente)
6. Lanzar interfaz: `python interfaz_gradio_v3.py` (abre navegador en http://localhost:7860)

**Disponibilidad futura:** Código fuente y datos anonimizados serán publicados en repositorio GitHub público post-defensa de tesis, con licencia MIT (permite uso académico y comercial con atribución). DOI Zenodo planificado para citación formal.

### **3.8.4. Control de calidad del código**

**Prácticas aplicadas:**

- **Modularización:** Separación clara entre procesamiento datos (3.3), features (3.4), modelamiento (3.5) e interfaz (3.6)
- **Configuración externa:** Hiperparámetros en `config.yaml`, no hardcoded en scripts
- **Logging:** Registro de métricas de entrenamiento en consola y archivos log
- **Validación de entrada:** Verificación de integridad de archivos CSV antes de procesamiento (tipos de datos, rangos físicos, timestamps válidos)

**Limitación:** Código no incluye testing unitario exhaustivo (pytest), integración continua (CI/CD), o documentación tipo Sphinx. Suficiente para reproducibilidad académica, requiere refactorización para producción.

---

**Referencias citadas en esta sección:**

- Ley 19.628. (1999). *Sobre protección de la vida privada*. Biblioteca del Congreso Nacional de Chile. https://www.bcn.cl/leychile/navegar?idNorma=141599
- Python Software Foundation. (2023). *Python 3.10 Documentation*. https://docs.python.org/3.10/
- Gradio. (2024). *Gradio Documentation - Version 4.8*. https://www.gradio.app/docs/
- PyPI - Python Package Index. (2024). *Official repository for open source Python packages*. https://pypi.org/

---

**Fin de Parte 4 - SECCIONES 3.7 + 3.8 (~800 palabras)**

---

## **RESUMEN CAPÍTULO 3 COMPLETO**

**Total Marco Metodológico:** ~6.500 palabras, 8 secciones

**Estructura completada:**

- **3.1:** Enfoque metodológico (cuantitativo aplicado data-driven)
- **3.2:** Arquitectura DOD (Raw→Curated→Feature Store, UTC ISO-8601)
- **3.3:** Fuentes de datos (ESVAL anonimizado, Rodelillo Code 330007, Open-Meteo, calendario social)
- **3.4:** 71 features (lags, rolling, cíclicas, climáticas, sociales, operacionales)
- **3.5:** Modelamiento predictivo (XGBoost R²=0.9903, RF, LightGBM)
- **3.6:** Interfaz Gradio 7 módulos (SOLUCIÓN PRINCIPAL)
- **3.7:** Ética y limitaciones (anonimización Ley 19.628, 21 meses suficientes, prueba de concepto)
- **3.8:** Recursos tecnológicos (Python 3.10, scikit-learn 1.3.2, xgboost 2.0.3, hardware i7 16GB RAM)

**Referencias totales verificables:** 15+ fuentes (Ley 19.628, ISO 8601, RFC 3339, Wickham 2014, Breiman 2001, Chen & Guestrin 2016, Ke et al. 2017, Gradio Docs, PyPI, DMC-SACLIM)

**Próximo capítulo pendiente:**
- **CAPÍTULO 4:** Análisis de Información (resultados experimentales, gráficos de desempeño, análisis de errores)
