# Modelo Predictivo de Demanda de Agua Potable - Gran Valparaíso

## 📋 Descripción del Proyecto

Este proyecto desarrolla un modelo predictivo para la demanda de agua potable en el Gran Valparaíso, Chile, utilizando técnicas de aprendizaje automático y análisis de series temporales. El modelo considera factores temporales, eventos sociales, feriados y patrones de consumo históricos para predecir la demanda futura.

## 🎯 Objetivos

- Predecir la demanda horaria de agua potable en m³
- Identificar patrones de consumo y tendencias
- Considerar eventos especiales (feriados, festivales, elecciones)
- Optimizar la gestión de recursos hídricos

## 📊 Datos

El proyecto utiliza:
- **Datos de volumen**: Series temporales horarias de consumo de agua (m³) en UTC
- **Calendario social**: Feriados chilenos, eventos especiales, vacaciones escolares
- **Período**: Año 2024 con datos horarios

### Estructura de Datos

```
data/
├── raw/              # Datos originales sin procesar
├── processed/        # Datos procesados y limpios
└── features/         # Features engineering results
```

## 🏗️ Estructura del Proyecto

```
.
├── .github/
│   └── copilot-instructions.md    # Instrucciones para Copilot
├── data/
│   ├── raw/                        # Datos crudos
│   ├── processed/                  # Datos procesados
│   └── features/                   # Features generadas
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb       # Análisis exploratorio
│   ├── 02_feature_engineering.ipynb        # Ingeniería de features
│   ├── 03_model_training.ipynb             # Entrenamiento de modelos
│   └── 04_model_evaluation.ipynb           # Evaluación y validación
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # Procesamiento de datos
│   ├── feature_engineering.py      # Generación de features
│   ├── models.py                   # Definición de modelos
│   ├── training.py                 # Entrenamiento
│   ├── evaluation.py               # Métricas y evaluación
│   └── utils.py                    # Utilidades
├── models/
│   └── trained/                    # Modelos entrenados guardados
├── config/
│   └── config.yaml                 # Configuración del proyecto
├── requirements.txt                # Dependencias Python
└── README.md                       # Este archivo
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.9 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar o abrir el proyecto en VS Code**

2. **Crear un entorno virtual (recomendado)**

```bash
python -m venv venv
```

3. **Activar el entorno virtual**

Windows (CMD):
```cmd
venv\Scripts\activate
```

Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
```

4. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

## 📈 Uso

### 1. Análisis Exploratorio de Datos

Abre y ejecuta el notebook:
```
notebooks/01_exploratory_analysis.ipynb
```

Este notebook incluye:
- Carga y visualización de datos
- Estadísticas descriptivas
- Análisis de tendencias y estacionalidad
- Identificación de patrones

### 2. Ingeniería de Features

```
notebooks/02_feature_engineering.ipynb
```

Genera features como:
- Características temporales (hora, día, mes)
- Encoding cíclico (seno/coseno)
- Lags y ventanas móviles
- Indicadores de eventos especiales

### 3. Entrenamiento de Modelos

```
notebooks/03_model_training.ipynb
```

Implementa y entrena:
- Modelos ARIMA/SARIMA
- Facebook Prophet
- Modelos de ML (Random Forest, XGBoost, LightGBM)
- Redes neuronales

### 4. Evaluación

```
notebooks/04_model_evaluation.ipynb
```

Evalúa modelos con métricas:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

## 🔧 Configuración

Edita `config/config.yaml` para ajustar:
- Rutas de datos
- Parámetros de modelos
- Configuración de validación
- Métricas a utilizar

## 📊 Modelos Implementados

### Modelos Estadísticos
- **ARIMA/SARIMA**: Para capturar tendencias y estacionalidad
- **Prophet**: Para series temporales con efectos de calendario

### Modelos de Machine Learning
- **Random Forest**: Ensemble de árboles de decisión
- **XGBoost**: Gradient boosting optimizado
- **LightGBM**: Gradient boosting eficiente

### Features Principales
- Componentes temporales (hora, día de semana, mes)
- Encoding cíclico de variables temporales
- Indicadores de feriados y eventos
- Lags de consumo previo
- Promedios móviles
- Tendencias estacionales

## 📉 Resultados Esperados

El modelo debe poder:
- Predecir demanda horaria con precisión aceptable
- Identificar picos de demanda
- Anticipar cambios en días festivos
- Adaptarse a patrones estacionales

## 🛠️ Desarrollo

### Agregar Nuevas Features

Edita `src/feature_engineering.py` y agrega tus funciones personalizadas.

### Agregar Nuevos Modelos

Define nuevos modelos en `src/models.py` siguiendo la estructura existente.

### Ejecutar Scripts de Python

```bash
python src/training.py
python src/evaluation.py
```

## 📝 Notas Importantes

- Los datos están en UTC, considera conversión a hora local chilena (-3 horas)
- El proyecto considera eventos específicos de Chile
- Valida siempre con datos out-of-sample
- Documenta cambios en los modelos

## 🤝 Contribuciones

Para contribuir al proyecto:
1. Documenta tus cambios
2. Sigue las guías de estilo de Python (PEP 8)
3. Comenta el código en español cuando sea apropiado
4. Actualiza este README si es necesario

## 📄 Licencia

Este proyecto es para fines académicos y de investigación.

## 👥 Autor

Proyecto de Tesis - Gran Valparaíso, Chile

## 📧 Contacto

Para preguntas o sugerencias sobre el proyecto, contactar al equipo de desarrollo.

---

**Última actualización**: Noviembre 2025
