# 🎯 CHANGELOG - Historial de Versiones

## Versión 2.0 Final (8 de diciembre de 2025) 🔒

### 🆕 Nuevas Características
- ✅ **Histograma de MAPE** agregado como 4ta métrica en fila 1
- ✅ **3 Scatter plots individuales** (XGBoost, RandomForest, LightGBM)
- ✅ **Métricas integradas** en cada scatter plot (R², RMSE, MAE, MAPE)
- ✅ **Serie temporal expandida** a todo el ancho (4 columnas)
- ✅ **Leyenda optimizada** en formato vertical, esquina inferior derecha
- ✅ **Títulos mejorados** para cada subplot
- ✅ **Layout 3x4** (antes 2x3) para mejor aprovechamiento del espacio

### 🎨 Mejoras Visuales
- Líneas más gruesas en serie temporal (Real: 2.5px, Predicciones: 2px)
- Cajas de métricas con fondo semitransparente y bordes coloreados
- Líneas diagonales más visibles en scatter plots (1.5px)
- Leyenda con fuente más grande (13px) y título "Modelos" (14px)
- Altura total aumentada a 1150px para mejor visualización

### 📊 Distribución del Espacio
- Fila 1 (22%): 4 barras de métricas
- Fila 2 (38%): Serie temporal completa
- Fila 3 (40%): 3 scatter plots + espacio para leyenda

### 🔧 Cambios Técnicos
- Estructura de subplots: `specs` con colspan y None para optimizar espacio
- Referencias de ejes: x1-x8, y1-y8 (8 subplots + espacios vacíos)
- Nombres de leyenda más descriptivos ("Demanda Real", "Predicción XGBoost", etc.)
- Títulos de subplot con contexto ("Todo el Período" vs "Últimos 7 Días")

### 📝 Documentación
- ✅ `VERSION_V2.0_FINAL.md` - Documentación completa de V2.0
- ✅ `NO_MODIFICAR_V2.md` - Advertencia de bloqueo
- ✅ `CHANGELOG.md` - Historial de cambios (este archivo)

### 🔐 Git
- Commit: `7e35f7e` - "🔒 V2.0 FINAL - Interfaz con visualizaciones mejoradas"
- Tag: `v2.0-final`
- Branch: `prediccion-con-temperatura`

---

## Versión 1.0 Final (7 de diciembre de 2025) 🔒

### 🆕 Características Iniciales
- ✅ Sistema completo de planificación de producción (Qin)
- ✅ 3 modelos ML: LightGBM, XGBoost, RandomForest
- ✅ Dataset de 162 features (sin EMAs)
- ✅ Split 70/15/15 (Train/Val/Test)
- ✅ Visualización de 3 métricas: R², RMSE, MAE
- ✅ Serie temporal de últimos 7 días
- ✅ Layout 2x3

### 🔧 Correcciones Realizadas
- ✅ Eliminación de EMAs (mejor performance)
- ✅ Modelos reentrenados con NumPy 1.24.3
- ✅ Corrección de carga de modelos (XGBoost cargaba LightGBM)
- ✅ Estandarización de hiperparámetros RandomForest
- ✅ Corrección de texto sobre influencia de temperatura
- ✅ Fix de valores cortados en gráficos (cliponaxis=False)

### 📊 Métricas Validadas
- LightGBM: R²=0.6307, MAE=1,650 m³/h
- RandomForest: R²=0.6173, MAE=1,696 m³/h
- XGBoost: R²=0.5637, MAE=1,928 m³/h

### 📝 Documentación
- ✅ `VERSION_FINAL_NO_MODIFICAR.txt` - Documentación completa de V1.0
- ✅ `NO_MODIFICAR_INTERFAZ.md` - Advertencia de bloqueo

### 🔐 Git
- Estado: Bloqueada y respaldada
- Archivo: `interfaz_planificacion_qin_v1.py`

---

## Comparación V1.0 vs V2.0

| Característica | V1.0 | V2.0 |
|----------------|------|------|
| **Métricas mostradas** | 3 | 4 (+ MAPE) |
| **Scatter plots** | ❌ | ✅ (3) |
| **Métricas en scatter** | N/A | ✅ |
| **Serie temporal ancho** | 3 cols | 4 cols |
| **Leyenda** | Horizontal | Vertical |
| **Layout** | 2x3 | 3x4 |
| **Altura** | 900px | 1150px |
| **Líneas serie** | 1.5-2px | 2-2.5px |
| **Optimización espacio** | Básica | Avanzada |

---

## Roadmap Futuro

### Posibles Características para V3.0 (Si se requiere)
- [ ] Comparación de múltiples períodos de testing
- [ ] Análisis de errores por hora del día
- [ ] Distribución de errores (histogramas)
- [ ] Métricas por semana/mes
- [ ] Análisis de influencia de variables (SHAP values)
- [ ] Exportación de gráficos en alta resolución
- [ ] Modo oscuro para visualizaciones

### Mejoras Potenciales
- [ ] Optimización de velocidad de carga
- [ ] Cache de predicciones
- [ ] Zoom interactivo en scatter plots
- [ ] Filtros por rango de fechas
- [ ] Descarga de datos en CSV/Excel

---

## Notas Importantes

### ⚠️ Protocolo de Versiones
- V1.0: **BLOQUEADA** - No modificar
- V2.0: **BLOQUEADA** - No modificar
- V3.0+: Crear nuevas versiones solo si es necesario

### 🔒 Archivos Protegidos
- `interfaz_planificacion_qin_v1.py`
- `interfaz_planificacion_qin_v2.py`
- `models/forecasting/modelo_forecasting_lgbm.pkl`
- `models/forecasting/modelo_forecasting_xgboost.pkl`
- `data/processed/dataset_features_completo.csv`

---

**Última actualización:** 8 de diciembre de 2025  
**Mantenedor:** Equipo Tesis 3.0  
**Repositorio:** https://github.com/zorack78/Tesis-V3.0-Interfaz
