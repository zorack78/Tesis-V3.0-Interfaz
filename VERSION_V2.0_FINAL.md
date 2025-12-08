# 🔒 VERSIÓN 2.0 FINAL - BLOQUEADA
## Sistema de Planificación de Producción (Qin) - Interfaz Gradio

**Fecha de bloqueo:** 8 de diciembre de 2025  
**Archivo:** `interfaz_planificacion_qin_v2.py`  
**Estado:** ✅ PRODUCCIÓN - NO MODIFICAR

---

## 📊 RESUMEN DE LA VERSIÓN 2.0

### Mejoras sobre V1.0
La versión 2.0 incorpora visualizaciones mejoradas en la pestaña de Testing:

#### **Nuevas Características:**
1. **Histograma de MAPE** agregado a las métricas (ahora 4 métricas: R², RMSE, MAE, MAPE)
2. **Scatter plots individuales** para cada modelo ML mostrando predicción vs real (todo el período de testing)
3. **Métricas en cada scatter plot** con cajas de texto mostrando R², RMSE, MAE y MAPE
4. **Serie temporal optimizada** ocupando todo el ancho para mejor visualización
5. **Leyenda mejorada** posicionada en esquina inferior derecha con formato vertical

---

## 🎨 ESTRUCTURA DE VISUALIZACIÓN

### Layout de Gráficos (3 filas x 4 columnas)

**Fila 1: Métricas Globales (Todo el Período de Testing)**
- Columna 1: R² Score (barras)
- Columna 2: RMSE (m³/h) (barras)
- Columna 3: MAE (m³/h) (barras)
- Columna 4: MAPE (%) (barras)

**Fila 2: Serie Temporal (Últimos 7 Días)**
- Columnas 1-4: Serie temporal ocupando todo el ancho
- Muestra últimos 7 días (168 horas) del período de testing
- Líneas: Demanda Real (negro, 2.5px) + 3 predicciones (punteadas, 2px)
- Leyenda vertical en esquina inferior derecha

**Fila 3: Scatter Plots (Predicción vs Real - Todo el Período)**
- Columna 1: XGBoost V3.0 con métricas
- Columna 2: RandomForest con métricas
- Columna 3: LightGBM con métricas
- Columna 4: Vacía (espacio para leyenda)

Cada scatter plot incluye:
- Línea diagonal gris (predicción perfecta)
- Caja de métricas en esquina superior izquierda
- 2,255 puntos (todo el período de testing)

---

## 📈 MÉTRICAS DE LOS MODELOS

### Período de Testing: 26/06/2025 - 30/09/2025 (2,255 horas)

| Modelo | R² | RMSE (m³/h) | MAE (m³/h) | MAPE (%) | Ranking |
|--------|---------|-------------|------------|----------|---------|
| **LightGBM** | 0.6307 | 2,628 | 1,650 | 17.67% | 🥇 1° |
| **RandomForest** | 0.6173 | 2,676 | 1,696 | 18.35% | 🥈 2° |
| **XGBoost V3.0** | 0.5637 | 2,857 | 1,928 | 20.71% | 🥉 3° |

**Modelo por defecto:** LightGBM (mejor desempeño global)

---

## 🎨 PALETA DE COLORES

```python
colors = ['#2E86AB', '#A23B72', '#F18701']
```

- **XGBoost V3.0:** #2E86AB (Azul)
- **RandomForest:** #A23B72 (Magenta)
- **LightGBM:** #F18701 (Naranja)
- **Demanda Real:** Negro (black)
- **Línea diagonal:** Gris (gray)

---

## 📐 CONFIGURACIÓN DE LAYOUT

### Dimensiones y Espaciado
```python
height = 1150px
rows = 3
cols = 4
vertical_spacing = 0.10
horizontal_spacing = 0.08
row_heights = [0.22, 0.38, 0.40]  # 22% métricas, 38% serie, 40% scatter
```

### Especificaciones de Subplots
```python
specs = [
    [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
    [{'type': 'scatter', 'colspan': 4}, None, None, None],
    [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}, None]
]
```

### Configuración de Leyenda
```python
legend = dict(
    orientation="v",           # Vertical
    yanchor="bottom",
    y=0.02,                    # 2% desde abajo
    xanchor="right",
    x=0.98,                    # 98% desde izquierda
    bgcolor='rgba(255,255,255,0.95)',
    bordercolor='gray',
    borderwidth=2,
    font=dict(size=13),
    title=dict(text='<b>Modelos</b>', font=dict(size=14)),
    itemsizing='constant',
    tracegroupgap=8
)
```

---

## 🔧 CONFIGURACIÓN TÉCNICA

### Dataset
- **Total registros:** 15,031
- **Features:** 162 (sin EMAs)
- **Split:** 70% / 15% / 15%
- **Train:** 10,521 registros (hasta 22/03/2025)
- **Validación:** 2,255 registros
- **Test:** 2,255 registros (26/06/2025 - 30/09/2025)

### Modelos ML

#### LightGBM (Modelo por Defecto)
```python
Ubicación: models/forecasting/modelo_forecasting_lgbm.pkl
NumPy: 1.24.3 (anaconda)
Protocolo pickle: 4
R² Validación: 0.6293
R² Testing: 0.6307
MAE: 1,650 m³/h
```

#### XGBoost V3.0
```python
Ubicación: models/forecasting/modelo_forecasting_xgboost.pkl
NumPy: 1.24.3 (anaconda)
Protocolo pickle: 4
R² Validación: 0.5634
R² Testing: 0.5637
MAE: 1,928 m³/h
```

#### RandomForest
```python
Entrenamiento: On-the-fly
Hiperparámetros:
  - n_estimators=200
  - max_depth=15
  - min_samples_split=10
  - min_samples_leaf=4
  - random_state=42
  - n_jobs=-1
R² Testing: 0.6173
MAE: 1,696 m³/h
```

### Features
```
Archivo: models/forecasting/features.txt
Total: 162 features
Categorías:
  - Lags de Qin y Volumen (48 features)
  - Rolling windows (24 features)
  - Clima (temperatura, humedad relativa) (6 features)
  - Calendario (hora, día semana, mes, festivos) (15 features)
  - Features hidráulicas (Q_net, tasas de cambio) (69 features)
```

---

## 📝 FUNCIÓN PRINCIPAL MODIFICADA

### `_generar_grafico_testing_tres_modelos()`

**Líneas:** 1965-2178 (aprox.)

**Cambios respecto a V1.0:**
1. Estructura de 2x3 → 3x4 subplots
2. Agregado de barra MAPE en fila 1
3. Serie temporal expandida a 4 columnas
4. 3 scatter plots con métricas individuales
5. Leyenda reposicionada a esquina inferior derecha
6. Títulos mejorados para cada subplot
7. Líneas más gruesas (mejor visibilidad)
8. Cajas de métricas con título "Métricas (Todo el Período)"

**Referencias de ejes (xref/yref):**
- Fila 1: x1-x4, y1-y4 (barras de métricas)
- Fila 2: x5, y5 (serie temporal)
- Fila 3: x6-x8, y6-y8 (scatter plots)

---

## 🚀 CÓMO EJECUTAR

### Requisitos
```bash
# Entorno: anaconda (NumPy 1.24.3)
python --version  # Python 3.x
```

### Ejecución
```bash
cd "c:\Users\socce\Downloads\rafa\Tesis3.0-Interfaz"
C:\Users\socce\anaconda3\python.exe interfaz_planificacion_qin_v2.py
```

### Acceso
```
URL local: http://127.0.0.1:7867
```

### Uso
1. Abrir navegador en URL local
2. Ir a pestaña "Testing"
3. Hacer clic en "Evaluar 3 Modelos ML"
4. Esperar ~30 segundos (entrenamiento de RandomForest)
5. Ver gráficos comparativos con métricas

---

## 🔐 ARCHIVOS BLOQUEADOS

### NO MODIFICAR
- ✅ `interfaz_planificacion_qin_v1.py` - Versión 1.0 (bloqueada anteriormente)
- ✅ `interfaz_planificacion_qin_v2.py` - **Versión 2.0 (ACTUAL - BLOQUEADA)**
- ✅ `models/forecasting/modelo_forecasting_lgbm.pkl`
- ✅ `models/forecasting/modelo_forecasting_xgboost.pkl`
- ✅ `models/forecasting/features.txt`
- ✅ `data/processed/dataset_features_completo.csv`

### Modificables (para futuras versiones)
- `interfaz_planificacion_qin_v3.py` (si se requiere)
- Scripts de análisis exploratorio
- Notebooks de desarrollo

---

## 📊 DIFERENCIAS V1.0 vs V2.0

| Aspecto | V1.0 | V2.0 |
|---------|------|------|
| **Métricas mostradas** | 3 (R², RMSE, MAE) | 4 (+ MAPE) |
| **Scatter plots** | No | Sí (3 individuales) |
| **Métricas en scatter** | No | Sí (caja con 4 métricas) |
| **Serie temporal ancho** | 3 columnas | 4 columnas (completo) |
| **Leyenda posición** | Horizontal centro-abajo | Vertical esquina inferior derecha |
| **Layout** | 2 filas x 3 cols | 3 filas x 4 cols |
| **Altura total** | 900px | 1150px |
| **Tamaño leyenda** | 11px | 13px |
| **Líneas serie temporal** | 1.5-2px | 2-2.5px |

---

## 🎯 VALIDACIÓN FINAL

### Checklist de Verificación
- [x] Todas las 4 métricas se muestran correctamente
- [x] MAPE aparece como 4ta barra
- [x] Serie temporal ocupa todo el ancho
- [x] 3 scatter plots visibles con todos los puntos
- [x] Métricas en cajas de cada scatter plot
- [x] Leyenda en esquina inferior derecha
- [x] Colores consistentes (azul/magenta/naranja)
- [x] Títulos descriptivos en cada subplot
- [x] Sin errores de ejecución
- [x] Responsive en navegador

### Prueba Final Realizada
```
Fecha: 8 de diciembre de 2025
Usuario: socce
Comando: C:\Users\socce\anaconda3\python.exe interfaz_planificacion_qin_v2.py
Resultado: ✅ Exitoso
URL: http://127.0.0.1:7867
Navegador: Verificado visualmente
```

---

## 🔄 HISTORIAL DE VERSIONES

### V1.0 (Bloqueada)
- Fecha: 7 de diciembre de 2025
- Archivo: `interfaz_planificacion_qin_v1.py`
- Características: 3 métricas, 1 serie temporal, layout 2x3

### V2.0 (Actual - Bloqueada)
- Fecha: 8 de diciembre de 2025
- Archivo: `interfaz_planificacion_qin_v2.py`
- Características: 4 métricas, 3 scatter plots, serie temporal completa, layout 3x4

---

## ⚠️ PROTOCOLO PARA FUTURAS MODIFICACIONES

### Si se requieren cambios:
1. **NO MODIFICAR** `interfaz_planificacion_qin_v2.py`
2. Crear nueva versión: `interfaz_planificacion_qin_v3.py`
3. Documentar cambios en `VERSION_V3.0_FINAL.md`
4. Mantener V1.0 y V2.0 como respaldo
5. Actualizar este documento solo para correcciones

### Para crear V3.0:
```bash
copy interfaz_planificacion_qin_v2.py interfaz_planificacion_qin_v3.py
# Realizar modificaciones en v3.py
# Documentar en VERSION_V3.0_FINAL.md
```

---

## 📞 CONTACTO Y SOPORTE

**Proyecto:** Tesis 3.0 - Predicción de Demanda de Agua Potable  
**Ubicación:** Gran Valparaíso, Chile  
**Repositorio:** Tesis-V3.0-Interfaz  
**Branch:** prediccion-con-temperatura  

---

## 🔒 FIRMA DE BLOQUEO

```
Versión: 2.0
Estado: BLOQUEADA ✅
Fecha: 8 de diciembre de 2025
Hash del archivo: [Pendiente generar]
Última modificación exitosa: 2025-12-08 14:30 (aprox.)

ESTA VERSIÓN HA SIDO VALIDADA Y NO DEBE SER MODIFICADA.
CUALQUIER CAMBIO FUTURO DEBE REALIZARSE EN UNA NUEVA VERSIÓN (V3.0+).
```

---

**FIN DEL DOCUMENTO**
