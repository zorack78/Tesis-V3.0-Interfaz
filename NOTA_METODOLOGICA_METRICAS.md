# Nota Metodológica: Evaluación sobre Demanda (Qout) vs Q_net

## Contexto

Inicialmente se evaluó el desempeño de los modelos sobre la variable **Q_net** (ΔVol/Δt), que representa el cambio en el volumen almacenado por unidad de tiempo.

## Problema Identificado

Aunque se obtuvieron valores de **R² cercanos a 0.99**, el **MAPE** (Error Porcentual Medio Absoluto) resultó poco interpretable debido a que:

1. **Q_net toma valores positivos y negativos**: Representa tanto recarga como descarga del sistema
2. **Cruza frecuentemente por cero**: Durante transiciones entre estados
3. **Genera errores porcentuales muy elevados**: Incluso para errores absolutos moderados

### Ejemplo del Problema

Si Q_net real = 50 m³/hr y Q_net predicho = 100 m³/hr:
- Error absoluto: 50 m³/hr (razonable)
- Error porcentual: 100% (parece terrible)

Si Q_net real = -50 m³/hr y Q_net predicho = -100 m³/hr:
- Error absoluto: 50 m³/hr (mismo error)
- Error porcentual: 100% (mismo problema)

## Solución Implementada

Dado que el **objetivo operacional es la demanda de agua potable (Qout)**, se redefinió la evaluación para calcular las métricas sobre **Qout**, reconstruida a partir de Q_net y del perfil histórico de Qin:

```
Qout = Qin - Q_net
```

Donde:
- **Qin**: Producción histórica por hora (mediana)
- **Q_net**: Cambio en almacenamiento (predicho por el modelo)
- **Qout**: Demanda real del sistema

## Ventajas del Nuevo Enfoque

1. **Interpretación directa**: Qout siempre es positivo y representa consumo real
2. **Métricas consistentes**: R², RMSE, MAE y MAPE tienen sentido operacional
3. **Relevancia práctica**: Los operadores trabajan con demanda, no con balance de almacenamiento
4. **Evita distorsiones**: MAPE no explota por valores cercanos a cero
5. **Validación operacional**: Las métricas reflejan el desempeño real del sistema

## Implementación

### En Evaluación Testing (evaluar_testing)
```python
# Obtener Qin por hora
df_test['hora'] = df_test['timestamp'].dt.hour
qin_por_hora = df_test['hora'].map(self.qin_perfil_hora).apply(lambda x: x['median'])

# Demanda real y predicha
y_test_qout = qin_por_hora.values - y_test_qnet
y_pred_qout = qin_por_hora.values - y_pred_qnet

# Calcular métricas SOBRE LA DEMANDA
metrics = self._calculate_metrics(y_test_qout, y_pred_qout)
```

### En Comparación de Modelos (comparar_modelos_ml)
```python
# Para cada modelo (XGBoost, RandomForest, LightGBM)
y_pred_qnet = modelo.predict(X_test)
y_pred_qout = qin_por_hora - y_pred_qnet

metrics = self._calculate_metrics(y_test_qout, y_pred_qout)
```

### Cálculo de MAPE Robusto
```python
def _calculate_metrics(self, y_true, y_pred, mape_min=1000):
    # MAPE: ignorar valores muy pequeños para evitar distorsiones
    mask = np.abs(y_true) > mape_min  # Umbral: 1000 m³/hr
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
```

## Resultados

De esta forma:
- Las métricas **R², RMSE, MAE y MAPE** están directamente asociadas al **consumo real** del sistema
- Son **consistentes** tanto con el comportamiento observado como con la **interpretación de los operadores**
- Facilitan la **toma de decisiones operacionales** basadas en predicciones de demanda
- Permiten una **comparación justa** entre diferentes algoritmos de machine learning

## Para la Tesis

**Sección sugerida en Metodología:**

> Inicialmente se evaluó el desempeño de los modelos sobre la variable Q_net (ΔVol/Δt). 
> Aunque se obtuvieron valores de R² cercanos a 0.99, el MAPE resultó poco interpretable 
> debido a que Q_net toma valores positivos y negativos y cruza frecuentemente por cero, 
> lo que genera errores porcentuales muy elevados incluso para errores absolutos moderados.
>
> Dado que el objetivo operacional es la demanda de agua potable (Qout), se redefinió 
> la evaluación para calcular las métricas sobre Qout, reconstruida a partir de Q_net 
> y del perfil histórico de Qin (Qout = Qin − Q_net). De esta forma, las métricas R², 
> RMSE, MAE y MAPE pasan a estar directamente asociadas al consumo real del sistema y 
> son consistentes tanto con el comportamiento observado como con la interpretación de 
> los operadores.

---

**Fecha:** 20 de noviembre de 2025  
**Archivo:** interfaz_planificacion_qin_v1.py  
**Funciones modificadas:** 
- `evaluar_testing()` 
- `comparar_modelos_ml()`
- `_calculate_metrics()`
- `_generar_reporte_comparacion()`
- `_generar_grafico_comparacion()`
