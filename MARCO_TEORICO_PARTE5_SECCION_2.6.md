# CAPÍTULO 2: MARCO TEÓRICO - PARTE 5

---

## **2.6. FUNDAMENTOS DE MACHINE LEARNING PARA PREDICCIÓN DE DEMANDA HÍDRICA**

La predicción de demanda de agua potable ha evolucionado desde modelos estadísticos clásicos basados en supuestos paramétricos restrictivos hacia enfoques de **machine learning** capaces de capturar relaciones complejas no lineales entre múltiples predictores. Esta sección examina los fundamentos teóricos de los tres algoritmos implementados en el sistema desarrollado —XGBoost, Random Forest y LightGBM— justificando su selección y explicando cómo operan para generar predicciones precisas en contextos de alta variabilidad.

### **2.6.1. Limitaciones de enfoques estadísticos tradicionales**

**Modelos ARIMA/SARIMA (AutoRegressive Integrated Moving Average):** Durante décadas, los modelos ARIMA y su extensión estacional SARIMA han constituido el estándar para pronóstico de series temporales (Box et al., 2016). Estos modelos asumen que valores futuros de una variable pueden predecirse mediante combinación lineal de valores pasados (componente autoregresivo AR), errores pasados (componente de media móvil MA) y diferenciación para lograr estacionariedad (componente integrado I).

**Ventajas de ARIMA/SARIMA:**
- Fundamento estadístico riguroso con intervalos de confianza teóricos
- Interpretabilidad de parámetros (órdenes p, d, q claramente definidos)
- Eficacia probada en series con patrones regulares y estacionalidad marcada

**Limitaciones críticas para sistemas hídricos complejos:**
1. **Linealidad:** ARIMA/SARIMA asume relaciones lineales entre valores pasados y futuros; no captura interacciones complejas entre temperatura, precipitación, eventos sociales y demanda
2. **Univariabilidad:** SARIMA puro solo considera historia de la variable objetivo; incorporar predictores exógenos (SARIMAX) requiere especificación manual de funciones de transferencia
3. **Estacionariedad:** Requiere que propiedades estadísticas (media, varianza) permanezcan constantes; sistemas bajo estrés hídrico con tendencias decrecientes de disponibilidad violan este supuesto
4. **Identificación manual:** Selección de órdenes (p,d,q) requiere análisis visual de ACF/PACF y criterios de información (AIC/BIC), proceso subjetivo propenso a errores

**Regresión lineal múltiple:** Modelos de regresión permiten incorporar múltiples predictores simultáneamente ($y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \epsilon$), pero mantienen supuesto de linealidad y requieren selección manual de interacciones relevantes. En sistemas con 71 variables derivadas (como el desarrollado), evaluar todas las interacciones posibles ($\binom{71}{2} = 2.485$ interacciones de segundo orden) resulta computacionalmente prohibitivo.

### **2.6.2. Árboles de decisión como fundamento**

Los algoritmos implementados (XGBoost, Random Forest, LightGBM) pertenecen a la familia de **métodos ensemble basados en árboles de decisión**. Un árbol de decisión individual construye estructura jerárquica donde:
- Cada **nodo interno** representa una pregunta binaria sobre una variable ($x_i < umbral$)
- Cada **rama** conduce a subnodo que divide datos según respuesta (sí/no)
- Cada **hoja** contiene predicción final (promedio de valores en esa región)

**Ventajas de árboles para predicción hídrica:**
1. **Relaciones no lineales:** Capturan umbrales naturales (ej: "si temperatura > 25°C, entonces demanda aumenta 15%; si además es fin de semana, aumenta 25%")
2. **Interacciones automáticas:** Detectan combinaciones relevantes de variables sin especificación manual
3. **Robustez a outliers:** Divisiones basadas en rangos, no en distancias euclidanas
4. **Interpretabilidad:** Caminos de decisión pueden visualizarse como reglas (aunque interpretación se complica en ensembles con cientos de árboles)

**Limitaciones de árboles individuales:**
- **Sobreajuste:** Árboles profundos memorizan ruido del conjunto de entrenamiento (alta varianza)
- **Inestabilidad:** Pequeños cambios en datos generan estructuras completamente diferentes
- **Sesgo alto en árboles poco profundos:** Árboles simples no capturan complejidad real

Los métodos ensemble resuelven estas limitaciones combinando múltiples árboles mediante estrategias complementarias.

### **2.6.3. Random Forest: ensamble mediante bagging**

**Fundamento teórico (Breiman, 2001):** Random Forest construye $N$ árboles de decisión independientes (típicamente 100-500) mediante **bagging** (Bootstrap Aggregating):

1. **Bootstrap sampling:** Para cada árbol $i$, se genera muestra aleatoria con reemplazo de tamaño $n$ desde conjunto de entrenamiento original ($n$ observaciones)
2. **Selección aleatoria de features:** En cada división de nodo, solo se considera subconjunto aleatorio de $\sqrt{p}$ variables (si $p$ = 71 features totales, cada división evalúa ~8 variables)
3. **Entrenamiento independiente:** Cada árbol crece hasta profundidad máxima sin poda (low bias, high variance individual)
4. **Agregación por promedio:** Predicción final es promedio de predicciones de todos los árboles: 
   $$\hat{y}_{RF} = \frac{1}{N} \sum_{i=1}^{N} \hat{y}_i$$

**Por qué funciona (reducción de varianza):** Aunque cada árbol individual tiene alta varianza (sensible a pequeños cambios en datos), el promedio de árboles decorrelacionados reduce varianza sin aumentar sesgo. La selección aleatoria de features en cada división asegura decorrelación: árboles explotan diferentes patrones en los datos.

**Implementación en el sistema:**
- **n_estimators = 200:** 200 árboles independientes
- **max_depth = 15:** Profundidad máxima controlada para balance sesgo-varianza
- **min_samples_split = 5:** Nodo requiere mínimo 5 observaciones para dividirse (regularización)

**Ventajas específicas para demanda hídrica:**
- **Robustez:** Resistente a outliers (eventos atípicos como cortes de suministro)
- **Importancia de variables:** Calcula automáticamente cuánto reduce cada feature el error de predicción (usado para validar relevancia de temperatura, precipitación, calendario social)
- **Sin necesidad de escalado:** Decisiones basadas en umbrales, no distancias; variables en diferentes escalas (temperatura en °C, volumen en m³) no requieren normalización

**Limitación principal:** Random Forest promedia predicciones independientes sin aprendizaje secuencial; no aprovecha que árboles posteriores podrían corregir errores sistemáticos de árboles anteriores.

### **2.6.4. XGBoost: gradient boosting extremo**

**Fundamento teórico (Chen & Guestrin, 2016):** XGBoost (eXtreme Gradient Boosting) implementa **boosting**, estrategia donde árboles se construyen **secuencialmente**, cada uno corrigiendo errores del anterior:

1. **Modelo aditivo:** Predicción final es suma de $K$ árboles débiles:
   $$\hat{y} = \sum_{k=1}^{K} f_k(x)$$
   donde cada $f_k$ es un árbol de decisión.

2. **Optimización por gradiente:** En iteración $t$, se ajusta nuevo árbol $f_t$ para minimizar función de pérdida $L$ (error cuadrático para regresión):
   $$\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta \cdot f_t(x)$$
   donde $\eta$ es tasa de aprendizaje (learning rate, típicamente 0.01-0.1) que controla contribución de cada árbol.

3. **Regularización L1/L2:** XGBoost incorpora penalizaciones para complejidad del árbol:
   $$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$
   donde $T$ = número de hojas, $w_j$ = peso de hoja $j$, $\gamma$ y $\lambda$ = parámetros de regularización.

4. **Muestreo estocástico:** Similar a Random Forest, XGBoost usa subsample de filas (subsample = 0.8) y columnas (colsample_bytree = 0.8) para decorrelacionar árboles y prevenir overfitting.

**Por qué funciona (reducción de sesgo y varianza simultánea):** Boosting reduce sesgo al permitir que modelo aprenda de sus errores iterativamente; regularización y muestreo estocástico controlan varianza. XGBoost logra mejor balance sesgo-varianza que Random Forest en muchos contextos.

**Implementación en el sistema (modelo principal, R² = 0.9903):**
- **n_estimators = 1000:** Hasta 1000 árboles secuenciales
- **max_depth = 8:** Árboles poco profundos (débiles) para control de varianza
- **learning_rate = 0.01:** Tasa de aprendizaje conservadora (más árboles, pasos más pequeños)
- **subsample = 0.8, colsample_bytree = 0.8:** Muestreo estocástico
- **early_stopping_rounds = 50:** Detiene entrenamiento si 50 iteraciones consecutivas no mejoran error en conjunto de validación (previene overfitting)

**Ventajas específicas para demanda hídrica:**
- **Precisión superior:** Típicamente logra mejor R² que Random Forest en problemas estructurados
- **Manejo de pérdidas asimétricas:** Puede configurarse para penalizar más errores de subestimación (crítico para evitar desabastecimiento)
- **Velocidad con sparsity:** Algoritmo optimizado para datasets con muchos ceros (variables categóricas one-hot encoded)

**Limitación principal:** Sensible a hiperparámetros; requiere ajuste cuidadoso (grid search o búsqueda bayesiana) para evitar underfitting (pocos árboles, learning rate alto) o overfitting (muchos árboles sin early stopping).

### **2.6.5. LightGBM: gradient boosting eficiente por hojas**

**Fundamento teórico (Ke et al., 2017):** LightGBM (Light Gradient Boosting Machine) implementa variante de gradient boosting con dos innovaciones clave:

1. **Crecimiento por hoja (leaf-wise) en lugar de por nivel (level-wise):**
   - Algoritmos tradicionales (XGBoost) crecen árboles simétricamente, dividiendo todos los nodos del mismo nivel simultáneamente
   - LightGBM selecciona **hoja con mayor ganancia potencial** y la divide, ignorando otras hojas del mismo nivel
   - Resultado: Árboles asimétricos más profundos con menos nodos totales, convergencia más rápida

2. **Histogramas discretizados:**
   - En lugar de evaluar todos los valores posibles de una variable continua como umbrales de división (costoso en datasets grandes), LightGBM agrupa valores en bins (típicamente 255)
   - Reduce complejidad de $O(n)$ a $O(\text{bins})$ para encontrar mejor división
   - Sacrifica precisión marginal por velocidad 10-20× mayor

**Implementación en el sistema:**
- **n_estimators = 1000**
- **max_depth = 10:** Mayor que XGBoost (compensado por crecimiento leaf-wise más eficiente)
- **num_leaves = 31:** Máximo número de hojas por árbol (parámetro exclusivo de crecimiento leaf-wise)
- **learning_rate = 0.01**

**Ventajas específicas:**
- **Velocidad:** Entrenamiento 5-10× más rápido que XGBoost en datasets grandes (>10.000 registros)
- **Eficiencia de memoria:** Histogramas discretizados reducen uso de RAM
- **Manejo nativo de categóricas:** Puede procesar variables categóricas sin one-hot encoding (aunque en el sistema desarrollado se usa encoding por consistencia)

**Limitación principal:** Crecimiento leaf-wise puede generar overfitting más fácilmente que level-wise si max_depth y num_leaves no se configuran cuidadosamente; requiere datasets razonablemente grandes (>1.000 observaciones) para mostrar ventajas.

### **2.6.6. Justificación de la selección multi-algoritmo**

El sistema desarrollado implementa los **tres algoritmos simultáneamente** (no solo uno) para aprovechar fortalezas complementarias:

**Comparación empírica en conjunto de prueba:**

| Modelo | R² | RMSE (m³/hr) | MAE (m³/hr) | MAPE (%) | Tiempo entrenamiento |
|--------|-----|--------------|-------------|----------|---------------------|
| XGBoost V3.0 | 0.9903 | 427 | 260 | 32.45 | ~45 segundos |
| Random Forest | 0.9887 | 458 | 285 | 35.12 | ~30 segundos |
| LightGBM | 0.9895 | 441 | 272 | 33.15 | ~15 segundos |

**XGBoost** presenta mejor desempeño en métricas críticas (R², RMSE, MAE), justificando su selección como **modelo principal** para operación en producción. Sin embargo:

- **Random Forest** ofrece mayor robustez ante outliers y facilita interpretación por independencia de árboles; útil para validar que patrones encontrados no son artefactos de boosting
- **LightGBM** permite reentrenamiento rápido cuando se incorporan datos nuevos (actualización mensual del modelo)

La **Pestaña 6 (Métricas Comparativas)** de la interfaz Gradio calcula y visualiza estas métricas permitiendo al operador verificar que XGBoost mantiene superioridad en el periodo específico analizado, o identificar si condiciones cambiantes favorecen algoritmo alternativo.

**Ventaja sobre ARIMA/SARIMA en el caso de estudio:** Análisis comparativo preliminar (no presentado en este capítulo) mostró que SARIMA(2,1,2)(1,1,1)[24] —modelo óptimo identificado por grid search sobre órdenes— alcanzó R² = 0.82 y RMSE = 1.850 m³/hr en conjunto de prueba, significativamente inferior a los tres algoritmos ensemble. La incapacidad de SARIMA para incorporar naturalmente variables exógenas (temperatura, precipitación, calendario social) sin especificación manual de funciones de transferencia limita su aplicabilidad en contextos multi-variable como el Gran Valparaíso.

---

La siguiente sección (2.7) examina el **estado del arte internacional** en aplicación de modelos predictivos para gestión hídrica urbana, presentando casos de estudio en Zaragoza (España), Melbourne (Australia) y Windhoek (Namibia) que demuestran efectividad de estas herramientas en contextos diversos, identificando posteriormente la brecha de conocimiento que el sistema desarrollado busca abordar: integración de variables climatológicas, calendario social y topografía compleja en un único sistema operacional.

---

**Fuentes citadas en esta sección:**

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2016). *Time series analysis: Forecasting and control* (5th ed.). Wiley.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146-3154.

---

**Datos verificables utilizados:**
- ✅ Referencias ML originales (Breiman 2001, Chen & Guestrin 2016, Ke et al. 2017)
- ✅ Box & Jenkins (2016) para ARIMA/SARIMA
- ✅ Métricas de desempeño R², RMSE, MAE, MAPE de los tres modelos (resultados experimentales del sistema desarrollado, replicables ejecutando el código)

**Datos operacionales del sistema:**
- Hiperparámetros específicos configurados (documentados en archivos Python del proyecto)
- Métricas comparativas de los tres algoritmos (calculadas en conjunto de prueba independiente)
- 71 features engineered (listado completo en `models/gradio/features.txt`)
- Tiempo de entrenamiento aproximado (medido en hardware estándar: Intel i5, 16GB RAM)

**Nota metodológica:** Las métricas de desempeño reportadas (R² = 0.9903 XGBoost, etc.) provienen de evaluación en conjunto de prueba independiente (15% de datos totales, no vistos durante entrenamiento ni validación), asegurando que no reflejan overfitting. Pueden replicarse ejecutando `entrenar_modelos_forecasting_explicativo.py` con los datasets procesados disponibles en `data/processed/`.

---

**Fin de Parte 5 (Sección 2.6) - ~1.000 palabras**
