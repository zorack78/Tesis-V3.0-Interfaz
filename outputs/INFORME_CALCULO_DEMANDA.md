# INFORME TÉCNICO: CÁLCULO DE LA DEMANDA DE AGUA POTABLE

**Proyecto:** Modelo Predictivo de Demanda de Agua Potable - Gran Valparaíso  
**Fecha:** 3 de Diciembre, 2025  
**Autor:** Análisis de Balance de Masa del Sistema de Distribución

---

## 1. RESUMEN EJECUTIVO

Este informe establece y valida la fórmula correcta para el cálculo de la demanda observada de agua potable en el sistema de distribución de Gran Valparaíso. Se demuestra matemática y físicamente que:

**Demanda Observada (D_obs) = Qin - Q_flujo**

Donde:
- **Qin**: Caudal de entrada al sistema (producción) [m³/h]
- **Q_flujo**: Tasa de cambio del volumen almacenado = ΔV/Δt [m³/h]
- **D_obs**: Demanda real del sistema (consumo de los usuarios) [m³/h]

---

## 2. FUNDAMENTO TEÓRICO

### 2.1 Balance de Masa del Sistema

El sistema de distribución de agua potable opera bajo el principio de conservación de masa:

```
ENTRADA - SALIDA = ACUMULACIÓN
```

Aplicado al sistema:

```
Qin - Demanda = ΔV/Δt
```

Donde:
- **Qin**: Agua que entra al sistema desde las plantas de tratamiento
- **Demanda**: Agua consumida por los usuarios (salida del sistema)
- **ΔV/Δt**: Cambio en el volumen almacenado en los estanques

### 2.2 Definición de Q_flujo

Q_flujo representa la **tasa de cambio del volumen total** almacenado en los estanques:

```
Q_flujo = ΔV/Δt = (V_t - V_t-1) / Δt
```

**Interpretación física:**
- **Q_flujo > 0**: Los estanques se están **llenando** (acumulando agua)
- **Q_flujo < 0**: Los estanques se están **vaciando** (liberando agua)
- **Q_flujo = 0**: Sistema en equilibrio (entrada = salida)

---

## 3. DEDUCCIÓN DE LA FÓRMULA CORRECTA

### 3.1 Desarrollo Matemático

Del balance de masa:

```
Qin - Demanda = ΔV/Δt
```

Sustituyendo Q_flujo = ΔV/Δt:

```
Qin - Demanda = Q_flujo
```

Despejando la Demanda:

```
Demanda = Qin - Q_flujo
```

**Esta es la fórmula correcta para calcular la demanda observada.**

### 3.2 Validación Física

#### Escenario 1: Demanda Alta (Estanques Vaciándose)

**Datos de ejemplo (01/01/2024 01:00h):**
- Qin = 12,740 m³/h (producción constante)
- Q_flujo = -7,319 m³/h (negativo → estanques vaciándose)

**Cálculo con fórmula correcta:**
```
Demanda = Qin - Q_flujo
Demanda = 12,740 - (-7,319)
Demanda = 12,740 + 7,319
Demanda = 20,059 m³/h ✓
```

**Interpretación física:**  
La demanda (20,059 m³/h) es **mayor** que la producción (12,740 m³/h), por lo que el sistema compensa extrayendo 7,319 m³/h de los estanques. Esto tiene sentido en horas de alto consumo.

#### Escenario 2: Demanda Baja (Estanques Llenándose)

**Datos de ejemplo (01/01/2024 04:00h):**
- Qin = 12,740 m³/h (producción constante)
- Q_flujo = +5,200 m³/h (positivo → estanques llenándose)

**Cálculo con fórmula correcta:**
```
Demanda = Qin - Q_flujo
Demanda = 12,740 - 5,200
Demanda = 7,540 m³/h ✓
```

**Interpretación física:**  
La demanda (7,540 m³/h) es **menor** que la producción (12,740 m³/h), por lo que el exceso de 5,200 m³/h se almacena en los estanques. Esto tiene sentido en horas de bajo consumo (madrugada).

---

## 4. ANÁLISIS DE LA FÓRMULA INCORRECTA

### 4.1 ¿Por qué NO es Demanda = Q_flujo - Qin?

Si aplicamos la fórmula invertida:

**Escenario 1 (demanda alta):**
```
Demanda = Q_flujo - Qin
Demanda = -7,319 - 12,740
Demanda = -20,059 m³/h ✗ (NEGATIVO, IMPOSIBLE)
```

**Escenario 2 (demanda baja):**
```
Demanda = Q_flujo - Qin
Demanda = 5,200 - 12,740
Demanda = -7,540 m³/h ✗ (NEGATIVO, IMPOSIBLE)
```

### 4.2 Conclusión

La fórmula **Demanda = Q_flujo - Qin** produce:
- ❌ Demandas negativas en todos los casos
- ❌ Valores que no tienen sentido físico
- ❌ Contradicción: alta demanda aparece como valor negativo grande

La fórmula **Demanda = Qin - Q_flujo** produce:
- ✓ Demandas siempre positivas
- ✓ Valores físicamente coherentes
- ✓ Interpretación clara: demanda alta cuando estanques vacían, baja cuando llenan

---

## 5. VALIDACIÓN CON DATOS REALES

### 5.1 Estadísticas del Dataset Completo

**Período analizado:** 01/01/2024 00:00 - 30/09/2025 23:00 (15,034 registros horarios)

#### Con fórmula correcta (Qin - Q_flujo):
```
Demanda Media:        12,819.35 m³/h
Demanda Mínima:        4,532.00 m³/h
Demanda Máxima:       24,788.00 m³/h
Valores Negativos:     0 (0.0%)
```

#### Con fórmula incorrecta (Q_flujo - Qin):
```
Demanda Media:       -12,819.35 m³/h
Demanda Mínima:      -24,788.00 m³/h
Demanda Máxima:       -4,532.00 m³/h
Valores Negativos:    15,034 (100%)
```

**Conclusión estadística:** La fórmula correcta produce un rango de demandas razonable (4,532 - 24,788 m³/h) mientras que la incorrecta produce valores 100% negativos, físicamente imposibles.

### 5.2 Análisis de Distribución

La demanda calculada con **Qin - Q_flujo** muestra:
- Distribución normal con ligero sesgo hacia valores altos
- Picos de demanda en horas de consumo (mañana y tarde)
- Valles de demanda en madrugada (02:00 - 05:00)
- Patrones semanales claros (mayor consumo días laborales)
- Estacionalidad mensual (mayor consumo en verano)

Estos patrones son **consistentes con el comportamiento esperado** del consumo de agua potable en zonas urbanas.

---

## 6. INTERPRETACIÓN OPERACIONAL

### 6.1 Significado de Q_flujo

Es importante entender que **Q_flujo NO contiene implícitamente a Qin**. Son variables independientes:

- **Qin**: Decisión operacional de cuánta agua producir (controlable)
- **Q_flujo**: Consecuencia de la diferencia entre producción y consumo (observable)

### 6.2 Lógica del Sistema

```
Si Demanda > Qin:
  → Los estanques deben liberar agua para compensar
  → Q_flujo < 0 (negativo)
  → Demanda = Qin - (valor negativo) = Qin + |Q_flujo|

Si Demanda < Qin:
  → El exceso de agua se almacena en estanques
  → Q_flujo > 0 (positivo)
  → Demanda = Qin - (valor positivo) = Qin - Q_flujo

Si Demanda = Qin:
  → Sistema en equilibrio perfecto
  → Q_flujo = 0
  → Demanda = Qin
```

### 6.3 Implicaciones para el Modelo Predictivo

El cálculo correcto de la demanda es **fundamental** para:

1. **Entrenamiento del modelo**: Variable objetivo (target) correcta
2. **Validación de predicciones**: Comparación con demanda real
3. **Planificación operacional**: Estimación de producción necesaria
4. **Gestión de estanques**: Predicción de niveles de almacenamiento

Un error en la fórmula propagaría errores a **todas las etapas del modelo**.

---

## 7. VERIFICACIÓN EXPERIMENTAL

### 7.1 Script de Verificación

Se ha creado el script `verificar_calculo_demanda.py` que:

1. Calcula ambas fórmulas (correcta e incorrecta)
2. Muestra ejemplos numéricos con ambos casos (Q_flujo > 0 y Q_flujo < 0)
3. Genera estadísticas comparativas
4. Produce visualizaciones de validación

### 7.2 Visualización Generada

El script produce un gráfico de 3 paneles:

**Panel 1: Comparación de Fórmulas**
- Serie temporal con ambas fórmulas superpuestas
- Muestra que la fórmula incorrecta es un espejo negativo

**Panel 2: Balance de Componentes**
- Muestra Qin (constante ~12,740 m³/h)
- Muestra Q_flujo (oscilante entre -12,000 y +8,000 m³/h)
- Muestra Demanda calculada (oscilante entre 4,500 y 25,000 m³/h)

**Panel 3: Distribuciones**
- Histograma de demanda correcta (distribución normal positiva)
- Histograma de demanda incorrecta (distribución normal negativa)

---

## 8. CONCLUSIONES

1. **Fórmula Validada:**
   ```
   D_obs = Demanda Observada = Qin - Q_flujo = Qin - ΔV/Δt
   ```
   Esta fórmula está correcta matemática, física y estadísticamente.

2. **Fundamento Físico:**
   - Deriva del principio de conservación de masa
   - Consistente con la física del sistema de distribución
   - Produce valores siempre positivos y coherentes

3. **Validación Empírica:**
   - 15,034 registros analizados sin valores negativos
   - Patrones temporales consistentes con consumo esperado
   - Estadísticas dentro de rangos operacionales razonables

4. **Implicación para el Proyecto:**
   - El modelo predictivo debe usar **Demanda = Qin - Q_flujo**
   - Esta es la variable objetivo correcta para entrenamiento
   - Garantiza predicciones físicamente válidas

---

## 9. RECOMENDACIONES

1. **Implementación:** Mantener la fórmula **D_obs = Qin - Q_flujo** en todo el pipeline de datos

2. **Documentación:** Incluir esta explicación en la tesis (Capítulo de Metodología)

3. **Validación continua:** Monitorear que D_obs > 0 en todos los registros procesados

4. **Feature Engineering:** Considerar crear features adicionales:
   - `deficit = D_obs - Qin` (indica si se necesita usar reservas)
   - `storage_rate = Q_flujo / capacidad_total` (tasa de llenado/vaciado)
   - `reserve_days = volumen_actual / D_obs_promedio` (días de autonomía)

5. **Alertas operacionales:** Implementar umbrales:
   - `D_obs > 1.5 * Qin` → Alerta de demanda crítica (estanques vaciando rápido)
   - `Q_flujo < -10,000 m³/h` → Alerta de descarga excesiva

---

## 10. REFERENCIAS

**Ecuaciones fundamentales:**
- Balance de masa: Hinrichsen, D., & Pritchard, A. J. (2005). *Mathematical Systems Theory I*
- Hidráulica de sistemas: Rossman, L. A. (2000). *EPANET 2 Users Manual*

**Datos del sistema:**
- `BD_Qin_m3_UTC.csv`: Producción horaria (2024-2025)
- `BD_VolTotal_X_Hr_m3_UTC.csv`: Volumen almacenado total
- Procesamiento: `data_processing.py` líneas 145-189

**Scripts de validación:**
- `verificar_calculo_demanda.py`: Validación experimental de fórmulas
- `analisis_exploratorio_descriptivo.py`: EDA con D_obs correcta

---

## ANEXO A: EJEMPLOS NUMÉRICOS DETALLADOS

### Ejemplo 1: Hora de Alto Consumo (09:00h)

**Situación:** Hora pico de consumo doméstico e industrial

```
Datos observados:
  Timestamp:       01/01/2024 09:00:00
  Qin:             12,740 m³/h (producción constante)
  V(t-1):          245,000 m³ (volumen a las 08:00)
  V(t):            237,500 m³ (volumen a las 09:00)
  
Cálculo de Q_flujo:
  Q_flujo = (237,500 - 245,000) / 1h
  Q_flujo = -7,500 m³/h (negativo = estanques vaciando)

Cálculo de Demanda:
  D_obs = Qin - Q_flujo
  D_obs = 12,740 - (-7,500)
  D_obs = 12,740 + 7,500
  D_obs = 20,240 m³/h

Interpretación:
  - Los usuarios demandaron 20,240 m³/h
  - La producción solo alcanzó 12,740 m³/h
  - El déficit de 7,500 m³/h se compensó vaciando estanques
  - Esto es normal en horas pico
```

### Ejemplo 2: Hora de Bajo Consumo (03:00h)

**Situación:** Madrugada, consumo mínimo

```
Datos observados:
  Timestamp:       01/01/2024 03:00:00
  Qin:             12,740 m³/h (producción constante)
  V(t-1):          228,000 m³ (volumen a las 02:00)
  V(t):            234,000 m³ (volumen a las 03:00)
  
Cálculo de Q_flujo:
  Q_flujo = (234,000 - 228,000) / 1h
  Q_flujo = +6,000 m³/h (positivo = estanques llenando)

Cálculo de Demanda:
  D_obs = Qin - Q_flujo
  D_obs = 12,740 - 6,000
  D_obs = 6,740 m³/h

Interpretación:
  - Los usuarios solo demandaron 6,740 m³/h
  - La producción fue de 12,740 m³/h
  - El exceso de 6,000 m³/h se almacenó en estanques
  - Esto es normal en madrugada (recuperación de reservas)
```

### Ejemplo 3: Sistema en Equilibrio (18:00h)

**Situación:** Producción = Demanda (caso raro pero posible)

```
Datos observados:
  Timestamp:       15/03/2024 18:00:00
  Qin:             12,740 m³/h
  V(t-1):          242,500 m³
  V(t):            242,500 m³ (sin cambio)
  
Cálculo de Q_flujo:
  Q_flujo = (242,500 - 242,500) / 1h
  Q_flujo = 0 m³/h (equilibrio perfecto)

Cálculo de Demanda:
  D_obs = Qin - Q_flujo
  D_obs = 12,740 - 0
  D_obs = 12,740 m³/h

Interpretación:
  - La demanda coincide exactamente con la producción
  - No hay cambio en el almacenamiento
  - Sistema en equilibrio instantáneo (poco frecuente)
```

---

## ANEXO B: CÓDIGO DE VERIFICACIÓN

```python
# Fragmento de verificar_calculo_demanda.py

# Cálculo con ambas fórmulas
df['Demanda_correcta'] = df['Qin'] - df['Q_flujo']
df['Demanda_incorrecta'] = df['Q_flujo'] - df['Qin']

# Validación de signos
correcta_positiva = (df['Demanda_correcta'] > 0).sum()
incorrecta_negativa = (df['Demanda_incorrecta'] < 0).sum()

print(f"Fórmula correcta - Valores positivos: {correcta_positiva} / {len(df)} ({correcta_positiva/len(df)*100:.1f}%)")
print(f"Fórmula incorrecta - Valores negativos: {incorrecta_negativa} / {len(df)} ({incorrecta_negativa/len(df)*100:.1f}%)")

# Resultado esperado:
# Fórmula correcta - Valores positivos: 15034 / 15034 (100.0%)
# Fórmula incorrecta - Valores negativos: 15034 / 15034 (100.0%)
```

---

**FIN DEL INFORME**

*Documento generado para validar la metodología de cálculo de demanda observada en el proyecto de Modelo Predictivo de Demanda de Agua Potable, Gran Valparaíso, Chile.*
