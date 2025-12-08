# Nomenclatura Estandarizada del Proyecto

## Fecha de Estandarización: 6 de diciembre, 2025

---

## Variables Principales

### Q_net (Flujo Neto del Sistema)

**Nombre estándar:** `Q_net` o `Q_net_m3h`  
**Unidades:** m³/hr (metros cúbicos por hora)

**Definición física:**
```
Q_net = ΔVol/Δt = Tasa de cambio del volumen almacenado
```

**Interpretación:**
- **Q_net > 0**: Sistema almacenando agua (llenado de estanques)
- **Q_net < 0**: Sistema descargando agua (vaciado de estanques)
- **Q_net ≈ 0**: Régimen estacionario (equilibrio entre producción y consumo)

**Nombres DEPRECADOS:**
- ~~Q_flujo~~ ❌ (antiguo nombre en datos RAW)
- ~~QFlujo~~ ❌
- ~~flujo~~ ❌

**Nota:** Q_net es la variable TARGET que predicen los modelos de ML.

---

### Qin (Producción Total)

**Nombre estándar:** `Qin` o `Qin_m3h`  
**Unidades:** m³/hr

**Definición:**
```
Qin = Caudal total de producción ingresando al sistema
```

**Fuente de datos:** `BD_Qin_m3_Local.csv`

---

### Qout (Demanda Real del Sistema)

**Nombre estándar:** `Qout` o `Demanda` o `D_obs`  
**Unidades:** m³/hr

**Definición:**
```
Qout = Demanda = Consumo real del sistema
```

**Cálculo:**
```
Qout = Qin - Q_net
```

**Nota:** Qout siempre debe ser ≥ 0 (no puede haber demanda negativa).

---

## Balance Hídrico del Sistema

### Ecuación Fundamental

```
Qin = Qout + Q_net + Pérdidas
```

Donde asumimos: **Pérdidas ≈ 0**

Por lo tanto:
```
Qin = Qout + Q_net
```

### Despejando las variables

**Para calcular demanda:**
```
Qout = Qin - Q_net
```

**Para calcular producción requerida:**
```
Qin_requerido = Qout_predicha + Q_net_objetivo
```

Donde:
- `Qout_predicha`: Demanda estimada para el periodo
- `Q_net_objetivo`: Cambio de almacenamiento deseado (ej: +500 m³/hr para recargar)

---

## Features Derivadas de Q_net

### Lags (Rezagos temporales)

- `Q_net_m3h__lag_168h`: Q_net de hace 1 semana (168 horas)
  - **USO:** Captura patrón semanal (legítimo para predicción)
  
### Diferencias

- `Q_net_m3h__diff_168h`: Diferencia con hace 1 semana
  ```
  diff_168h = Q_net(t) - Q_net(t-168)
  ```

### EMAs (DEPRECADAS - Contienen data leakage)

- ~~`Q_net_m3h__ema_win_6h`~~ ❌ **ELIMINADA**
- ~~`Q_net_m3h__ema_win_12h`~~ ❌ **ELIMINADA**
- ~~`Q_net_m3h__ema_win_24h`~~ ❌ **ELIMINADA**

**Razón de eliminación:** Estas features usan ventanas de corto plazo que contienen datos del test set al predecir, causando data leakage tipo "1-step ahead".

---

## Archivos de Datos

### Datos RAW

| Archivo | Columna Original | Columna Renombrada | Estado |
|---------|------------------|-------------------|--------|
| `BD_Q_flujo_x_Hr_m3hr_LIMPIO.csv` | `Q_flujo_m3hr` | `Q_net_m3h` | ✅ Renombrar |
| `BD_Qin_m3_Local.csv` | `Qin_m3_hr` | `Qin_m3h` | ✅ OK |
| `BD_VolTotal_X_Hr_m3_UTC.csv` | `Volumen_Total_m3` | - | ✅ OK |

### Datos Procesados

| Archivo | Target | Features Q_net |
|---------|--------|---------------|
| `dataset_features_completo.csv` | `Q_net_m3h` | `Q_net_m3h__lag_168h`, `Q_net_m3h__diff_168h` |
| `data_train.csv` | `Q_net_m3h` | ✅ OK |
| `data_validation.csv` | `Q_net_m3h` | ✅ OK |
| `data_test.csv` | `Q_net_m3h` | ✅ OK |

---

## Uso en Código

### ✅ CORRECTO

```python
# Predecir Q_net
q_net_predicho = modelo.predict(X)

# Calcular demanda
qin_hora = perfil_qin[hora]
qout = qin_hora - q_net_predicho

# Clasificar estado
if q_net_predicho < -500:
    estado = 'DESCARGA'  # Alta demanda
elif q_net_predicho > 500:
    estado = 'RECARGA'   # Baja demanda
else:
    estado = 'EQUILIBRIO'
```

### ❌ INCORRECTO (Nombres antiguos)

```python
# NO usar estos nombres:
q_flujo = modelo.predict(X)  # ❌
qout = qin_hora - q_flujo    # ❌

# Tampoco mezclar:
q_net = modelo.predict(X)
q_flujo = q_net              # ❌ Confuso, usar solo q_net
```

---

## Convención de Nombres

### Variables en código Python

| Tipo | Formato | Ejemplo |
|------|---------|---------|
| Variable local | `snake_case` | `q_net`, `q_net_predicho` |
| Columna DataFrame | `snake_case` | `Q_net_m3h`, `Q_net_m3h__lag_168h` |
| Constante | `UPPER_CASE` | `Q_NET_THRESHOLD = 500` |

### Features en datasets

**Patrón:** `<variable>_<unidad>__<transformación>_<parámetro>`

Ejemplos:
- `Q_net_m3h` → Variable base
- `Q_net_m3h__lag_168h` → Lag de 168 horas
- `Q_net_m3h__diff_168h` → Diferencia con hace 168h
- `clima_temp_c__slope_lin_win_72h` → Pendiente lineal de temperatura en ventana de 72h

---

## Glosario Completo

| Término | Significado | Unidades |
|---------|-------------|----------|
| **Q_net** | Flujo neto del sistema (ΔVol/Δt) | m³/hr |
| **Qin** | Producción total | m³/hr |
| **Qout** | Demanda real del sistema | m³/hr |
| **Vtotal** | Volumen total almacenado en estanques | m³ |
| **ΔVol** | Cambio de volumen | m³ |
| **Δt** | Intervalo de tiempo | hr |

---

## Regímenes de Operación

Basados en el valor de Q_net:

| Q_net (m³/hr) | Régimen | Símbolo | Interpretación |
|---------------|---------|---------|----------------|
| < -500 | DESCARGA | 🔴 | Alta demanda (Qout > Qin) |
| -500 a +500 | EQUILIBRIO | 🟡 | Balance (Qout ≈ Qin) |
| > +500 | RECARGA | 🟢 | Baja demanda (Qin > Qout) |

---

## Referencias en la Tesis

**Capítulo de Metodología - Sección de Variables:**

> "El modelo predictivo se centra en estimar Q_net, definido como el flujo neto del sistema (Q_net = ΔVol/Δt), que representa la tasa de cambio del volumen almacenado en los estanques del sistema de distribución. Un valor positivo de Q_net indica que el sistema está almacenando agua (llenado), mientras que un valor negativo indica descarga (vaciado).
>
> El balance hídrico del sistema se expresa como:
> ```
> Qin = Qout + Q_net
> ```
> Donde Qin es la producción total, Qout la demanda real del sistema, y Q_net el cambio en el almacenamiento. A partir de Q_net predicho y el perfil histórico de Qin, se estima la demanda futura (Qout = Qin - Q_net), permitiendo optimizar la producción requerida."

---

**Autor:** Equipo de Desarrollo  
**Proyecto:** Modelo Predictivo Demanda Agua Potable - Gran Valparaíso  
**Última actualización:** 6 de diciembre, 2025
