# HALLAZGOS: Valores Bisagra y Patrones Condicionales

**Análisis de Calendar + Clima + Qin + Q_flujo (las 4 fuentes simultáneamente)**

---

## 🎯 VALORES BISAGRA IDENTIFICADOS (Con Sustento de Datos)

### 1. **Umbrales de Temperatura**

| Umbral | Valor | Comportamiento |
|--------|-------|----------------|
| **Q25 (Frío)** | **10.5°C** | Demanda baja (+145.8 m³/hr promedio) |
| **Q75 (Calor)** | **16.3°C** | Demanda alta (-1,230 m³/hr promedio) |

**Interpretación**: 
- Por debajo de 10.5°C: Sistema en recuperación (más recuperación que demanda)
- Por encima de 16.3°C: Alta demanda de agua potable (consumo supera recuperación)
- **Cambio crítico**: Entre 16-18°C la demanda aumenta 29,546% (de -1 m³/hr a -428 m³/hr)

### 2. **Horas Críticas del Día**

| Hora | Tipo | Valor | Contexto |
|------|------|-------|----------|
| **12:00** | Máxima Demanda | **-5,222 m³/hr** | Mediodía - pico de consumo |
| **04:00** | Máxima Recuperación | **+5,149 m³/hr** | Madrugada - llenado de tanques |

**Horas bisagra (cambios >30% entre horas consecutivas)**:
- **05h → 06h**: Caída 35-57% (inicia demanda matutina)
- **06h → 07h**: Caída 55-67% (personas se despiertan)
- **07h → 08h**: Caída 47-223% (**crítico**: inicio actividades)
- **08h → 09h**: Aumento 120-844% (demanda se acelera)
- **21h → 22h**: Caída 106-367% (fin de actividades)
- **22h → 23h**: Aumento 124-4,170% (**crítico**: inicio recuperación)

### 3. **Umbrales de Producción (Qin)**

| Umbral | Valor | Contexto |
|--------|-------|----------|
| **Q25 (Bajo)** | **10,998 m³/hr** | Producción reducida (típico en frío: 8.6°C) |
| **Q75 (Alto)** | **12,614 m³/hr** | Producción elevada (típico en calor: 20.2°C) |

**Confirmación**: Mayor temperatura → Mayor Qin → Respuesta a mayor consumo

---

## 📊 PATRONES CONDICIONALES DESCUBIERTOS

### **Patrón 1: Año Nuevo depende del clima**

| Clima | n | Temp | Q_net | Qin | vs Baseline |
|-------|---|------|-------|-----|-------------|
| Normal | 23 | 13.5°C | **+1,671 m³/hr** | 12,953 | **+1,393 m³/hr** recuperación |
| Calor | 22 | 19.5°C | **-1,895 m³/hr** | 12,669 | **-669 m³/hr** más demanda |

**Hallazgo clave**: 
- Año Nuevo con clima normal: sistema se recupera (+1,671)
- Año Nuevo con calor: alta demanda (-1,895)
- **Diferencia**: 3,566 m³/hr (214% de cambio según clima!)

### **Patrón 2: Fiestas Patrias muestra comportamiento inverso**

| Clima | n | Temp | Q_net | Qin | vs Baseline |
|-------|---|------|-------|-----|-------------|
| Frío | 61 | 9.0°C | **+643 m³/hr** | 10,511 | +505 recuperación |
| Normal | 26 | 13.3°C | **-1,056 m³/hr** | 10,924 | **-1,343 demanda** |
| Calor | 5 | 17.6°C | **+486 m³/hr** | 11,061 | +1,718 recuperación |

**Hallazgo clave**: 
- En clima normal (13.3°C), Fiestas Patrias genera **MÁS demanda** que baseline (-1,343)
- En calor (17.6°C), recuperación aumenta (+1,718)
- **Hipótesis**: En septiembre (invierno-primavera), asados y celebraciones aumentan consumo

### **Patrón 3: Festival de Viña con clima cálido**

| Clima | n | Temp | Q_net | Qin | vs Baseline |
|-------|---|------|-------|-----|-------------|
| Normal | 171 | 14.3°C | **+567 m³/hr** | 12,486 | +291 recuperación |
| Calor | 135 | 20.0°C | **-735 m³/hr** | 12,713 | +513 demanda |

**Hallazgo clave**: 
- Festival en calor: demanda aumenta 513 m³/hr vs baseline
- Verano + turismo + evento masivo = alta demanda

### **Patrón 4: Fines de Semana relativamente estables**

| Clima | n | Temp | Q_net | Qin | Δ vs Baseline |
|-------|---|------|-------|-----|---------------|
| Frío | 1,145 | 8.6°C | +176 | 11,136 | +44 |
| Normal | 2,079 | 13.2°C | +274 | 11,816 | -12 |
| Calor | 1,081 | 20.2°C | -1,246 | 12,520 | -23 |

**Hallazgo clave**: 
- Fines de semana muestran POCA diferencia vs días laborales (-12 a +44)
- Clima domina sobre el efecto calendario en este caso

### **Patrón 5: Vacaciones Escolares altamente dependientes del clima**

| Clima | n | Temp | Q_net | Qin | Δ vs Baseline |
|-------|---|------|-------|-----|---------------|
| Frío | 268 | 7.7°C | **+318** | 11,305 | **+185 recuperación** |
| Normal | 202 | 13.3°C | **-809** | 11,340 | **-1,122 demanda** |
| Calor | 77 | 19.6°C | **-1,293** | 11,459 | **-64 demanda** |

**Hallazgo clave**: 
- En frío (julio): recuperación aumenta +185 (menos gente en casa)
- En clima normal: demanda aumenta -1,122 (actividades en casa)
- **Rango total**: 1,611 m³/hr de diferencia según clima

---

## 🔥 RELACIÓN TEMPERATURA × QIN × DEMANDA

### Tabla Cruzada (valores promedio Q_net en m³/hr):

|  | Qin Muy Bajo | Qin Bajo | Qin Normal | Qin Alto | Qin Muy Alto |
|--------------|---------|------|--------|------|----------|
| **Muy Frío**  | +452 | +15 | -625 | **+731** | +148 |
| **Frío**      | +58 | +182 | +687 | **+746** | **+952** |
| **Normal**    | -351 | -175 | +357 | **+470** | **+1,162** |
| **Cálido**    | -586 | -1,072 | -813 | **+136** | **+1,053** |
| **Muy Cálido** | **-1,822** | **-1,506** | **-1,477** | **-1,597** | **-1,208** |

**Hallazgos clave**:
1. **Qin Alto + Temperaturas frías**: MÁXIMA recuperación (+746 a +952)
2. **Temperaturas muy cálidas (>21°C)**: Alta demanda **independiente** de Qin (-1,208 a -1,822)
3. **Patrón diagonal**: A mayor temperatura, se requiere mayor Qin para mantener balance

---

## 🌡️ CAMBIOS DE TEMPERATURA vs CAMBIOS DE DEMANDA

### Ventana de 6 horas (la más correlacionada: -0.487):

| Condición | ΔT 6h | Q_net | Qin | n |
|-----------|-------|-------|-----|---|
| **Enfriamiento fuerte** | **-5.9°C** | **+3,802** | 11,928 | 3,807 |
| Estable | -0.3°C | -1,047 | 11,535 | 7,467 |
| **Calentamiento fuerte** | **+6.6°C** | **-2,295** | 12,143 | 3,754 |

**Hallazgo crítico**:
- **Rango total**: 6,097 m³/hr de diferencia entre enfriamiento (-6°C) y calentamiento (+7°C)
- Si la temperatura sube 6.6°C en 6 horas → demanda aumenta 2,295 m³/hr
- Si la temperatura baja 5.9°C en 6 horas → recuperación aumenta 3,802 m³/hr
- **Este es el patrón más fuerte identificado**

### Ventana de 3 horas:

| Condición | ΔT 3h | Q_net | Diferencia |
|-----------|-------|-------|------------|
| Enfriamiento | -3.5°C | **+2,707** | — |
| Calentamiento | +3.9°C | **-2,795** | **5,502 m³/hr** |

### Ventana de 1 hora:

| Condición | ΔT 1h | Q_net | Diferencia |
|-----------|-------|-------|------------|
| Enfriamiento | -1.4°C | **+1,601** | — |
| Calentamiento | +1.6°C | **-2,636** | **4,237 m³/hr** |

**Conclusión**: A menor ventana temporal, **mayor la reactividad** del consumo a cambios de temperatura

---

## 📈 IMPLICACIONES PARA EL MODELO

### 1. **Features más importantes identificadas (con evidencia)**:
- ✅ `clima_temp_delta_6h` (correlación: -0.487)
- ✅ `clima_temp_delta_3h` (correlación: -0.441)
- ✅ `clima_temp_c__slope_lin_win_6h` (correlación: -0.471)
- ✅ `hora` × `dia_semana` (horas bisagra varían por día)
- ✅ `sist_Qin_m3h` × `clima_temp_c` (tabla cruzada muestra interacciones)

### 2. **Umbrales para features categóricas**:

```python
# Temperatura
df['temp_categoria'] = pd.cut(df['clima_temp_c'], 
                              bins=[0, 10.5, 16.3, 35],
                              labels=['Frío', 'Normal', 'Calor'])

# Qin
df['qin_categoria'] = pd.cut(df['sist_Qin_m3h'], 
                             bins=[0, 10998, 12614, 20000],
                             labels=['Bajo', 'Normal', 'Alto'])

# Hora del día
df['periodo_dia'] = pd.cut(df['hora'], 
                           bins=[0, 6, 12, 18, 24],
                           labels=['Madrugada', 'Mañana', 'Tarde', 'Noche'])
```

### 3. **Interacciones críticas a incluir**:
- `temp_categoria × feriado × clima`
- `delta_temp_6h × hora_del_dia`
- `qin_categoria × temp_categoria`
- `evento_especial × mes × temp_categoria`

### 4. **Necesidad de más años de datos**:
Confirmado: eventos como "Año Nuevo en miércoles vs sábado" tienen **SOLO 1-2 observaciones por año**. Con dataset actual (2024-09/2025):
- Año Nuevo 2024: n=24 (probablemente lunes)
- Año Nuevo 2025: n=21 (probablemente miércoles)
- **Insuficiente** para analizar efecto día de la semana en eventos anuales únicos

---

## 🎯 CONCLUSIONES FINALES

1. **Los valores NO fueron inventados**: Todos los umbrales tienen sustento en los datos analizados

2. **Clima domina sobre calendario**: 
   - Temperatura explica más varianza que eventos especiales
   - PERO la combinación (evento + clima) amplifica o atenúa efectos

3. **Cambios de temperatura > Valores absolutos**:
   - `delta_temp_6h` 4x más correlacionado que `temp_c`
   - Sistema reacciona a **variaciones**, no solo a valores estáticos

4. **Horas bisagra validadas**:
   - 07h-09h: transición crítica recuperación → demanda
   - 12h: pico de demanda
   - 22h-23h: transición crítica demanda → recuperación
   - 04h: pico de recuperación

5. **Qin confirma consumo**:
   - En calor (>16.3°C): Qin aumenta +1,318 m³/hr (12,464 vs 11,146)
   - Correlación: alta temperatura → alta demanda → alta producción

6. **Eventos especiales requieren análisis condicional**:
   - "Año Nuevo" sin contexto: insuficiente
   - "Año Nuevo + Calor": -1,895 m³/hr (alta demanda)
   - "Año Nuevo + Normal": +1,671 m³/hr (recuperación)
   - **Diferencia**: 3,566 m³/hr (214%)

---

## 📁 Archivos Generados

1. **outputs/valores_bisagra_umbrales.csv**: Tabla resumen de umbrales críticos
2. **outputs/patrones_eventos_clima.csv**: Análisis condicional eventos × clima
3. **Terminal output completo**: Tablas hora × día, temperatura × Qin, cambios temporales

**Estos valores ahora tienen sustento estadístico y pueden usarse para configurar el modelo predictivo.**
