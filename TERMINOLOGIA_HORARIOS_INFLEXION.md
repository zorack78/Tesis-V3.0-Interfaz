# 📘 Terminología: Horarios de Inflexión

## 🎯 Definición

**Horarios de Inflexión** son los momentos del día donde la demanda de agua presenta cambios significativos, alcanzando valores máximos o mínimos.

## 📊 Características

### Horario de Inflexión Máximo
- **Horario:** 12:00 - 14:00 (mediodía)
- **Demanda promedio:** ~17,000 m³/hr
- **Significado:** Momento de mayor consumo del día
- **Causa:** Actividades diurnas en su punto máximo

### Horario de Inflexión Mínimo
- **Horario:** 02:00 - 05:00 (madrugada)
- **Demanda promedio:** ~5,000-6,000 m³/hr
- **Significado:** Momento de menor consumo del día
- **Causa:** Actividad reducida durante la noche

## 🔄 Diferencia con Terminología Anterior

| Término Anterior | Término Actual | Razón del Cambio |
|-----------------|----------------|------------------|
| Hora pico | Horario de inflexión máximo | Más preciso técnicamente |
| Hora valle | Horario de inflexión mínimo | Evita confusión con topología |
| Pico/Valle | Máximo/Mínimo | Claridad conceptual |

## 📈 Interpretación en el Sistema

### Para Volumen Almacenado (variable antigua)
- **Máximo a las 7:00 AM:** Sistema lleno antes del consumo diurno
- NO representa demanda, sino almacenamiento

### Para Demanda (variable actual - CORRECTA)
- **Máximo a las 12:00 PM:** Horario de inflexión donde se consume más
- **Mínimo a las 4:00 AM:** Horario de inflexión donde se consume menos
- SÍ representa consumo real del sistema

## 💡 Uso en Contexto

### ✅ Correcto:
- "El horario de inflexión máximo es a las 12:00"
- "La demanda alcanza su máximo alrededor del mediodía"
- "El horario de mínimo consumo es en la madrugada"
- "Horarios de inflexión: máximo 12:00, mínimo 4:00"

### ❌ Evitar (confuso):
- "Hora pico" (puede confundirse con tráfico vehicular)
- "Hora valle" (puede confundirse con geografía)
- "Peak hour" (anglicismo)

## 🌍 Contexto Internacional

En la literatura técnica se usan diversos términos:
- **Peak demand hours** (inglés) → Horarios de inflexión máxima
- **Off-peak hours** (inglés) → Horarios de inflexión mínima
- **Horas de punta** (español) → Horarios de inflexión máxima
- **Horas de demanda máxima/mínima** → Descriptivo y claro

## 📋 Aplicación en el Modelo

El modelo identifica estos horarios de inflexión mediante:

1. **Análisis de datos históricos:** 15,222 registros de demanda real
2. **Feature 'hour':** Importancia 14.14% en el modelo
3. **Patrones aprendidos:** El modelo reconoce automáticamente los horarios donde la demanda cambia

### Horarios de Inflexión por Estación

| Estación | Máximo | Mínimo | Diferencia |
|----------|--------|--------|------------|
| Verano | 18,455 m³/hr (12:00) | 4,557 m³/hr (4:00) | +14.9% vs invierno |
| Invierno | 16,254 m³/hr (12:00) | 5,126 m³/hr (4:00) | Baseline |

## 🎓 Nota Técnica

Los "horarios de inflexión" representan puntos de cambio en la curva de demanda:

```
Demanda (m³/hr)
    │
18k │        ┌───── Inflexión máxima (12:00)
    │       ╱ ╲
12k │     ╱     ╲
    │   ╱         ╲
 6k │ ╱             ╲___
    │╱                   ╲
    └─────────────────────── Hora del día
    0  4  8  12  16  20  24
       ↑           
    Inflexión mínima (4:00)
```

Estos puntos son fundamentales para:
- **Planificación operativa:** Dimensionar capacidad
- **Gestión de recursos:** Optimizar bombeo
- **Mantenimiento:** Programar en horarios de baja demanda
- **Predicción:** Validar que el modelo capture el comportamiento real

---

**Fecha de actualización:** 11 de Noviembre 2025  
**Versión:** 1.0
