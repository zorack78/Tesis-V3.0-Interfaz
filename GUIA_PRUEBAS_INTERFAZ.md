# 🧪 Guía Rápida de Pruebas - Interfaz con Gráficos

## 🚀 Iniciar Interfaz

```bash
python interfaz_demanda_v1.py
```

**Salida esperada:**
```
🔄 Inicializando sistema de predicción de DEMANDA...
✅ Modelo de DEMANDA cargado
✅ Features cargadas: 21
✅ Datos históricos cargados: 15222 registros
================================================================================
🚀 INICIANDO INTERFAZ DE PREDICCIÓN DE DEMANDA
================================================================================
* Running on local URL:  http://0.0.0.0:7870
```

**Acceder:** http://localhost:7870

---

## 🎯 Pestaña 1: Predicción Puntual

### Prueba 1 - Mediodía (Hora Pico)
```
Fecha: 2025-12-25
Hora: 12
```

**Resultado esperado:**
- 📊 Gráfico: Curva azul del día completo
- ⭐ Estrella roja en hora 12 (máximo del día)
- 📝 Demanda: ~17,000 m³/hr (alta)
- 🎯 Período: "Mediodía"

### Prueba 2 - Madrugada (Hora Valle)
```
Fecha: 2025-12-25
Hora: 4
```

**Resultado esperado:**
- 📊 Gráfico: Curva azul del día completo
- ⭐ Estrella roja en hora 4 (mínimo del día)
- 📝 Demanda: ~5,000-8,000 m³/hr (baja)
- 🎯 Período: "Madrugada"

---

## 📅 Pestaña 2: Predicción Día Completo

### Prueba 3 - Día de Verano
```
Fecha: 2025-01-15 (Verano)
```

**Resultado esperado:**
- 📊 **Gráfico 1:** Serie temporal 24h
  - ⭐ Estrella roja: Hora ~12:00 (máximo)
  - ⭐ Estrella verde: Hora ~04:00 (mínimo)
  - Línea naranja: Promedio ~12,500-13,000 m³/hr
  
- 📊 **Gráfico 2:** Barras por período
  - Tarde (12-18): Mayor demanda
  - Madrugada (00-06): Menor demanda
  
- 📝 **Estadísticas:**
  - Total día: ~300,000 m³
  - Promedio: ~12,500 m³/hr
  - Variación: 40-60%

### Prueba 4 - Día de Invierno
```
Fecha: 2025-07-15 (Invierno)
```

**Resultado esperado:**
- 📊 Mismos gráficos, valores menores
- 📝 **Estadísticas:**
  - Promedio: ~11,000 m³/hr (menor que verano)
  - Patrón similar pero demanda reducida ~15%

---

## 📈 Pestaña 3: Predicción 72 Horas

### Prueba 5 - 3 Días Continuos (ANTES FALLABA ❌ AHORA FUNCIONA ✅)
```
Fecha Inicio: 2025-12-25
```

**Resultado esperado:**
- 📊 **Gráfico 1:** Curva continua 72 horas
  - 3 ciclos día-noche visibles
  - ⭐ 5 Estrellas rojas: Picos (mediodías)
  - ⭐ 5 Estrellas verdes: Valles (madrugadas)
  - Patrón repetitivo claro
  
- 📊 **Gráfico 2:** Barras agrupadas por día
  - 3 grupos (Día 1, 2, 3)
  - Cada grupo: Promedio, Máximo, Mínimo
  - Valores sobre barras
  
- 📝 **Estadísticas:**
  - Total 72h: ~900,000 m³
  - Promedio: ~12,500 m³/hr
  - Resumen por día con totales

### Prueba 6 - Verificación de Corrección de Fecha
```
Fecha Inicio: 2025-06-15
```

**Antes:** ❌ Error: time data '15/06/2025 00:00' does not match format '%Y-%m-%d %H:%M'

**Ahora:** ✅ Funciona correctamente
- Sistema detecta formato DD/MM/YYYY automáticamente
- Convierte a datetime correctamente
- Gráficos se generan sin errores

---

## ✅ Checklist de Validación

### Visual
- [ ] Todos los gráficos se muestran correctamente
- [ ] Estrellas rojas en máximos visibles
- [ ] Estrellas verdes en mínimos visibles
- [ ] Curvas suaves sin saltos
- [ ] Colores correctos (azul, rojo, verde, naranja)
- [ ] Títulos y etiquetas legibles
- [ ] Leyendas presentes

### Funcional
- [ ] Pestaña 1: Gráfico + texto aparecen
- [ ] Pestaña 2: 2 gráficos + texto aparecen
- [ ] Pestaña 3: 2 gráficos + texto aparecen (SIN ERROR DE FECHA)
- [ ] Cambiar fecha funciona en todas las pestañas
- [ ] Cambiar hora funciona en pestaña 1
- [ ] Valores numéricos coinciden con gráficos

### Patrones
- [ ] Verano > Invierno (diferencia ~15%)
- [ ] Pico: 12:00-14:00
- [ ] Valle: 02:00-05:00
- [ ] Patrón repetitivo en 72h visible
- [ ] Fin de semana vs día laboral (si aplica)

---

## 🐛 Problemas Comunes y Soluciones

### Problema 1: "Error de formato de fecha"
**Antes:** ❌ Ocurría en pestaña 72h
**Solución:** ✅ CORREGIDO - Sistema maneja ambos formatos automáticamente

### Problema 2: "Gráfico no aparece"
**Causa:** Matplotlib backend incorrecto
**Solución:** Verificar `matplotlib.use('Agg')` al inicio de cada función

### Problema 3: "Layout descuadrado"
**Causa:** Proporciones de columnas incorrectas
**Solución:** Verificar `scale=1` para texto, `scale=2` para gráfico

### Problema 4: "Valores muy bajos/altos"
**Causa:** Fecha fuera de rango histórico o features faltantes
**Solución:** Usar fechas en rango 2024-2025, datos disponibles

---

## 📊 Valores de Referencia

### Demanda Típica
```
Verano (Dic-Feb):
  - Promedio: ~12,500-13,000 m³/hr
  - Pico: ~17,000-18,000 m³/hr (12:00-14:00)
  - Valle: ~6,000-8,000 m³/hr (02:00-05:00)

Invierno (Jun-Ago):
  - Promedio: ~10,500-11,500 m³/hr (-15% vs verano)
  - Pico: ~14,000-15,000 m³/hr
  - Valle: ~5,000-7,000 m³/hr
```

### Distribución por Período (Verano)
```
Madrugada (00-06): ~40,000 m³ (13%)
Mañana (06-12): ~75,000 m³ (25%)
Tarde (12-18): ~110,000 m³ (37%) ← MAYOR
Noche (18-24): ~75,000 m³ (25%)
```

---

## 🎯 Casos de Prueba Específicos

### Test 1: Feriado Nacional
```
Fecha: 2025-09-18 (Fiestas Patrias)
Resultado esperado: Demanda puede variar vs día normal
```

### Test 2: Fin de Semana
```
Fecha: 2025-12-27 (Sábado)
Fecha: 2025-12-28 (Domingo)
Resultado esperado: Patrón diferente a día laboral
```

### Test 3: Cambio de Estación
```
Fecha: 2024-06-21 (Inicio Invierno)
Fecha: 2024-12-21 (Inicio Verano)
Resultado esperado: Transición de demanda visible
```

### Test 4: Evento Especial
```
Fecha: 2025-01-01 (Año Nuevo)
Resultado esperado: Patrón atípico posible
```

---

## 📸 Capturas Esperadas

### Pestaña 1 - Predicción Puntual
```
+-------------------+---------------------------+
|   TEXTO           |        GRÁFICO            |
| - Fecha           | [Curva azul 24h]          |
| - Hora: 12        | [★ roja en hora 12]       |
| - Demanda: 17k    | [Área sombreada]          |
| - Estación        | [Eje X: 0-23]             |
| - Período         | [Eje Y: Demanda]          |
+-------------------+---------------------------+
```

### Pestaña 2 - Día Completo
```
+-------------------+---------------------------+
|   TEXTO           |      GRÁFICO 1            |
| - Estadísticas    | [Curva 24h]               |
| - Total día       | [★ roja máx, ★ verde mín] |
| - Promedio        | [Línea naranja promedio]  |
| - Máximo/Mínimo   |                           |
|                   |      GRÁFICO 2            |
|                   | [4 Barras períodos]       |
|                   | [Valores sobre barras]    |
+-------------------+---------------------------+
```

### Pestaña 3 - 72 Horas
```
+-------------------+---------------------------+
|   TEXTO           |      GRÁFICO 1            |
| - 72h Stats       | [Curva continua 3 días]   |
| - Total           | [5★ rojas, 5★ verdes]     |
| - Resumen Día 1   | [Patrón repetitivo]       |
| - Resumen Día 2   |                           |
| - Resumen Día 3   |      GRÁFICO 2            |
|                   | [3 grupos barras]         |
|                   | [Prom, Máx, Mín x día]    |
+-------------------+---------------------------+
```

---

## ⚡ Prueba Rápida (1 minuto)

```bash
# Terminal 1
python interfaz_demanda_v1.py

# Navegador: http://localhost:7870

# Pestaña 1: Ingresar fecha 2025-12-25, hora 12 → Click "Predecir"
# ✅ Ver: Gráfico + texto, estrella roja en hora 12

# Pestaña 2: Ingresar fecha 2025-12-25 → Click "Predecir Día"
# ✅ Ver: 2 gráficos (temporal + barras) + texto

# Pestaña 3: Ingresar fecha 2025-12-25 → Click "Predecir 72h"
# ✅ Ver: 2 gráficos (72h continua + comparación) + texto
# ✅ SIN ERROR DE FECHA

# Terminal 1: Ctrl+C para detener
```

**Tiempo estimado:** 1 minuto
**Criterio de éxito:** Las 3 pestañas muestran gráficos sin errores

---

## 🎉 Resultado Exitoso

Si todas las pruebas pasan:

```
================================================================================
✅ VALIDACIÓN COMPLETA EXITOSA
================================================================================

🎯 Pestaña 1 (Puntual):      ✅ Gráfico + Texto
🎯 Pestaña 2 (Día Completo): ✅ 2 Gráficos + Texto  
🎯 Pestaña 3 (72 Horas):     ✅ 2 Gráficos + Texto (FECHA CORREGIDA)

🐛 Errores de formato:       ✅ 0
📊 Calidad visualizaciones:  ✅ EXCELENTE
🎨 Patrones validados:       ✅ Pico 12:00, Valle 04:00
⚡ Rendimiento:              ✅ <2 segundos por predicción

🎉 ¡INTERFAZ LISTA PARA PRODUCCIÓN!
================================================================================
```

---

**Última actualización:** 2025-01-XX  
**Estado:** ✅ OPERACIONAL  
**Puerto:** 7870
