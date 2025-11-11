"""
Análisis detallado de variaciones en los datos históricos
Para verificar si realmente hay diferencias por estación, clima, día de semana, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 80)
print("ANÁLISIS DE VARIACIÓN REAL EN DATOS HISTÓRICOS")
print("=" * 80)

# Cargar datos procesados con clima
data_path = Path('data/processed/data_processed_complete.csv')
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

# Convertir timestamp a datetime
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
df = df.sort_values('timestamp_utc')

print(f"\n📊 Datos cargados: {len(df)} registros")
print(f"Período: {df['timestamp_utc'].min()} a {df['timestamp_utc'].max()}")

# Identificar columna de volumen
volumen_col = None
for col in [' Volumen_Total_m3', 'Volumen_Total_m3', 'volumen_total_m3']:
    if col in df.columns:
        volumen_col = col
        break

if volumen_col is None:
    print("❌ No se encontró columna de volumen")
    print(f"Columnas disponibles: {df.columns.tolist()[:10]}")
    exit(1)

print(f"✅ Columna de volumen: {volumen_col}")

# Agregar información temporal (si no existen ya)
if 'hora' not in df.columns:
    df['hora'] = df['timestamp_utc'].dt.hour
if 'dia_semana' not in df.columns:
    df['dia_semana'] = df['timestamp_utc'].dt.dayofweek
if 'mes' not in df.columns:
    df['mes'] = df['timestamp_utc'].dt.month

df['dia_semana_nombre'] = df['timestamp_utc'].dt.day_name()
df['es_fin_semana'] = df['dia_semana'].isin([5, 6])

# Definir estaciones (hemisferio sur)
def get_estacion(mes):
    if mes in [12, 1, 2]:
        return 'Verano'
    elif mes in [3, 4, 5]:
        return 'Otoño'
    elif mes in [6, 7, 8]:
        return 'Invierno'
    else:
        return 'Primavera'

df['estacion'] = df['mes'].apply(get_estacion)

print("\n" + "=" * 80)
print("1️⃣ ANÁLISIS POR ESTACIÓN")
print("=" * 80)

estaciones = df.groupby('estacion')[volumen_col].agg(['mean', 'std', 'min', 'max', 'count'])
estaciones = estaciones.reindex(['Verano', 'Otoño', 'Invierno', 'Primavera'])
print("\n📊 Consumo promedio por estación:")
print(estaciones.to_string())

# Calcular variación porcentual
verano = estaciones.loc['Verano', 'mean']
invierno = estaciones.loc['Invierno', 'mean']
var_estacional = ((verano - invierno) / invierno) * 100
print(f"\n🌡️  Variación Verano vs Invierno: {var_estacional:.2f}%")
print(f"   Verano: {verano:,.0f} m³/hora")
print(f"   Invierno: {invierno:,.0f} m³/hora")
print(f"   Diferencia: {abs(verano - invierno):,.0f} m³/hora")

print("\n" + "=" * 80)
print("2️⃣ ANÁLISIS POR DÍA DE SEMANA")
print("=" * 80)

dias = df.groupby('dia_semana_nombre')[volumen_col].agg(['mean', 'std', 'count'])
orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dias = dias.reindex(orden_dias)
print("\n📊 Consumo promedio por día:")
print(dias.to_string())

# Laboral vs fin de semana
laboral = df[~df['es_fin_semana']][volumen_col].mean()
fin_semana = df[df['es_fin_semana']][volumen_col].mean()
var_semana = ((laboral - fin_semana) / fin_semana) * 100
print(f"\n📅 Días laborales vs Fin de semana:")
print(f"   Laborales: {laboral:,.0f} m³/hora")
print(f"   Fin de semana: {fin_semana:,.0f} m³/hora")
print(f"   Variación: {var_semana:.2f}%")
print(f"   Diferencia: {abs(laboral - fin_semana):,.0f} m³/hora")

print("\n" + "=" * 80)
print("3️⃣ ANÁLISIS POR HORA DEL DÍA")
print("=" * 80)

horas = df.groupby('hora')[volumen_col].agg(['mean', 'std'])
print("\n📊 Consumo por hora del día:")
print(horas.to_string())

hora_max = horas['mean'].idxmax()
hora_min = horas['mean'].idxmin()
consumo_max = horas.loc[hora_max, 'mean']
consumo_min = horas.loc[hora_min, 'mean']
var_diaria = ((consumo_max - consumo_min) / consumo_min) * 100

print(f"\n⏰ Variación durante el día:")
print(f"   Hora pico: {hora_max}:00 → {consumo_max:,.0f} m³/hora")
print(f"   Hora valle: {hora_min}:00 → {consumo_min:,.0f} m³/hora")
print(f"   Variación: {var_diaria:.2f}%")
print(f"   Diferencia: {consumo_max - consumo_min:,.0f} m³/hora")

print("\n" + "=" * 80)
print("4️⃣ ANÁLISIS POR CLIMA (Temperatura)")
print("=" * 80)

# Buscar columna de temperatura
temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'temperatura' in col.lower()]
if temp_cols:
    temp_col = temp_cols[0]
    print(f"✅ Columna temperatura encontrada: {temp_col}")
    
    # Crear categorías de temperatura
    df['temp_categoria'] = pd.cut(df[temp_col], 
                                   bins=[-np.inf, 10, 15, 20, np.inf],
                                   labels=['Frío (<10°C)', 'Templado (10-15°C)', 
                                          'Cálido (15-20°C)', 'Caluroso (>20°C)'])
    
    temp_analysis = df.groupby('temp_categoria')[volumen_col].agg(['mean', 'std', 'count'])
    print("\n📊 Consumo por temperatura:")
    print(temp_analysis.to_string())
    
    # Correlación temperatura-consumo
    corr = df[[temp_col, volumen_col]].corr().iloc[0, 1]
    print(f"\n🌡️  Correlación temperatura-consumo: {corr:.4f}")
    
    frio = df[df[temp_col] < 10][volumen_col].mean()
    calor = df[df[temp_col] > 20][volumen_col].mean()
    if not np.isnan(frio) and not np.isnan(calor):
        var_temp = ((calor - frio) / frio) * 100
        print(f"   Consumo con frío (<10°C): {frio:,.0f} m³/hora")
        print(f"   Consumo con calor (>20°C): {calor:,.0f} m³/hora")
        print(f"   Variación: {var_temp:.2f}%")
else:
    print("❌ No se encontró columna de temperatura en los datos")

print("\n" + "=" * 80)
print("5️⃣ ANÁLISIS POR PRECIPITACIÓN")
print("=" * 80)

# Buscar columna de precipitación
precip_cols = [col for col in df.columns if 'precip' in col.lower() or 'lluvia' in col.lower() or 'rain' in col.lower()]
if precip_cols:
    precip_col = precip_cols[0]
    print(f"✅ Columna precipitación encontrada: {precip_col}")
    
    df['hay_lluvia'] = df[precip_col] > 0
    
    sin_lluvia = df[~df['hay_lluvia']][volumen_col].mean()
    con_lluvia = df[df['hay_lluvia']][volumen_col].mean()
    
    dias_lluvia = df['hay_lluvia'].sum()
    dias_sin_lluvia = (~df['hay_lluvia']).sum()
    
    print(f"\n🌧️  Consumo según precipitación:")
    print(f"   Sin lluvia: {sin_lluvia:,.0f} m³/hora ({dias_sin_lluvia} registros)")
    print(f"   Con lluvia: {con_lluvia:,.0f} m³/hora ({dias_lluvia} registros)")
    
    if dias_lluvia > 0:
        var_lluvia = ((con_lluvia - sin_lluvia) / sin_lluvia) * 100
        print(f"   Variación: {var_lluvia:.2f}%")
        print(f"   Diferencia: {abs(con_lluvia - sin_lluvia):,.0f} m³/hora")
else:
    print("❌ No se encontró columna de precipitación en los datos")

print("\n" + "=" * 80)
print("6️⃣ ANÁLISIS COMBINADO: DÍA LABORAL vs FIN DE SEMANA POR HORA")
print("=" * 80)

laboral_por_hora = df[~df['es_fin_semana']].groupby('hora')[volumen_col].mean()
finde_por_hora = df[df['es_fin_semana']].groupby('hora')[volumen_col].mean()

print("\n📊 Diferencias por hora:")
print(f"{'Hora':<8} {'Laboral':<15} {'Fin Semana':<15} {'Diferencia':<15} {'% Var':<10}")
print("-" * 70)
for hora in range(24):
    lab = laboral_por_hora.get(hora, 0)
    fin = finde_por_hora.get(hora, 0)
    dif = lab - fin
    pct = ((lab - fin) / fin * 100) if fin > 0 else 0
    print(f"{hora:02d}:00    {lab:>10,.0f} m³   {fin:>10,.0f} m³   {dif:>10,.0f} m³   {pct:>6.2f}%")

print("\n" + "=" * 80)
print("7️⃣ RESUMEN EJECUTIVO")
print("=" * 80)

print("\n🎯 MAGNITUD DE VARIACIONES:")
print(f"   1. Variación HORA DEL DÍA:     {var_diaria:>6.2f}% ({consumo_max - consumo_min:>8,.0f} m³/hora)")
print(f"   2. Variación ESTACIONAL:       {var_estacional:>6.2f}% ({abs(verano - invierno):>8,.0f} m³/hora)")
print(f"   3. Variación DÍA SEMANA:       {var_semana:>6.2f}% ({abs(laboral - fin_semana):>8,.0f} m³/hora)")
if temp_cols and not np.isnan(frio) and not np.isnan(calor):
    print(f"   4. Variación TEMPERATURA:      {var_temp:>6.2f}% ({abs(calor - frio):>8,.0f} m³/hora)")

print("\n⚠️  CONCLUSIÓN:")
if var_diaria > var_estacional and var_diaria > var_semana:
    print("   ✅ La HORA DEL DÍA es efectivamente el factor dominante")
    print(f"   ✅ Pero la variación estacional ({var_estacional:.2f}%) y semanal ({var_semana:.2f}%)")
    print("      SÍ son significativas y el modelo DEBE capturarlas")
else:
    print("   ⚠️  Hay otros factores tan importantes como la hora del día")
    print("      El modelo necesita mejorar la captura de estos patrones")

print("\n" + "=" * 80)
print("ANÁLISIS COMPLETADO")
print("=" * 80)
