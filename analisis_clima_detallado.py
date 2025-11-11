"""
Análisis completo con datos de clima
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("ANÁLISIS DETALLADO CON DATOS DE CLIMA")
print("=" * 80)

# Cargar datos
volumen_df = pd.read_csv('data/processed/data_processed_complete.csv')
volumen_df.columns = volumen_df.columns.str.strip()
volumen_df['timestamp_utc'] = pd.to_datetime(volumen_df['timestamp_utc'])

clima_df = pd.read_csv('data/processed/clima_chile_v3.csv')
clima_df['timestamp'] = pd.to_datetime(clima_df['timestamp'])

print(f"\n📊 Volumen: {len(volumen_df)} registros")
print(f"📊 Clima: {len(clima_df)} registros")

# Unir datos
df = pd.merge(volumen_df, clima_df, 
              left_on='timestamp_utc', right_on='timestamp', how='inner')

print(f"✅ Datos unidos: {len(df)} registros")

# Columna de volumen
volumen_col = 'Volumen_Total_m3'

# Asegurar columnas temporales
if 'hora' not in df.columns:
    df['hora'] = df['timestamp_utc'].dt.hour
if 'dia_semana' not in df.columns:
    df['dia_semana'] = df['timestamp_utc'].dt.dayofweek
if 'mes' not in df.columns:
    df['mes'] = df['timestamp_utc'].dt.month

df['es_fin_semana'] = df['dia_semana'].isin([5, 6])

# Definir estaciones
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
print("1️⃣ VARIACIÓN POR TEMPERATURA")
print("=" * 80)

# Rangos de temperatura
df['temp_rango'] = pd.cut(df['temp'], 
                          bins=[-np.inf, 12, 15, 18, np.inf],
                          labels=['Frío (<12°C)', 'Templado (12-15°C)', 
                                 'Cálido (15-18°C)', 'Caluroso (>18°C)'])

temp_analysis = df.groupby('temp_rango')[volumen_col].agg(['mean', 'std', 'count'])
print("\n📊 Consumo por temperatura:")
print(temp_analysis.to_string())

# Correlación
corr_temp = df[['temp', volumen_col]].corr().iloc[0, 1]
print(f"\n🌡️  Correlación temperatura-consumo: {corr_temp:.4f}")

# Extremos
frio = df[df['temp'] < 12][volumen_col].mean()
calor = df[df['temp'] > 18][volumen_col].mean()
if not np.isnan(frio) and not np.isnan(calor):
    var_temp = ((calor - frio) / frio) * 100
    print(f"   Frío (<12°C): {frio:,.0f} m³/hora")
    print(f"   Calor (>18°C): {calor:,.0f} m³/hora")
    print(f"   Variación: {var_temp:.2f}%")
    print(f"   Diferencia: {abs(calor - frio):,.0f} m³/hora")

print("\n" + "=" * 80)
print("2️⃣ VARIACIÓN POR PRECIPITACIÓN")
print("=" * 80)

df['hay_lluvia'] = df['mmhr'] > 0

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

# Analizar por intensidad de lluvia
df['intensidad_lluvia'] = pd.cut(df['mmhr'], 
                                  bins=[-0.1, 0, 0.5, 2, np.inf],
                                  labels=['Sin lluvia', 'Llovizna (0-0.5mm)', 
                                         'Lluvia (0.5-2mm)', 'Lluvia fuerte (>2mm)'])

lluvia_analysis = df.groupby('intensidad_lluvia')[volumen_col].agg(['mean', 'std', 'count'])
print("\n📊 Consumo por intensidad de lluvia:")
print(lluvia_analysis.to_string())

print("\n" + "=" * 80)
print("3️⃣ VARIACIÓN POR HUMEDAD")
print("=" * 80)

df['humedad_rango'] = pd.cut(df['HR'], 
                             bins=[0, 60, 75, 85, 100],
                             labels=['Baja (<60%)', 'Media (60-75%)', 
                                    'Alta (75-85%)', 'Muy Alta (>85%)'])

humedad_analysis = df.groupby('humedad_rango')[volumen_col].agg(['mean', 'std', 'count'])
print("\n📊 Consumo por humedad:")
print(humedad_analysis.to_string())

corr_hr = df[['HR', volumen_col]].corr().iloc[0, 1]
print(f"\n💧 Correlación humedad-consumo: {corr_hr:.4f}")

print("\n" + "=" * 80)
print("4️⃣ ANÁLISIS POR ESTACIÓN Y CLIMA")
print("=" * 80)

# Promedio por estación
estaciones = df.groupby('estacion')[volumen_col].agg(['mean', 'std', 'count'])
estaciones = estaciones.reindex(['Verano', 'Otoño', 'Invierno', 'Primavera'])
print("\n📊 Consumo por estación:")
print(estaciones.to_string())

# Temperatura promedio por estación
temp_estacion = df.groupby('estacion')['temp'].mean()
print("\n🌡️  Temperatura promedio por estación:")
for est in ['Verano', 'Otoño', 'Invierno', 'Primavera']:
    if est in temp_estacion.index:
        print(f"   {est}: {temp_estacion[est]:.1f}°C")

print("\n" + "=" * 80)
print("5️⃣ PATRONES POR HORA EN DIFERENTES CONDICIONES")
print("=" * 80)

# Hora pico en diferentes condiciones
print("\n⏰ HORA PICO (7:00) en diferentes condiciones:")
hora_pico = df[df['hora'] == 7]

print(f"\n   Global: {hora_pico[volumen_col].mean():,.0f} m³")
print(f"   Verano: {hora_pico[hora_pico['estacion']=='Verano'][volumen_col].mean():,.0f} m³")
print(f"   Invierno: {hora_pico[hora_pico['estacion']=='Invierno'][volumen_col].mean():,.0f} m³")
print(f"   Con calor (>18°C): {hora_pico[hora_pico['temp']>18][volumen_col].mean():,.0f} m³")
print(f"   Con frío (<12°C): {hora_pico[hora_pico['temp']<12][volumen_col].mean():,.0f} m³")
print(f"   Con lluvia: {hora_pico[hora_pico['hay_lluvia']][volumen_col].mean():,.0f} m³")
print(f"   Sin lluvia: {hora_pico[~hora_pico['hay_lluvia']][volumen_col].mean():,.0f} m³")
print(f"   Lunes: {hora_pico[hora_pico['dia_semana']==0][volumen_col].mean():,.0f} m³")
print(f"   Domingo: {hora_pico[hora_pico['dia_semana']==6][volumen_col].mean():,.0f} m³")

print("\n" + "=" * 80)
print("6️⃣ RESUMEN EJECUTIVO")
print("=" * 80)

# Calcular todas las variaciones
var_hora = ((df.groupby('hora')[volumen_col].mean().max() - 
             df.groupby('hora')[volumen_col].mean().min()) / 
            df.groupby('hora')[volumen_col].mean().min() * 100)

var_estacion = ((df.groupby('estacion')[volumen_col].mean().max() - 
                 df.groupby('estacion')[volumen_col].mean().min()) / 
                df.groupby('estacion')[volumen_col].mean().min() * 100)

var_dia_semana = ((df[~df['es_fin_semana']][volumen_col].mean() - 
                   df[df['es_fin_semana']][volumen_col].mean()) / 
                  df[df['es_fin_semana']][volumen_col].mean() * 100)

print(f"\n🎯 MAGNITUD DE VARIACIONES (ordenado por importancia):")
print(f"\n   1. HORA DEL DÍA:        {var_hora:>6.2f}%  ← FACTOR DOMINANTE")
print(f"   2. TEMPERATURA:         {var_temp:>6.2f}%")
if dias_lluvia > 0:
    print(f"   3. PRECIPITACIÓN:       {abs(var_lluvia):>6.2f}%")
print(f"   4. ESTACIÓN:            {var_estacion:>6.2f}%")
print(f"   5. DÍA DE SEMANA:       {abs(var_dia_semana):>6.2f}%")

print(f"\n📈 CORRELACIONES:")
print(f"   • Temperatura → Consumo: {corr_temp:>6.4f}")
print(f"   • Humedad → Consumo:     {corr_hr:>6.4f}")

print("\n✅ CONCLUSIONES:")
print("\n   1. La HORA DEL DÍA domina (ciclos diarios de ~39%)")
print("   2. La TEMPERATURA tiene impacto significativo (~" + f"{abs(var_temp):.1f}%)")
print("   3. La PRECIPITACIÓN afecta el consumo (~" + f"{abs(var_lluvia):.1f}%)" if dias_lluvia > 0 else "")
print("   4. La estacionalidad es moderada (~" + f"{var_estacion:.1f}%)")
print("   5. El día de semana tiene efecto mínimo (~" + f"{abs(var_dia_semana):.1f}%)")

print("\n⚠️  IMPLICACIONES PARA EL MODELO:")
print("   • El modelo DEBE capturar variables climáticas (temp, lluvia)")
print("   • Las predicciones DEBEN variar según condiciones climáticas")
print("   • Si las predicciones no varían, falta información climática")

print("\n" + "=" * 80)
