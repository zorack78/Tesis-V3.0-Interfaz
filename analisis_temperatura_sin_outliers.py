"""
Análisis de Temperatura vs Demanda (SIN OUTLIERS)
Filtrando valores extremos de demanda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

print("="*80)
print("ANÁLISIS: TEMPERATURA vs DEMANDA (SIN OUTLIERS)")
print("="*80)

# Cargar datos
clima_path = Path('data/raw/BD_Clima2024a202509_UTC.csv')
df_clima = pd.read_csv(clima_path)
df_clima['timestamp'] = pd.to_datetime(df_clima['timestamp'])
df_clima.set_index('timestamp', inplace=True)

demanda_path = Path('data/processed/data_processed_demanda_valid.csv')
df_demanda = pd.read_csv(demanda_path)
df_demanda['timestamp_utc'] = pd.to_datetime(df_demanda['timestamp_utc'])
df_demanda.set_index('timestamp_utc', inplace=True)

# Merge
df_merged = df_demanda.join(df_clima[['temp', 'HR', 'mmhr']], how='inner')
df_clean = df_merged.dropna(subset=['Demanda_m3_hr', 'temp'])

print(f"\n📊 Datos iniciales: {len(df_clean)} registros")
print(f"   Demanda min: {df_clean['Demanda_m3_hr'].min():,.0f} m³/hr")
print(f"   Demanda max: {df_clean['Demanda_m3_hr'].max():,.0f} m³/hr")

# FILTRAR OUTLIERS: 0 a 30,000 m³/hr (rango razonable)
print(f"\n🧹 Filtrando outliers...")
outliers_antes = len(df_clean)
df_clean = df_clean[
    (df_clean['Demanda_m3_hr'] >= 0) &
    (df_clean['Demanda_m3_hr'] <= 30000)
].copy()
outliers_removidos = outliers_antes - len(df_clean)

print(f"✅ Outliers removidos: {outliers_removidos} ({outliers_removidos/outliers_antes*100:.2f}%)")
print(f"✅ Datos limpios: {len(df_clean)} registros")
print(f"   Demanda min: {df_clean['Demanda_m3_hr'].min():,.0f} m³/hr")
print(f"   Demanda max: {df_clean['Demanda_m3_hr'].max():,.0f} m³/hr")
print(f"   Demanda media: {df_clean['Demanda_m3_hr'].mean():,.0f} m³/hr")

# Correlación SIN outliers
print("\n" + "="*80)
print("CORRELACIÓN (SIN OUTLIERS)")
print("="*80)

pearson_r, pearson_p = pearsonr(df_clean['temp'], df_clean['Demanda_m3_hr'])
spearman_r, spearman_p = spearmanr(df_clean['temp'], df_clean['Demanda_m3_hr'])

print(f"\n📈 Correlación de Pearson:")
print(f"   r = {pearson_r:.4f}")
print(f"   p-value = {pearson_p:.2e}")

print(f"\n📈 Correlación de Spearman:")
print(f"   ρ = {spearman_r:.4f}")
print(f"   p-value = {spearman_p:.2e}")

if abs(pearson_r) > 0.7:
    interpretacion = "FUERTE"
elif abs(pearson_r) > 0.4:
    interpretacion = "MODERADA"
elif abs(pearson_r) > 0.2:
    interpretacion = "DÉBIL"
else:
    interpretacion = "MUY DÉBIL"

print(f"\n💡 Interpretación: Correlación {interpretacion}")

# Análisis por rangos de temperatura (sin outliers)
print("\n" + "="*80)
print("DEMANDA POR RANGOS DE TEMPERATURA (SIN OUTLIERS)")
print("="*80)

bins = [0, 10, 15, 20, 25, 30, 50]
labels = ['0-10°C', '10-15°C', '15-20°C', '20-25°C', '25-30°C', '>30°C']
df_clean['temp_range'] = pd.cut(df_clean['temp'], bins=bins, labels=labels)

demanda_por_temp = df_clean.groupby('temp_range', observed=False)['Demanda_m3_hr'].agg([
    'count', 'mean', 'std', 'min', 'max'
])

print(f"\n📊 Demanda promedio por rango:")
for idx, row in demanda_por_temp.iterrows():
    if row['count'] > 0:
        print(f"   {idx:12} : {row['mean']:>8,.0f} m³/hr (n={row['count']:>5})")

# Variación
demanda_min_rango = demanda_por_temp['mean'].min()
demanda_max_rango = demanda_por_temp['mean'].max()
variacion_pct = ((demanda_max_rango - demanda_min_rango) / demanda_por_temp['mean'].mean()) * 100

print(f"\n📊 Variación entre rangos:")
print(f"   Mínimo: {demanda_min_rango:,.0f} m³/hr")
print(f"   Máximo: {demanda_max_rango:,.0f} m³/hr")
print(f"   Diferencia: {demanda_max_rango - demanda_min_rango:,.0f} m³/hr")
print(f"   Variación: {variacion_pct:.1f}%")

# Análisis estacional
df_clean['month'] = df_clean.index.month
df_clean['season'] = df_clean['month'].map({
    12: 'Verano', 1: 'Verano', 2: 'Verano',
    3: 'Otoño', 4: 'Otoño', 5: 'Otoño',
    6: 'Invierno', 7: 'Invierno', 8: 'Invierno',
    9: 'Primavera', 10: 'Primavera', 11: 'Primavera'
})

print("\n" + "="*80)
print("ANÁLISIS ESTACIONAL (SIN OUTLIERS)")
print("="*80)

estaciones = df_clean.groupby('season').agg({
    'temp': ['mean', 'min', 'max'],
    'Demanda_m3_hr': ['mean', 'min', 'max', 'count']
})

print(f"\n📅 Por estación:")
for season in ['Verano', 'Otoño', 'Invierno', 'Primavera']:
    if season in estaciones.index:
        temp_mean = estaciones.loc[season, ('temp', 'mean')]
        dem_mean = estaciones.loc[season, ('Demanda_m3_hr', 'mean')]
        count = estaciones.loc[season, ('Demanda_m3_hr', 'count')]
        print(f"   {season:12} : Temp={temp_mean:5.1f}°C  Demanda={dem_mean:8,.0f} m³/hr  (n={count:.0f})")

# Comparar verano vs invierno
verano_dem = estaciones.loc['Verano', ('Demanda_m3_hr', 'mean')]
invierno_dem = estaciones.loc['Invierno', ('Demanda_m3_hr', 'mean')]
dif_estacional = verano_dem - invierno_dem
pct_estacional = (dif_estacional / invierno_dem) * 100

print(f"\n🌡️  Comparación Verano vs Invierno:")
print(f"   Verano:    {verano_dem:,.0f} m³/hr")
print(f"   Invierno:  {invierno_dem:,.0f} m³/hr")
print(f"   Diferencia: {dif_estacional:+,.0f} m³/hr ({pct_estacional:+.1f}%)")

# Recomendación final
print("\n" + "="*80)
print("CONCLUSIÓN FINAL")
print("="*80)

print(f"\n📋 Resumen:")
print(f"   • Correlación: {pearson_r:.4f} ({interpretacion})")
print(f"   • Variación por temperatura: {variacion_pct:.1f}%")
print(f"   • Variación estacional: {pct_estacional:.1f}%")

if abs(pearson_r) > 0.3 or variacion_pct > 10:
    print(f"\n✅ RECOMENDACIÓN: INCLUIR temperatura")
    print(f"   • Mejorará predicciones en rangos de temperatura extremos")
    print(f"   • Capturará mejor el efecto estacional")
else:
    print(f"\n⚠️  Correlación débil, pero INCLUIR de todas formas porque:")
    print(f"   • Hay variación estacional significativa ({pct_estacional:.1f}%)")
    print(f"   • Puede mejorar predicciones en días atípicos")
    print(f"   • Información adicional sin costo significativo")

print("\n" + "="*80)
