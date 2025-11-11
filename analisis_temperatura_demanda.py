"""
Análisis de relación entre Temperatura y Demanda de Agua

Objetivo: Verificar si incluir temperatura mejora el modelo predictivo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

print("="*80)
print("ANÁLISIS: TEMPERATURA vs DEMANDA")
print("="*80)

# 1. Cargar datos
print("\n📂 Cargando datos...")

# Clima
clima_path = Path('data/raw/BD_Clima2024a202509_UTC.csv')
df_clima = pd.read_csv(clima_path)
df_clima['timestamp'] = pd.to_datetime(df_clima['timestamp'])
df_clima.set_index('timestamp', inplace=True)
print(f"✅ Clima cargado: {len(df_clima)} registros")
print(f"   Columnas: {df_clima.columns.tolist()}")

# Demanda
demanda_path = Path('data/processed/data_processed_demanda_valid.csv')
df_demanda = pd.read_csv(demanda_path)
df_demanda['timestamp_utc'] = pd.to_datetime(df_demanda['timestamp_utc'])
df_demanda.set_index('timestamp_utc', inplace=True)
print(f"✅ Demanda cargada: {len(df_demanda)} registros")

# 2. Merge de datos
print("\n🔗 Combinando temperatura y demanda...")
df_merged = df_demanda.join(df_clima[['temp', 'HR', 'mmhr']], how='inner')
print(f"✅ Datos combinados: {len(df_merged)} registros")

# Verificar NaNs
print(f"\n🔍 Valores faltantes:")
print(df_merged[['Demanda_m3_hr', 'temp', 'HR', 'mmhr']].isnull().sum())

# Limpiar NaNs
df_clean = df_merged.dropna(subset=['Demanda_m3_hr', 'temp'])
print(f"✅ Datos limpios: {len(df_clean)} registros")

# 3. Estadísticas descriptivas
print("\n" + "="*80)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*80)

print(f"\n📊 Temperatura:")
print(f"   Min:     {df_clean['temp'].min():.1f}°C")
print(f"   Max:     {df_clean['temp'].max():.1f}°C")
print(f"   Media:   {df_clean['temp'].mean():.1f}°C")
print(f"   Mediana: {df_clean['temp'].median():.1f}°C")
print(f"   Std:     {df_clean['temp'].std():.1f}°C")

print(f"\n💧 Demanda:")
print(f"   Min:     {df_clean['Demanda_m3_hr'].min():,.0f} m³/hr")
print(f"   Max:     {df_clean['Demanda_m3_hr'].max():,.0f} m³/hr")
print(f"   Media:   {df_clean['Demanda_m3_hr'].mean():,.0f} m³/hr")
print(f"   Mediana: {df_clean['Demanda_m3_hr'].median():,.0f} m³/hr")
print(f"   Std:     {df_clean['Demanda_m3_hr'].std():,.0f} m³/hr")

# 4. Correlación
print("\n" + "="*80)
print("CORRELACIÓN TEMPERATURA-DEMANDA")
print("="*80)

# Pearson (lineal)
pearson_r, pearson_p = pearsonr(df_clean['temp'], df_clean['Demanda_m3_hr'])
print(f"\n📈 Correlación de Pearson (lineal):")
print(f"   r = {pearson_r:.4f}")
print(f"   p-value = {pearson_p:.2e}")
print(f"   Significancia: {'✅ SÍ' if pearson_p < 0.05 else '❌ NO'} (α=0.05)")

# Spearman (monotónica)
spearman_r, spearman_p = spearmanr(df_clean['temp'], df_clean['Demanda_m3_hr'])
print(f"\n📈 Correlación de Spearman (monotónica):")
print(f"   ρ = {spearman_r:.4f}")
print(f"   p-value = {spearman_p:.2e}")
print(f"   Significancia: {'✅ SÍ' if spearman_p < 0.05 else '❌ NO'} (α=0.05)")

# Interpretación
if abs(pearson_r) > 0.7:
    interpretacion = "FUERTE"
elif abs(pearson_r) > 0.4:
    interpretacion = "MODERADA"
elif abs(pearson_r) > 0.2:
    interpretacion = "DÉBIL"
else:
    interpretacion = "MUY DÉBIL o NULA"

print(f"\n💡 Interpretación: Correlación {interpretacion}")
if pearson_r > 0:
    print(f"   → Relación POSITIVA: ↑ Temperatura → ↑ Demanda")
else:
    print(f"   → Relación NEGATIVA: ↑ Temperatura → ↓ Demanda")

# 5. Análisis por rangos de temperatura
print("\n" + "="*80)
print("DEMANDA POR RANGOS DE TEMPERATURA")
print("="*80)

# Crear bins de temperatura
bins = [0, 10, 15, 20, 25, 30, 50]
labels = ['0-10°C', '10-15°C', '15-20°C', '20-25°C', '25-30°C', '>30°C']
df_clean['temp_range'] = pd.cut(df_clean['temp'], bins=bins, labels=labels)

demanda_por_temp = df_clean.groupby('temp_range')['Demanda_m3_hr'].agg([
    'count', 'mean', 'std', 'min', 'max'
])

print(f"\n📊 Demanda promedio por rango de temperatura:")
print(demanda_por_temp.to_string())

# 6. Análisis estacional
print("\n" + "="*80)
print("ANÁLISIS ESTACIONAL (Temperatura y Demanda)")
print("="*80)

df_clean['month'] = df_clean.index.month
df_clean['season'] = df_clean['month'].map({
    12: 'Verano', 1: 'Verano', 2: 'Verano',
    3: 'Otoño', 4: 'Otoño', 5: 'Otoño',
    6: 'Invierno', 7: 'Invierno', 8: 'Invierno',
    9: 'Primavera', 10: 'Primavera', 11: 'Primavera'
})

estaciones = df_clean.groupby('season').agg({
    'temp': ['mean', 'min', 'max'],
    'Demanda_m3_hr': ['mean', 'min', 'max']
})

print(f"\n📅 Estadísticas por estación:")
print(estaciones.to_string())

# 7. Crear gráficos
print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Scatter plot Temperatura vs Demanda
ax1 = axes[0, 0]
ax1.scatter(df_clean['temp'], df_clean['Demanda_m3_hr'], 
           alpha=0.3, s=10, color='#2E86AB')
ax1.set_xlabel('Temperatura (°C)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Demanda (m³/hr)', fontsize=12, fontweight='bold')
ax1.set_title(f'Temperatura vs Demanda\n(r={pearson_r:.3f})', 
             fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Línea de tendencia
z = np.polyfit(df_clean['temp'], df_clean['Demanda_m3_hr'], 1)
p = np.poly1d(z)
temp_sorted = np.sort(df_clean['temp'])
ax1.plot(temp_sorted, p(temp_sorted), "r--", linewidth=2, alpha=0.8, 
        label=f'Tendencia: y={z[0]:.1f}x+{z[1]:.1f}')
ax1.legend()

# 2. Boxplot Demanda por Rango de Temperatura
ax2 = axes[0, 1]
df_clean.boxplot(column='Demanda_m3_hr', by='temp_range', ax=ax2)
ax2.set_xlabel('Rango de Temperatura', fontsize=12, fontweight='bold')
ax2.set_ylabel('Demanda (m³/hr)', fontsize=12, fontweight='bold')
ax2.set_title('Distribución de Demanda por Rango de Temperatura', 
             fontsize=14, fontweight='bold')
plt.sca(ax2)
plt.xticks(rotation=45)

# 3. Serie temporal: Temperatura y Demanda (muestra)
ax3 = axes[1, 0]
# Tomar una muestra de 30 días
sample = df_clean['2024-07-01':'2024-07-30']
ax3_twin = ax3.twinx()

ax3.plot(sample.index, sample['Demanda_m3_hr'], 
        color='#2E86AB', linewidth=1.5, label='Demanda')
ax3_twin.plot(sample.index, sample['temp'], 
             color='#E63946', linewidth=1.5, label='Temperatura')

ax3.set_xlabel('Fecha', fontsize=12, fontweight='bold')
ax3.set_ylabel('Demanda (m³/hr)', fontsize=12, fontweight='bold', color='#2E86AB')
ax3_twin.set_ylabel('Temperatura (°C)', fontsize=12, fontweight='bold', color='#E63946')
ax3.set_title('Serie Temporal: Demanda y Temperatura (Jul 2024)', 
             fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# 4. Demanda promedio por estación
ax4 = axes[1, 1]
season_order = ['Verano', 'Otoño', 'Invierno', 'Primavera']
season_data = df_clean.groupby('season')['Demanda_m3_hr'].mean().reindex(season_order)
colors = ['#E63946', '#F4A261', '#2A9D8F', '#06D6A0']

bars = ax4.bar(season_order, season_data.values, color=colors, alpha=0.8, 
              edgecolor='black', linewidth=2)
ax4.set_ylabel('Demanda Promedio (m³/hr)', fontsize=12, fontweight='bold')
ax4.set_title('Demanda Promedio por Estación', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Agregar valores sobre barras
for bar, val in zip(bars, season_data.values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:,.0f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()

# Guardar figura
output_path = Path('outputs/analisis_temperatura_demanda.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Gráfico guardado: {output_path}")

# 8. Resumen y recomendaciones
print("\n" + "="*80)
print("CONCLUSIONES Y RECOMENDACIONES")
print("="*80)

print(f"\n📋 Resumen:")
print(f"   • Correlación Pearson: {pearson_r:.4f} ({interpretacion})")
print(f"   • Significancia estadística: {'✅ SÍ' if pearson_p < 0.05 else '❌ NO'}")
print(f"   • Rango de temperatura: {df_clean['temp'].min():.1f}°C a {df_clean['temp'].max():.1f}°C")
print(f"   • Variación demanda por temperatura: {((demanda_por_temp['mean'].max() - demanda_por_temp['mean'].min()) / demanda_por_temp['mean'].mean() * 100):.1f}%")

if abs(pearson_r) > 0.3 and pearson_p < 0.05:
    print(f"\n✅ RECOMENDACIÓN: INCLUIR temperatura en el modelo")
    print(f"   Razones:")
    print(f"   • Correlación estadísticamente significativa")
    print(f"   • Relación {'positiva' if pearson_r > 0 else 'negativa'} moderada/fuerte")
    print(f"   • Puede mejorar predicciones, especialmente en días extremos")
else:
    print(f"\n⚠️  PRECAUCIÓN: Correlación débil")
    print(f"   • Incluir temperatura puede no mejorar significativamente el modelo")
    print(f"   • Considerar otras variables meteorológicas (HR, precipitación)")
    print(f"   • El modelo actual con features temporales puede ser suficiente")

print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)
