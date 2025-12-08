# -*- coding: utf-8 -*-
"""
Analisis 7: Relacion Temperatura vs Demanda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr

print("=" * 80)
print("ANALISIS 7: TEMPERATURA VS DEMANDA")
print("=" * 80)

plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path('outputs/descriptivo')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar datos
print("\n[1] Cargando datos...")
df = pd.read_csv('../data/processed/data_train.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hora'] = df['timestamp'].dt.hour
df['mes'] = df['timestamp'].dt.month
print(f"   OK - Cargado: {len(df):,} registros")

# Calcular demanda (Qout = Qin - Q_net)
df['Demanda_m3h'] = df['sist_Qin_m3h'] - df['Q_net_m3h']
print(f"   Demanda calculada: media={df['Demanda_m3h'].mean():.1f} m3/hr")

# Filtrar datos validos
df_valid = df[['Demanda_m3h', 'clima_temp_c', 'hora', 'mes']].dropna()
df_valid = df_valid.rename(columns={'clima_temp_c': 'temp'})
print(f"   Registros validos: {len(df_valid):,}")

# Crear figura con 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Analisis de Relacion: Temperatura vs Demanda de Agua',
             fontsize=16, fontweight='bold', y=1.00)

# ============================================================================
# Subplot 1: Scatter plot Temperatura vs Demanda (color por hora)
# ============================================================================
print("\n[2] Generando scatter plot...")
ax1 = axes[0]

scatter = ax1.scatter(df_valid['temp'], df_valid['Demanda_m3h'], 
                      c=df_valid['hora'], cmap='twilight', 
                      alpha=0.3, s=10, edgecolors='none')

# Regresion lineal
z = np.polyfit(df_valid['temp'], df_valid['Demanda_m3h'], 1)
p = np.poly1d(z)
temp_range = np.linspace(df_valid['temp'].min(), df_valid['temp'].max(), 100)
ax1.plot(temp_range, p(temp_range), 'r--', linewidth=2, 
         label=f'y = {z[0]:.1f}x + {z[1]:.1f}')

# Calcular correlacion
corr, pval = pearsonr(df_valid['temp'], df_valid['Demanda_m3h'])
ax1.text(0.05, 0.95, f'r = {corr:.3f}\np < 0.001', 
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax1.set_xlabel('Temperatura (C)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Demanda (m3/hr)', fontsize=11, fontweight='bold')
ax1.set_title('Correlacion Temperatura - Demanda', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Colorbar para hora del dia
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Hora del Día', fontsize=10)

print(f"   Correlacion: r = {corr:.4f}, p-value = {pval:.4e}")

# ============================================================================
# Subplot 2: Demanda promedio por rango de temperatura
# ============================================================================
print("\n[3] Analizando por rangos de temperatura...")
ax2 = axes[1]

# Definir rangos de temperatura
bins = [df_valid['temp'].min(), 10, 15, 20, 25, df_valid['temp'].max()]
labels = ['<10C\n(Frio)', '10-15C\n(Fresco)', '15-20C\n(Templado)', 
          '20-25C\n(Calido)', '>25C\n(Caluroso)']
df_valid['temp_range'] = pd.cut(df_valid['temp'], bins=bins, labels=labels, include_lowest=True)

# Calcular estadisticas por rango
demanda_temp = df_valid.groupby('temp_range', observed=True)['Demanda_m3h'].agg(['mean', 'std', 'count']).reset_index()

# Grafico de barras
colors = ['blue', 'lightblue', 'green', 'orange', 'red']
bars = ax2.bar(range(len(demanda_temp)), demanda_temp['mean'], 
               color=colors[:len(demanda_temp)], alpha=0.7, edgecolor='black')
ax2.errorbar(range(len(demanda_temp)), demanda_temp['mean'], 
             yerr=demanda_temp['std'], fmt='none', color='black', capsize=5)

ax2.set_xticks(range(len(demanda_temp)))
ax2.set_xticklabels(demanda_temp['temp_range'], fontsize=9)
ax2.set_xlabel('Rango de Temperatura', fontsize=11, fontweight='bold')
ax2.set_ylabel('Demanda (m3/hr)', fontsize=11, fontweight='bold')
ax2.set_title('Demanda Promedio por Rango de Temperatura', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Agregar conteo de registros en cada barra
for i, (bar, count) in enumerate(zip(bars, demanda_temp['count'])):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
             f'n={count:,}', ha='center', va='bottom', fontsize=8)

print(f"   Rangos de temperatura analizados: {len(demanda_temp)}")
for _, row in demanda_temp.iterrows():
    print(f"     {row['temp_range']}: {row['mean']:.1f} +/- {row['std']:.1f} m3/hr (n={row['count']:,})")

# ============================================================================
# Subplot 3: Demanda horaria por estacion del año
# ============================================================================
print("\n[4] Analizando por estacion del año...")
ax3 = axes[2]

# Definir estaciones (Chile - hemisferio sur)
def get_estacion(mes):
    if mes in [12, 1, 2]:
        return 'Verano'
    elif mes in [3, 4, 5]:
        return 'Otoño'
    elif mes in [6, 7, 8]:
        return 'Invierno'
    else:
        return 'Primavera'

df_valid['estacion'] = df_valid['mes'].apply(get_estacion)

# Calcular demanda promedio por hora y estacion
demanda_estacion = df_valid.groupby(['hora', 'estacion'])['Demanda_m3h'].mean().reset_index()

# Graficar cada estacion
estaciones = ['Verano', 'Otoño', 'Invierno', 'Primavera']
colores_estacion = {'Verano': 'red', 'Otoño': 'orange', 
                    'Invierno': 'blue', 'Primavera': 'green'}

for estacion in estaciones:
    data = demanda_estacion[demanda_estacion['estacion'] == estacion]
    ax3.plot(data['hora'], data['Demanda_m3h'], 
             marker='o', linewidth=2, markersize=4,
             color=colores_estacion[estacion], label=estacion, alpha=0.8)

ax3.set_xlabel('Hora del Dia', fontsize=11, fontweight='bold')
ax3.set_ylabel('Demanda (m3/hr)', fontsize=11, fontweight='bold')
ax3.set_title('Patron Horario de Demanda por Estacion', fontsize=12, fontweight='bold')
ax3.set_xticks(range(0, 24, 2))
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10, loc='upper left')

print(f"   Estaciones analizadas: {len(estaciones)}")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_temperatura_demanda.png', dpi=300, bbox_inches='tight')
print(f"\n   OK - Guardado: {OUTPUT_DIR / '03_temperatura_demanda.png'}")
plt.close()

# 5. Guardar datos
print("\n[5] Guardando datos...")
demanda_temp.to_csv(OUTPUT_DIR / '03_demanda_por_temp.csv', index=False, float_format='%.2f')
demanda_estacion.to_csv(OUTPUT_DIR / '03_demanda_por_estacion.csv', index=False, float_format='%.2f')
print(f"   OK - Datos guardados")

print("\n" + "=" * 80)
print("ANALISIS COMPLETADO")
print("=" * 80)
