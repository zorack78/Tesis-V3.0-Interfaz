# -*- coding: utf-8 -*-
"""
Analisis 10: Error por Condiciones Operativas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

print("=" * 80)
print("ANALISIS 10: ERROR POR CONDICIONES OPERATIVAS")
print("=" * 80)

plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path('outputs/inferencial')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar datos y modelo
print("\n[1] Cargando datos...")
modelo = joblib.load('../models/forecasting/modelo_forecasting_xgboost.pkl')
with open('../models/forecasting/features.txt') as f:
    features = [line.strip() for line in f]

df_test = pd.read_csv('../data/processed/data_test.csv')
df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
df_test['hora'] = df_test['timestamp'].dt.hour
df_test['temp'] = df_test['clima_temp_c']  # Crear columna temp para análisis
print(f"   OK - {len(df_test):,} registros")

# 2. Generar predicciones
print("\n[2] Generando predicciones...")
features_disponibles = [f for f in features if f in df_test.columns]
X_test = df_test[features_disponibles].copy()

for feature in features:
    if feature not in features_disponibles:
        X_test[feature] = 0

X_test = X_test[features]
y_test = df_test['Q_net_m3h'].values
y_pred = modelo.predict(X_test)

# Calcular errores
df_test['y_pred'] = y_pred
df_test['error'] = y_test - y_pred
df_test['error_abs'] = np.abs(df_test['error'])
df_test['error_pct'] = (df_test['error'] / y_test) * 100

print(f"   MAE: {df_test['error_abs'].mean():.2f} m³")
print(f"   RMSE: {np.sqrt((df_test['error']**2).mean()):.2f} m³")

# Crear figura con 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Análisis de Error del Modelo por Condiciones Operativas',
             fontsize=16, fontweight='bold', y=1.00)

# ============================================================================
# Subplot 1: Error vs Hora del Dia (boxplot)
# ============================================================================
print("\n[3] Analizando error por hora del dia...")
ax1 = axes[0]

# Crear boxplot por hora
horas_data = [df_test[df_test['hora'] == h]['error'].values for h in range(24)]
bp = ax1.boxplot(horas_data, positions=range(24), widths=0.6, patch_artist=True,
                  showmeans=True, meanline=True)

# Colorear boxplots
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

# Lineas de medias y medianas
for median in bp['medians']:
    median.set_color('red')
    median.set_linewidth(2)

for mean in bp['means']:
    mean.set_color('green')
    mean.set_linewidth(2)
    mean.set_linestyle('--')

ax1.axhline(0, color='black', linestyle='-', linewidth=2)
ax1.set_xlabel('Hora del Día', fontsize=11, fontweight='bold')
ax1.set_ylabel('Error Q_net (m3/hr)', fontsize=11, fontweight='bold')
ax1.set_title('Distribución de Error por Hora del Día', fontsize=12, fontweight='bold')
ax1.set_xticks(range(0, 24, 2))
ax1.grid(True, alpha=0.3, axis='y')

# Calcular error promedio por hora
error_hora = df_test.groupby('hora')['error_abs'].mean()
hora_max_error = error_hora.idxmax()
hora_min_error = error_hora.idxmin()

textstr = f'Mayor error:\n{int(hora_max_error)}:00 hrs\n'
textstr += f'({error_hora.max():.1f} m³)\n\n'
textstr += f'Menor error:\n{int(hora_min_error)}:00 hrs\n'
textstr += f'({error_hora.min():.1f} m³)'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right', bbox=props)

print(f"   Hora con mayor error: {int(hora_max_error)}:00 ({error_hora.max():.1f} m³)")
print(f"   Hora con menor error: {int(hora_min_error)}:00 ({error_hora.min():.1f} m³)")

# ============================================================================
# Subplot 2: Error vs Temperatura (scatter con regresion)
# ============================================================================
print("\n[4] Analizando error vs temperatura...")
ax2 = axes[1]

# Filtrar datos validos de temperatura
df_temp_valid = df_test[['temp', 'error_abs']].dropna()

scatter = ax2.scatter(df_temp_valid['temp'], df_temp_valid['error_abs'],
                      alpha=0.3, s=15, c=df_temp_valid['error_abs'],
                      cmap='YlOrRd', edgecolors='none')

# Regresion polinomial (grado 2)
z = np.polyfit(df_temp_valid['temp'], df_temp_valid['error_abs'], 2)
p = np.poly1d(z)
temp_range = np.linspace(df_temp_valid['temp'].min(), df_temp_valid['temp'].max(), 100)
ax2.plot(temp_range, p(temp_range), 'r--', linewidth=2, 
         label=f'Ajuste Polinomial (grado 2)')

# Promedio por rangos de temperatura
temp_bins = pd.cut(df_temp_valid['temp'], bins=10)
temp_means = df_temp_valid.groupby(temp_bins)['error_abs'].mean()
temp_centers = [(interval.left + interval.right) / 2 for interval in temp_means.index]
ax2.plot(temp_centers, temp_means.values, 'bo-', linewidth=2, markersize=8,
         label='Media por rango', alpha=0.7)

ax2.set_xlabel('Temperatura (C)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Error Absoluto Q_net (m3/hr)', fontsize=11, fontweight='bold')
ax2.set_title('Error Absoluto vs Temperatura', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Error Absoluto', fontsize=10)

# Correlacion
from scipy.stats import pearsonr
corr, pval = pearsonr(df_temp_valid['temp'], df_temp_valid['error_abs'])

textstr = f'Correlación:\n'
textstr += f'r = {corr:.3f}\n'
textstr += f'p = {pval:.3e}\n\n'
if pval < 0.05:
    textstr += 'Significativa'
else:
    textstr += 'No significativa'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

print(f"   Correlacion error-temp: r={corr:.4f}, p={pval:.4e}")

# ============================================================================
# Subplot 3: Error vs Volumen de Estanques
# ============================================================================
print("\n[5] Analizando error vs volumen...")
ax3 = axes[2]

# Filtrar datos validos
df_vol_valid = df_test[['sist_Vtotal_m3', 'error_abs']].dropna()

scatter = ax3.scatter(df_vol_valid['sist_Vtotal_m3'], df_vol_valid['error_abs'],
                      alpha=0.3, s=15, c=df_vol_valid['error_abs'],
                      cmap='YlOrRd', edgecolors='none')

# Regresion lineal
z = np.polyfit(df_vol_valid['sist_Vtotal_m3'], df_vol_valid['error_abs'], 1)
p = np.poly1d(z)
vol_range = np.linspace(df_vol_valid['sist_Vtotal_m3'].min(), 
                        df_vol_valid['sist_Vtotal_m3'].max(), 100)
ax3.plot(vol_range, p(vol_range), 'r--', linewidth=2, 
         label=f'y = {z[0]:.4f}x + {z[1]:.1f}')

# Promedio por rangos de volumen
vol_bins = pd.cut(df_vol_valid['sist_Vtotal_m3'], bins=10)
vol_means = df_vol_valid.groupby(vol_bins)['error_abs'].mean()
vol_centers = [(interval.left + interval.right) / 2 for interval in vol_means.index]
ax3.plot(vol_centers, vol_means.values, 'bo-', linewidth=2, markersize=8,
         label='Media por rango', alpha=0.7)

ax3.set_xlabel('Volumen Total (m3)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Error Absoluto Q_net (m3/hr)', fontsize=11, fontweight='bold')
ax3.set_title('Error Absoluto vs Volumen en Estanques', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Formato eje X
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))

cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Error Absoluto', fontsize=10)

# Correlacion
corr_vol, pval_vol = pearsonr(df_vol_valid['sist_Vtotal_m3'], 
                               df_vol_valid['error_abs'])

textstr = f'Correlación:\n'
textstr += f'r = {corr_vol:.3f}\n'
textstr += f'p = {pval_vol:.3e}\n\n'
if pval_vol < 0.05:
    textstr += 'Significativa'
else:
    textstr += 'No significativa'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

print(f"   Correlacion error-volumen: r={corr_vol:.4f}, p={pval_vol:.4e}")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_error_condiciones.png', dpi=300, bbox_inches='tight')
print(f"\n   OK - Guardado: {OUTPUT_DIR / '02_error_condiciones.png'}")
plt.close()

# 6. Guardar datos
print("\n[6] Guardando analisis de errores...")
error_hora_df = df_test.groupby('hora').agg({
    'error_abs': ['mean', 'std', 'min', 'max'],
    'error': 'count'
}).round(2)
error_hora_df.to_csv(OUTPUT_DIR / '02_error_por_hora.csv')
print(f"   OK - Datos guardados")

print("\n" + "=" * 80)
print("ANALISIS COMPLETADO")
print("=" * 80)
