# -*- coding: utf-8 -*-
"""
Analisis 9: Residuos del Modelo (Diagnostico)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import joblib

print("=" * 80)
print("ANALISIS 9: RESIDUOS DEL MODELO (DIAGNOSTICO)")
print("=" * 80)

plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path('outputs/inferencial')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar datos de test y modelo
print("\n[1] Cargando modelo y datos...")
modelo = joblib.load('../models/forecasting/modelo_forecasting_xgboost.pkl')
with open('../models/forecasting/features.txt') as f:
    features = [line.strip() for line in f]

df_test = pd.read_csv('../data/processed/data_test.csv')
df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
print(f"   OK - {len(df_test):,} registros de test")

# 2. Preparar features y generar predicciones
print("\n[2] Generando predicciones...")
features_disponibles = [f for f in features if f in df_test.columns]
X_test = df_test[features_disponibles].copy()

# Llenar features faltantes con 0
for feature in features:
    if feature not in features_disponibles:
        X_test[feature] = 0

X_test = X_test[features]
y_test = df_test['Q_net_m3h'].values
y_pred = modelo.predict(X_test)

# Calcular residuos
residuos = y_test - y_pred
print(f"   Media de residuos: {residuos.mean():.2f}")
print(f"   Desv. std de residuos: {residuos.std():.2f}")

# Crear figura con 4 subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ============================================================================
# Subplot 1: Histograma de residuos
# ============================================================================
print("\n[3] Generando histograma de residuos...")
ax1 = fig.add_subplot(gs[0, 0])

# Histograma
n, bins, patches = ax1.hist(residuos, bins=50, alpha=0.7, color='steelblue', 
                             edgecolor='black', density=True)

# Curva normal teorica
mu, sigma = residuos.mean(), residuos.std()
x = np.linspace(residuos.min(), residuos.max(), 100)
ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
         label=f'Normal(mu={mu:.1f}, std={sigma:.1f})')

ax1.axvline(0, color='black', linestyle='--', linewidth=2, label='Cero')
ax1.set_xlabel('Residuos (m³)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Densidad de Probabilidad', fontsize=11, fontweight='bold')
ax1.set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Test de normalidad (Shapiro-Wilk en muestra)
sample_size = min(5000, len(residuos))
sample_residuos = np.random.choice(residuos, size=sample_size, replace=False)
shapiro_stat, shapiro_p = stats.shapiro(sample_residuos)

textstr = f'Test Shapiro-Wilk:\n'
textstr += f'Estadístico: {shapiro_stat:.4f}\n'
textstr += f'p-value: {shapiro_p:.4e}\n'
if shapiro_p > 0.05:
    textstr += 'Los residuos siguen\ndistribución normal'
else:
    textstr += 'Los residuos no siguen\ndistribución normal'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right', bbox=props)

print(f"   Test Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.4e}")

# ============================================================================
# Subplot 2: Q-Q Plot (normalidad)
# ============================================================================
print("\n[4] Generando Q-Q plot...")
ax2 = fig.add_subplot(gs[0, 1])

stats.probplot(residuos, dist="norm", plot=ax2)
ax2.get_lines()[0].set_marker('o')
ax2.get_lines()[0].set_markersize(3)
ax2.get_lines()[0].set_alpha(0.5)
ax2.get_lines()[0].set_color('steelblue')
ax2.get_lines()[1].set_color('red')
ax2.get_lines()[1].set_linewidth(2)

ax2.set_xlabel('Cuantiles Teóricos', fontsize=11, fontweight='bold')
ax2.set_ylabel('Cuantiles Observados', fontsize=11, fontweight='bold')
ax2.set_title('Q-Q Plot: Normalidad de Residuos', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# ============================================================================
# Subplot 3: Residuos vs Predicciones (homocedasticidad)
# ============================================================================
print("\n[5] Analizando homocedasticidad...")
ax3 = fig.add_subplot(gs[1, 0])

scatter = ax3.scatter(y_pred, residuos, alpha=0.3, s=10, c=np.abs(residuos), 
                      cmap='YlOrRd', edgecolors='none')
ax3.axhline(0, color='black', linestyle='--', linewidth=2)

# Lineas de referencia (+/- 2std)
ax3.axhline(2*sigma, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
ax3.axhline(-2*sigma, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

# Calcular promedio movil de residuos
sorted_indices = np.argsort(y_pred)
window_size = 100
ma_residuos = np.convolve(residuos[sorted_indices], 
                          np.ones(window_size)/window_size, mode='valid')
# Asegurar que ambos arrays tengan la misma longitud
ma_pred = y_pred[sorted_indices][window_size//2:window_size//2+len(ma_residuos)]
if len(ma_pred) == len(ma_residuos):
    ax3.plot(ma_pred, ma_residuos, color='blue', linewidth=2, 
             label='Promedio Móvil')

ax3.set_xlabel('Predicciones (m³)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Residuos (m³)', fontsize=11, fontweight='bold')
ax3.set_title('Residuos vs Predicciones (Homocedasticidad)', 
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('|Residuo|', fontsize=10)

# Test de Breusch-Pagan (homocedasticidad)
from scipy.stats import chi2
residuos_sq = residuos ** 2
correlation = np.corrcoef(y_pred, residuos_sq)[0, 1]

textstr = f'Correlación:\n'
textstr += f'Pred vs Residuo²:\n'
textstr += f'r = {correlation:.4f}'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

print(f"   Correlacion pred vs residuo²: {correlation:.4f}")

# ============================================================================
# Subplot 4: Residuos vs Tiempo (independencia)
# ============================================================================
print("\n[6] Analizando independencia temporal...")
ax4 = fig.add_subplot(gs[1, 1])

# Graficar residuos en el tiempo
ax4.plot(df_test['timestamp'], residuos, alpha=0.5, color='steelblue', 
         linewidth=0.5)
ax4.axhline(0, color='black', linestyle='--', linewidth=2)
ax4.axhline(2*sigma, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
            label='+/- 2std')
ax4.axhline(-2*sigma, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

# Promedio movil
df_test['residuos'] = residuos
df_test['res_ma'] = df_test['residuos'].rolling(window=24, center=True).mean()
ax4.plot(df_test['timestamp'], df_test['res_ma'], color='darkblue', 
         linewidth=2, label='Media Móvil 24h')

ax4.set_xlabel('Tiempo', fontsize=11, fontweight='bold')
ax4.set_ylabel('Residuos (m³)', fontsize=11, fontweight='bold')
ax4.set_title('Residuos vs Tiempo (Independencia)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# Test de Durbin-Watson (autocorrelacion)
from scipy.stats import shapiro
residuos_diff = np.diff(residuos)
dw_stat = np.sum(residuos_diff**2) / np.sum(residuos**2)

textstr = f'Durbin-Watson:\n'
textstr += f'DW = {dw_stat:.4f}\n'
if 1.5 < dw_stat < 2.5:
    textstr += 'No autocorrelación\nsignificativa'
else:
    textstr += 'Posible\nautocorrelación'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax4.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

print(f"   Durbin-Watson: {dw_stat:.4f}")

# Titulo principal
fig.suptitle('Análisis de Residuos del Modelo - Diagnóstico de Supuestos',
             fontsize=16, fontweight='bold', y=0.998)

plt.savefig(OUTPUT_DIR / '01_residuos_modelo.png', dpi=300, bbox_inches='tight')
print(f"\n   OK - Guardado: {OUTPUT_DIR / '01_residuos_modelo.png'}")
plt.close()

# 7. Guardar datos de residuos
print("\n[7] Guardando datos de residuos...")
df_residuos = pd.DataFrame({
    'timestamp': df_test['timestamp'],
    'y_real': y_test,
    'y_pred': y_pred,
    'residuos': residuos,
    'residuos_abs': np.abs(residuos),
    'residuos_pct': (residuos / y_test) * 100
})
df_residuos.to_csv(OUTPUT_DIR / '01_residuos_datos.csv', index=False, float_format='%.2f')
print(f"   OK - Datos guardados")

print("\n" + "=" * 80)
print("ANALISIS COMPLETADO")
print("=" * 80)
