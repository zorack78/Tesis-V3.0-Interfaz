# -*- coding: utf-8 -*-
"""
Analisis 12: Intervalos de Confianza de Predicciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from scipy import stats

print("=" * 80)
print("ANALISIS 12: INTERVALOS DE CONFIANZA")
print("=" * 80)

plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path('outputs/inferencial')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar modelo y datos
print("\n[1] Cargando modelo y datos...")
modelo = joblib.load('../models/forecasting/modelo_forecasting_xgboost.pkl')
with open('../models/forecasting/features.txt') as f:
    features = [line.strip() for line in f]

df_test = pd.read_csv('../data/processed/data_test.csv')
df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
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

# Calcular residuos para estimar desviacion estandar
residuos = y_test - y_pred
sigma = np.std(residuos)
print(f"   Desviacion estandar de residuos: {sigma:.2f} m³")

# 3. Calcular intervalos de confianza (95%)
print("\n[3] Calculando intervalos de confianza (95%)...")
z_score = 1.96  # Para 95% de confianza
margen_error = z_score * sigma

y_pred_lower = y_pred - margen_error
y_pred_upper = y_pred + margen_error

# Verificar cobertura
dentro_intervalo = (y_test >= y_pred_lower) & (y_test <= y_pred_upper)
cobertura = dentro_intervalo.sum() / len(y_test) * 100
print(f"   Cobertura del intervalo: {cobertura:.2f}%")

# 4. Seleccionar periodo representativo (7 dias)
print("\n[4] Seleccionando periodo de 7 dias...")
# Tomar primeros 7 dias del test set
inicio = df_test['timestamp'].min()
fin = inicio + pd.Timedelta(days=7)
mask = (df_test['timestamp'] >= inicio) & (df_test['timestamp'] < fin)

df_periodo = df_test[mask].copy()
y_periodo = y_test[mask]
y_pred_periodo = y_pred[mask]
y_lower_periodo = y_pred_lower[mask]
y_upper_periodo = y_pred_upper[mask]

print(f"   Periodo: {df_periodo['timestamp'].min()} a {df_periodo['timestamp'].max()}")
print(f"   Registros: {len(df_periodo)}")

# 5. Crear visualizacion
fig, axes = plt.subplots(2, 1, figsize=(18, 10))
fig.suptitle('Predicciones con Intervalos de Confianza (95%)',
             fontsize=16, fontweight='bold', y=0.995)

# ============================================================================
# Subplot 1: Serie temporal con intervalos
# ============================================================================
print("\n[5] Generando grafico de serie temporal...")
ax1 = axes[0]

# Banda de confianza
ax1.fill_between(df_periodo['timestamp'], y_lower_periodo, y_upper_periodo,
                  alpha=0.3, color='lightblue', label='Intervalo de Confianza 95%')

# Prediccion
ax1.plot(df_periodo['timestamp'], y_pred_periodo, 
         color='blue', linewidth=2, label='Predicción', marker='o', markersize=3)

# Valores reales
ax1.plot(df_periodo['timestamp'], y_periodo, 
         color='red', linewidth=1.5, label='Valor Real', marker='x', markersize=4,
         linestyle='--', alpha=0.8)

ax1.set_xlabel('Fecha y Hora', fontsize=11, fontweight='bold')
ax1.set_ylabel('Q_net (m3/hr)\nPos=Exceso | Neg=Deficit', fontsize=11, fontweight='bold')
ax1.set_title('Predicciones con Intervalo de Confianza - Periodo de 7 Días',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Formato eje Y (sin dividir por 1000, ya que Q_net está en escala menor)
# ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

# Estadisticas en texto
dentro_periodo = ((y_periodo >= y_lower_periodo) & 
                  (y_periodo <= y_upper_periodo)).sum()
cobertura_periodo = dentro_periodo / len(y_periodo) * 100

textstr = f'ESTADÍSTICAS:\n'
textstr += f'Cobertura: {cobertura_periodo:.1f}%\n'
textstr += f'({dentro_periodo}/{len(y_periodo)} puntos)\n\n'
textstr += f'Ancho intervalo:\n'
textstr += f'+/-{margen_error:.0f} m3/hr\n\n'
textstr += f'Std residuos:\n'
textstr += f'{sigma:.0f} m3/hr'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props)

# ============================================================================
# Subplot 2: Analisis detallado del intervalo
# ============================================================================
print("\n[6] Generando analisis detallado...")
ax2 = axes[1]

# Crear multiple subplots en ax2
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, figure=fig, left=0.08, right=0.95, bottom=0.08, top=0.43, wspace=0.3)

# Panel 2.1: Distribucion de errores vs intervalo
ax2_1 = fig.add_subplot(gs[0, 0])

errores = y_periodo - y_pred_periodo

ax2_1.hist(errores, bins=30, alpha=0.7, color='steelblue', edgecolor='black', density=True)

# Curva normal del intervalo
x = np.linspace(-3*sigma, 3*sigma, 100)
ax2_1.plot(x, stats.norm.pdf(x, 0, sigma), 'r-', linewidth=2, 
           label=f'N(0, {sigma:.0f})')

# Limites del intervalo
ax2_1.axvline(-margen_error, color='green', linestyle='--', linewidth=2, 
              label='Límites IC 95%')
ax2_1.axvline(margen_error, color='green', linestyle='--', linewidth=2)
ax2_1.axvline(0, color='black', linestyle='-', linewidth=1)

ax2_1.set_xlabel('Error (m³)', fontsize=10, fontweight='bold')
ax2_1.set_ylabel('Densidad', fontsize=10, fontweight='bold')
ax2_1.set_title('Distribución de Errores\nvs Intervalo de Confianza', 
                fontsize=11, fontweight='bold')
ax2_1.legend(fontsize=8)
ax2_1.grid(True, alpha=0.3)

# Panel 2.2: Ancho relativo del intervalo
ax2_2 = fig.add_subplot(gs[0, 1])

ancho_relativo = (margen_error * 2 / y_pred_periodo) * 100

ax2_2.plot(df_periodo['timestamp'], ancho_relativo, 
           color='purple', linewidth=1.5, marker='o', markersize=3)
ax2_2.axhline(ancho_relativo.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Media: {ancho_relativo.mean():.1f}%')

ax2_2.set_xlabel('Fecha y Hora', fontsize=10, fontweight='bold')
ax2_2.set_ylabel('Ancho del IC (%)', fontsize=10, fontweight='bold')
ax2_2.set_title('Ancho Relativo del Intervalo\n(% de la predicción)', 
                fontsize=11, fontweight='bold')
ax2_2.legend(fontsize=8)
ax2_2.grid(True, alpha=0.3)
ax2_2.tick_params(axis='x', rotation=45)

print(f"   Ancho promedio del intervalo: {ancho_relativo.mean():.2f}%")

# Panel 2.3: Cobertura por segmentos temporales
ax2_3 = fig.add_subplot(gs[0, 2])

df_periodo['hora'] = df_periodo['timestamp'].dt.hour
df_periodo['dentro_ic'] = dentro_periodo

cobertura_hora = df_periodo.groupby('hora')['dentro_ic'].agg(['sum', 'count'])
cobertura_hora['pct'] = (cobertura_hora['sum'] / cobertura_hora['count']) * 100

bars = ax2_3.bar(cobertura_hora.index, cobertura_hora['pct'], 
                  color='lightgreen', alpha=0.7, edgecolor='black')
ax2_3.axhline(95, color='red', linestyle='--', linewidth=2, 
              label='Objetivo: 95%')
ax2_3.axhline(cobertura_hora['pct'].mean(), color='blue', linestyle='-', 
              linewidth=2, label=f'Media: {cobertura_hora["pct"].mean():.1f}%')

# Colorear barras según cobertura
for i, (bar, pct) in enumerate(zip(bars, cobertura_hora['pct'])):
    if pct < 90:
        bar.set_color('red')
        bar.set_alpha(0.7)
    elif pct < 95:
        bar.set_color('yellow')
        bar.set_alpha(0.7)

ax2_3.set_xlabel('Hora del Día', fontsize=10, fontweight='bold')
ax2_3.set_ylabel('Cobertura (%)', fontsize=10, fontweight='bold')
ax2_3.set_title('Cobertura del IC\npor Hora del Día', fontsize=11, fontweight='bold')
ax2_3.set_ylim([0, 110])
ax2_3.legend(fontsize=8)
ax2_3.grid(True, alpha=0.3, axis='y')

plt.savefig(OUTPUT_DIR / '04_intervalos_confianza.png', dpi=300, bbox_inches='tight')
print(f"\n   OK - Guardado: {OUTPUT_DIR / '04_intervalos_confianza.png'}")
plt.close()

# 7. Guardar datos
print("\n[7] Guardando datos...")
df_resultados = pd.DataFrame({
    'timestamp': df_periodo['timestamp'],
    'valor_real': y_periodo,
    'prediccion': y_pred_periodo,
    'ic_lower': y_lower_periodo,
    'ic_upper': y_upper_periodo,
    'dentro_ic': dentro_periodo,
    'ancho_ic_m3': margen_error * 2,
    'ancho_ic_pct': ancho_relativo
})
df_resultados.to_csv(OUTPUT_DIR / '04_intervalos_datos.csv', index=False, float_format='%.2f')

# Resumen estadistico
resumen = {
    'confianza': 0.95,
    'z_score': z_score,
    'sigma_residuos': float(sigma),
    'margen_error_m3': float(margen_error),
    'cobertura_total_pct': float(cobertura),
    'cobertura_periodo_pct': float(cobertura_periodo),
    'ancho_promedio_pct': float(ancho_relativo.mean()),
    'registros_evaluados': int(len(y_test)),
    'registros_periodo': int(len(y_periodo))
}

import json
with open(OUTPUT_DIR / '04_intervalos_resumen.json', 'w') as f:
    json.dump(resumen, f, indent=2)

print(f"   OK - Datos guardados")

print("\n" + "=" * 80)
print("RESUMEN DE INTERVALOS DE CONFIANZA")
print("=" * 80)
print(f"Nivel de confianza: 95%")
print(f"Cobertura real: {cobertura:.2f}%")
print(f"Ancho del intervalo: +/-{margen_error:.0f} m3/hr")
print(f"Ancho relativo promedio: {ancho_relativo.mean():.2f}%")
print("=" * 80)
print("ANALISIS COMPLETADO")
print("=" * 80)
