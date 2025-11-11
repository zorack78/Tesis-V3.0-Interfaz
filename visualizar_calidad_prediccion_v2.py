"""
Visualización de Calidad de Predicción: Datos Reales vs Predicciones
Análisis gráfico completo del período de testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

print("="*80)
print("VISUALIZACIÓN: REAL VS PREDICCIÓN (PERÍODO TESTING)")
print("="*80)

# 1. Cargar datos completos
print("\n📂 Cargando datos...")
df_all = pd.read_csv('data/processed/data_processed_demanda_valid.csv')
df_all['timestamp_utc'] = pd.to_datetime(df_all['timestamp_utc'])
print(f"✅ Datos cargados: {len(df_all):,} registros")

# 2. Crear features
print("\n🔧 Calculando features...")

# LAGs
df_all['Demanda_lag_1h'] = df_all['Demanda_m3_hr'].shift(1)
df_all['Demanda_lag_2h'] = df_all['Demanda_m3_hr'].shift(2)
df_all['Demanda_lag_24h'] = df_all['Demanda_m3_hr'].shift(24)
df_all['Demanda_lag_168h'] = df_all['Demanda_m3_hr'].shift(168)

# Rolling
df_all['Demanda_rolling_mean_6h'] = df_all['Demanda_m3_hr'].rolling(6).mean()
df_all['Demanda_rolling_std_6h'] = df_all['Demanda_m3_hr'].rolling(6).std()
df_all['Demanda_rolling_mean_24h'] = df_all['Demanda_m3_hr'].rolling(24).mean()
df_all['Demanda_rolling_std_24h'] = df_all['Demanda_m3_hr'].rolling(24).std()

# Diferencias
df_all['Demanda_diff_1h'] = df_all['Demanda_m3_hr'].diff(1)
df_all['Demanda_diff_24h'] = df_all['Demanda_m3_hr'].diff(24)

# Ratio
df_all['Demanda_ratio_vs_24h'] = df_all['Demanda_m3_hr'] / (df_all['Demanda_lag_24h'] + 1)

print(f"✅ Features calculadas")

# 3. Limpiar y hacer split
df_clean = df_all.dropna(subset=[
    'Demanda_m3_hr', 'Demanda_lag_1h', 'Demanda_lag_24h', 
    'Demanda_lag_168h', 'Demanda_rolling_mean_24h'
]).reset_index(drop=True)

train_end = int(len(df_clean) * 0.70)
val_end = int(len(df_clean) * 0.85)

df_test = df_clean.iloc[val_end:].copy()
print(f"\n📊 Período de testing: {len(df_test):,} registros")
print(f"   Desde: {df_test['timestamp_utc'].min()}")
print(f"   Hasta: {df_test['timestamp_utc'].max()}")

# 4. Cargar modelo
print("\n🤖 Cargando modelo...")
with open('models/demanda/demanda_xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/demanda/features.txt', 'r') as f:
    feature_cols = [line.strip() for line in f if line.strip()]

print(f"✅ Modelo cargado con {len(feature_cols)} features")

# 5. Hacer predicciones
X_test = df_test[feature_cols]
y_test = df_test['Demanda_m3_hr']
y_pred = model.predict(X_test)

df_test['Demanda_pred'] = y_pred
df_test['error'] = y_test - y_pred
df_test['error_abs'] = np.abs(df_test['error'])
df_test['error_pct'] = (df_test['error_abs'] / y_test) * 100

# 6. Métricas
print("\n📊 Métricas:")
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(df_test['error_pct'])

print(f"   R²:    {r2:.4f}")
print(f"   RMSE:  {rmse:,.0f} m³/hr")
print(f"   MAE:   {mae:,.0f} m³/hr")
print(f"   MAPE:  {mape:.2f}%")

# 7. Crear visualización
print("\n📈 Generando gráficos...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)

color_real = '#2E86AB'
color_pred = '#A23B72'
color_error = '#F18F01'

# GRÁFICO 1: Serie temporal (primeros 7 días)
ax1 = fig.add_subplot(gs[0, :])
df_7dias = df_test.head(168)

ax1.plot(df_7dias['timestamp_utc'], df_7dias['Demanda_m3_hr'], 
         label='Real', color=color_real, linewidth=2.5, alpha=0.9)
ax1.plot(df_7dias['timestamp_utc'], df_7dias['Demanda_pred'], 
         label='Predicción', color=color_pred, linewidth=2, alpha=0.9, linestyle='--')
ax1.fill_between(df_7dias['timestamp_utc'], 
                  df_7dias['Demanda_m3_hr'], 
                  df_7dias['Demanda_pred'],
                  alpha=0.2, color=color_error, label='Error')

ax1.set_title('📅 Serie Temporal: Primeros 7 Días (Real vs Predicción)', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Fecha y Hora', fontsize=11)
ax1.set_ylabel('Demanda (m³/hr)', fontsize=11)
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# GRÁFICO 2: Scatter Real vs Predicho
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(y_test, y_pred, alpha=0.4, s=15, color=color_pred)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 
         'r--', linewidth=2.5, label='Ideal')
ax2.set_title(f'Real vs Predicho\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
ax2.set_xlabel('Demanda Real (m³/hr)', fontsize=10)
ax2.set_ylabel('Demanda Predicha (m³/hr)', fontsize=10)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# GRÁFICO 3: Distribución de Errores
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(df_test['error'], bins=50, color=color_error, alpha=0.7, edgecolor='black')
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
ax3.axvline(df_test['error'].mean(), color='green', linestyle='--', 
            linewidth=2, label=f"Media = {df_test['error'].mean():.0f}")
ax3.set_title(f'Distribución de Errores\nMAE = {mae:.0f} m³/hr', 
              fontsize=12, fontweight='bold')
ax3.set_xlabel('Error (Real - Predicho) m³/hr', fontsize=10)
ax3.set_ylabel('Frecuencia', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# GRÁFICO 4: Error Porcentual
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(df_test['error_pct'], bins=50, color=color_error, alpha=0.7, edgecolor='black')
ax4.axvline(mape, color='red', linestyle='--', linewidth=2, 
            label=f'MAPE = {mape:.2f}%')
ax4.set_title('Error Porcentual', fontsize=12, fontweight='bold')
ax4.set_xlabel('Error Absoluto (%)', fontsize=10)
ax4.set_ylabel('Frecuencia', fontsize=10)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_xlim(0, 20)

# GRÁFICO 5: Patrón por Hora
ax5 = fig.add_subplot(gs[2, 0])
patron_hora = df_test.groupby('hour').agg({
    'Demanda_m3_hr': 'mean',
    'Demanda_pred': 'mean'
}).reset_index()

ax5.plot(patron_hora['hour'], patron_hora['Demanda_m3_hr'], 
         'o-', label='Real', color=color_real, linewidth=2.5, markersize=7)
ax5.plot(patron_hora['hour'], patron_hora['Demanda_pred'], 
         's-', label='Predicho', color=color_pred, linewidth=2.5, markersize=7)
ax5.set_title('Patrón por Hora del Día', fontsize=12, fontweight='bold')
ax5.set_xlabel('Hora', fontsize=10)
ax5.set_ylabel('Demanda Promedio (m³/hr)', fontsize=10)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_xticks(range(0, 24, 2))

# GRÁFICO 6: Patrón por Día de Semana
ax6 = fig.add_subplot(gs[2, 1])
patron_dia = df_test.groupby('day_of_week').agg({
    'Demanda_m3_hr': 'mean',
    'Demanda_pred': 'mean'
}).reset_index()

dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
x = np.arange(len(dias))
width = 0.35

ax6.bar(x - width/2, patron_dia['Demanda_m3_hr'], width, 
        label='Real', color=color_real, alpha=0.8)
ax6.bar(x + width/2, patron_dia['Demanda_pred'], width, 
        label='Predicho', color=color_pred, alpha=0.8)
ax6.set_title('Patrón por Día de Semana', fontsize=12, fontweight='bold')
ax6.set_xlabel('Día', fontsize=10)
ax6.set_ylabel('Demanda Promedio (m³/hr)', fontsize=10)
ax6.set_xticks(x)
ax6.set_xticklabels(dias)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')

# GRÁFICO 7: Error por Hora
ax7 = fig.add_subplot(gs[2, 2])
error_hora = df_test.groupby('hour')['error_abs'].agg(['mean', 'std']).reset_index()

ax7.bar(error_hora['hour'], error_hora['mean'], 
        color=color_error, alpha=0.7, edgecolor='black')
ax7.errorbar(error_hora['hour'], error_hora['mean'], 
             yerr=error_hora['std'], fmt='none', color='black', 
             capsize=3, alpha=0.5)
ax7.set_title('Error por Hora', fontsize=12, fontweight='bold')
ax7.set_xlabel('Hora', fontsize=10)
ax7.set_ylabel('Error Absoluto (m³/hr)', fontsize=10)
ax7.grid(True, alpha=0.3, axis='y')
ax7.set_xticks(range(0, 24, 2))

# GRÁFICO 8: Residuos
ax8 = fig.add_subplot(gs[3, 0])
ax8.scatter(y_pred, df_test['error'], alpha=0.4, s=15, color=color_error)
ax8.axhline(0, color='red', linestyle='--', linewidth=2)
ax8.set_title('Residuos vs Predicción', fontsize=12, fontweight='bold')
ax8.set_xlabel('Demanda Predicha (m³/hr)', fontsize=10)
ax8.set_ylabel('Residuo (Real - Predicho)', fontsize=10)
ax8.grid(True, alpha=0.3)

# GRÁFICO 9: QQ Plot (normalidad residuos)
from scipy import stats
ax9 = fig.add_subplot(gs[3, 1])
stats.probplot(df_test['error'], dist="norm", plot=ax9)
ax9.set_title('Q-Q Plot (Normalidad)', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3)

# GRÁFICO 10: Últimos 3 días
ax10 = fig.add_subplot(gs[3, 2])
df_3dias = df_test.tail(72)

ax10.plot(df_3dias['timestamp_utc'], df_3dias['Demanda_m3_hr'], 
         'o-', label='Real', color=color_real, linewidth=2, markersize=4)
ax10.plot(df_3dias['timestamp_utc'], df_3dias['Demanda_pred'], 
         's-', label='Predicho', color=color_pred, linewidth=2, markersize=4)
ax10.set_title('Últimos 3 Días', fontsize=12, fontweight='bold')
ax10.set_xlabel('Fecha', fontsize=10)
ax10.set_ylabel('Demanda (m³/hr)', fontsize=10)
ax10.legend(fontsize=9)
ax10.grid(True, alpha=0.3)
plt.setp(ax10.xaxis.get_majorticklabels(), rotation=45)

# GRÁFICO 11: Detalle completo (todo el período)
ax11 = fig.add_subplot(gs[4, :])
ax11.plot(df_test['timestamp_utc'], df_test['Demanda_m3_hr'], 
         label='Real', color=color_real, linewidth=1, alpha=0.7)
ax11.plot(df_test['timestamp_utc'], df_test['Demanda_pred'], 
         label='Predicción', color=color_pred, linewidth=1, alpha=0.7, linestyle='--')
ax11.set_title('📊 Serie Completa del Período de Testing', 
              fontsize=13, fontweight='bold')
ax11.set_xlabel('Fecha', fontsize=11)
ax11.set_ylabel('Demanda (m³/hr)', fontsize=11)
ax11.legend(loc='upper right', fontsize=10)
ax11.grid(True, alpha=0.3)
plt.setp(ax11.xaxis.get_majorticklabels(), rotation=45)

# Título general
fig.suptitle(f'EVALUACIÓN DE CALIDAD - PERÍODO DE TESTING\n'
             f'R² = {r2:.4f} | RMSE = {rmse:,.0f} m³/hr | '
             f'MAE = {mae:,.0f} m³/hr | MAPE = {mape:.2f}%\n'
             f'{len(df_test):,} registros: '
             f'{df_test["timestamp_utc"].min().strftime("%Y-%m-%d")} a '
             f'{df_test["timestamp_utc"].max().strftime("%Y-%m-%d")}',
             fontsize=15, fontweight='bold', y=0.995)

# Guardar
output_dir = Path('outputs/figures')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'evaluacion_calidad_prediccion.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

print(f"\n✅ Gráfico guardado: {output_path}")

# Estadísticas detalladas
print("\n" + "="*80)
print("ESTADÍSTICAS DETALLADAS")
print("="*80)

print("\n📊 Distribución de Errores:")
print(f"   P5:   {np.percentile(df_test['error_abs'], 5):>7,.0f} m³/hr")
print(f"   P25:  {np.percentile(df_test['error_abs'], 25):>7,.0f} m³/hr")
print(f"   P50:  {np.percentile(df_test['error_abs'], 50):>7,.0f} m³/hr")
print(f"   P75:  {np.percentile(df_test['error_abs'], 75):>7,.0f} m³/hr")
print(f"   P95:  {np.percentile(df_test['error_abs'], 95):>7,.0f} m³/hr")

print("\n🎯 Precisión:")
e5 = (df_test['error_pct'] < 5).sum()
e10 = (df_test['error_pct'] < 10).sum()
print(f"   Error < 5%:  {e5:>5,} ({e5/len(df_test)*100:>5.1f}%)")
print(f"   Error < 10%: {e10:>5,} ({e10/len(df_test)*100:>5.1f}%)")

print(f"\n🎯 Calidad: {'EXCELENTE' if r2 > 0.95 else 'MUY BUENA' if r2 > 0.90 else 'BUENA'}")

print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)

plt.show()
