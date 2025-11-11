"""
Visualización de Calidad de Predicción: Datos Reales vs Predicciones
Análisis gráfico completo del período de testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

print("="*80)
print("VISUALIZACIÓN: DATOS REALES VS PREDICCIONES (PERÍODO DE TESTING)")
print("="*80)

# 1. Cargar datos completos y generar split de test
print("\n📂 Cargando datos completos...")
df_all = pd.read_csv('data/processed/data_processed_demanda_valid.csv')
df_all['timestamp_utc'] = pd.to_datetime(df_all['timestamp_utc'])
print(f"✅ Datos cargados: {len(df_all):,} registros")

# 2. Crear features necesarias para el modelo
print("\n🔧 Calculando features...")

# Features temporales básicas (ya existen en data_test.csv)
# hour, day_of_week, month, is_weekend

# Features cíclicas (ya existen)
# hour_sin, hour_cos, day_of_week_sin, day_of_week_cos

# Features de eventos (ya existen)
# feriado, temporada_turistica_alta

# LAGs de Demanda
df_test['Demanda_lag_1h'] = df_test['Demanda_m3_hr'].shift(1)
df_test['Demanda_lag_2h'] = df_test['Demanda_m3_hr'].shift(2)
df_test['Demanda_lag_24h'] = df_test['Demanda_m3_hr'].shift(24)
df_test['Demanda_lag_168h'] = df_test['Demanda_m3_hr'].shift(168)

# Rolling statistics
df_test['Demanda_rolling_mean_6h'] = df_test['Demanda_m3_hr'].rolling(window=6, min_periods=1).mean()
df_test['Demanda_rolling_std_6h'] = df_test['Demanda_m3_hr'].rolling(window=6, min_periods=1).std()
df_test['Demanda_rolling_mean_24h'] = df_test['Demanda_m3_hr'].rolling(window=24, min_periods=1).mean()
df_test['Demanda_rolling_std_24h'] = df_test['Demanda_m3_hr'].rolling(window=24, min_periods=1).std()

# Diferencias
df_test['Demanda_diff_1h'] = df_test['Demanda_m3_hr'].diff(1)
df_test['Demanda_diff_24h'] = df_test['Demanda_m3_hr'].diff(24)

# Ratio
df_test['Demanda_ratio_vs_24h'] = df_test['Demanda_m3_hr'] / (df_test['Demanda_lag_24h'] + 1)

# Limpiar registros con NaN en features críticas
df_test_clean = df_test.dropna(subset=[
    'Demanda_m3_hr',
    'Demanda_lag_1h', 'Demanda_lag_24h', 'Demanda_lag_168h',
    'Demanda_rolling_mean_24h', 'Demanda_rolling_std_24h'
]).reset_index(drop=True)

print(f"✅ Features calculadas")
print(f"   Registros después de limpiar NaN: {len(df_test_clean):,} (perdidos: {len(df_test) - len(df_test_clean)})")

# Usar df_test_clean de aquí en adelante
df_test = df_test_clean

# 3. Cargar modelo y hacer predicciones
print("\n🤖 Cargando modelo...")
with open('models/demanda/demanda_xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/demanda/features.txt', 'r') as f:
    feature_cols = [line.strip() for line in f if line.strip()]

print(f"✅ Modelo cargado con {len(feature_cols)} features")

# Preparar datos para predicción
X_test = df_test[feature_cols]
y_test = df_test['Demanda_m3_hr']
y_pred = model.predict(X_test)

# Agregar predicciones al dataframe
df_test['Demanda_pred'] = y_pred
df_test['error'] = y_test - y_pred
df_test['error_abs'] = np.abs(df_test['error'])
df_test['error_pct'] = (df_test['error_abs'] / y_test) * 100

# 3. Calcular métricas
print("\n📊 Métricas de evaluación:")
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(df_test['error_pct'])

print(f"   R² Score:    {r2:.4f}")
print(f"   RMSE:        {rmse:,.0f} m³/hr")
print(f"   MAE:         {mae:,.0f} m³/hr")
print(f"   MAPE:        {mape:.2f}%")

# 4. Crear visualización completa
print("\n📈 Generando gráficos...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# Color palette
color_real = '#2E86AB'
color_pred = '#A23B72'
color_error = '#F18F01'

# ============================================================================
# GRÁFICO 1: Serie temporal completa (primeros 7 días)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])
df_subset = df_test.head(168)  # Primeros 7 días (24h x 7)

ax1.plot(df_subset['timestamp_utc'], df_subset['Demanda_m3_hr'], 
         label='Demanda Real', color=color_real, linewidth=2, alpha=0.8)
ax1.plot(df_subset['timestamp_utc'], df_subset['Demanda_pred'], 
         label='Demanda Predicha', color=color_pred, linewidth=2, alpha=0.8, linestyle='--')
ax1.fill_between(df_subset['timestamp_utc'], 
                  df_subset['Demanda_m3_hr'], 
                  df_subset['Demanda_pred'],
                  alpha=0.2, color=color_error, label='Error')

ax1.set_title('📅 Serie Temporal: Primeros 7 Días del Período de Testing\n(Comparación Real vs Predicción)', 
              fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel('Fecha y Hora', fontsize=11)
ax1.set_ylabel('Demanda (m³/hr)', fontsize=11)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# ============================================================================
# GRÁFICO 2: Scatter Plot Real vs Predicho
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])

ax2.scatter(y_test, y_pred, alpha=0.5, s=10, color=color_pred)
# Línea ideal (y=x)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 
         'r--', linewidth=2, label='Predicción perfecta')

ax2.set_title(f'📊 Real vs Predicho\n(R² = {r2:.4f})', 
              fontsize=12, fontweight='bold')
ax2.set_xlabel('Demanda Real (m³/hr)', fontsize=10)
ax2.set_ylabel('Demanda Predicha (m³/hr)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ============================================================================
# GRÁFICO 3: Distribución de Errores
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])

ax3.hist(df_test['error'], bins=50, color=color_error, alpha=0.7, edgecolor='black')
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
ax3.axvline(df_test['error'].mean(), color='green', linestyle='--', 
            linewidth=2, label=f'Media = {df_test["error"].mean():.0f}')

ax3.set_title(f'📉 Distribución de Errores\n(MAE = {mae:.0f} m³/hr)', 
              fontsize=12, fontweight='bold')
ax3.set_xlabel('Error (Real - Predicho) m³/hr', fontsize=10)
ax3.set_ylabel('Frecuencia', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# ============================================================================
# GRÁFICO 4: Error Absoluto Porcentual
# ============================================================================
ax4 = fig.add_subplot(gs[1, 2])

ax4.hist(df_test['error_pct'], bins=50, color=color_error, alpha=0.7, edgecolor='black')
ax4.axvline(mape, color='red', linestyle='--', linewidth=2, 
            label=f'MAPE = {mape:.2f}%')

ax4.set_title('📊 Error Porcentual Absoluto', fontsize=12, fontweight='bold')
ax4.set_xlabel('Error Absoluto (%)', fontsize=10)
ax4.set_ylabel('Frecuencia', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_xlim(0, 20)  # Limitar a 20% para mejor visualización

# ============================================================================
# GRÁFICO 5: Patrón por Hora del Día
# ============================================================================
ax5 = fig.add_subplot(gs[2, 0])

patron_hora = df_test.groupby('hour').agg({
    'Demanda_m3_hr': 'mean',
    'Demanda_pred': 'mean'
}).reset_index()

ax5.plot(patron_hora['hour'], patron_hora['Demanda_m3_hr'], 
         'o-', label='Real', color=color_real, linewidth=2, markersize=6)
ax5.plot(patron_hora['hour'], patron_hora['Demanda_pred'], 
         's-', label='Predicho', color=color_pred, linewidth=2, markersize=6)

ax5.set_title('⏰ Patrón Promedio por Hora del Día', fontsize=12, fontweight='bold')
ax5.set_xlabel('Hora del Día', fontsize=10)
ax5.set_ylabel('Demanda Promedio (m³/hr)', fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_xticks(range(0, 24, 2))

# Marcar horarios de inflexión
hora_max_real = patron_hora.loc[patron_hora['Demanda_m3_hr'].idxmax(), 'hour']
hora_min_real = patron_hora.loc[patron_hora['Demanda_m3_hr'].idxmin(), 'hour']
ax5.axvline(hora_max_real, color='green', linestyle='--', alpha=0.5, 
            label=f'Máx Real: {hora_max_real}:00')
ax5.axvline(hora_min_real, color='orange', linestyle='--', alpha=0.5, 
            label=f'Mín Real: {hora_min_real}:00')

# ============================================================================
# GRÁFICO 6: Patrón por Día de la Semana
# ============================================================================
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

ax6.set_title('📅 Patrón por Día de la Semana', fontsize=12, fontweight='bold')
ax6.set_xlabel('Día de la Semana', fontsize=10)
ax6.set_ylabel('Demanda Promedio (m³/hr)', fontsize=10)
ax6.set_xticks(x)
ax6.set_xticklabels(dias)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

# ============================================================================
# GRÁFICO 7: Error por Hora del Día
# ============================================================================
ax7 = fig.add_subplot(gs[2, 2])

error_por_hora = df_test.groupby('hour')['error_abs'].agg(['mean', 'std']).reset_index()

ax7.bar(error_por_hora['hour'], error_por_hora['mean'], 
        color=color_error, alpha=0.7, edgecolor='black')
ax7.errorbar(error_por_hora['hour'], error_por_hora['mean'], 
             yerr=error_por_hora['std'], fmt='none', color='black', 
             capsize=3, alpha=0.5)

ax7.set_title('⚠️ Error Absoluto por Hora', fontsize=12, fontweight='bold')
ax7.set_xlabel('Hora del Día', fontsize=10)
ax7.set_ylabel('Error Absoluto Promedio (m³/hr)', fontsize=10)
ax7.grid(True, alpha=0.3, axis='y')
ax7.set_xticks(range(0, 24, 2))

# ============================================================================
# GRÁFICO 8: Residuos vs Valores Predichos
# ============================================================================
ax8 = fig.add_subplot(gs[3, 0])

ax8.scatter(y_pred, df_test['error'], alpha=0.5, s=10, color=color_error)
ax8.axhline(0, color='red', linestyle='--', linewidth=2)

ax8.set_title('📊 Residuos vs Valores Predichos', fontsize=12, fontweight='bold')
ax8.set_xlabel('Demanda Predicha (m³/hr)', fontsize=10)
ax8.set_ylabel('Residuo (Real - Predicho)', fontsize=10)
ax8.grid(True, alpha=0.3)

# ============================================================================
# GRÁFICO 9: Últimos 3 días (detalle fino)
# ============================================================================
ax9 = fig.add_subplot(gs[3, 1:])

df_ultimos = df_test.tail(72)  # Últimos 3 días

ax9.plot(df_ultimos['timestamp_utc'], df_ultimos['Demanda_m3_hr'], 
         'o-', label='Demanda Real', color=color_real, linewidth=2, 
         markersize=4, alpha=0.8)
ax9.plot(df_ultimos['timestamp_utc'], df_ultimos['Demanda_pred'], 
         's-', label='Demanda Predicha', color=color_pred, linewidth=2, 
         markersize=4, alpha=0.8)

# Sombrear áreas de error
for i in range(len(df_ultimos)):
    color_area = 'green' if abs(df_ultimos.iloc[i]['error_pct']) < 5 else 'orange'
    if abs(df_ultimos.iloc[i]['error_pct']) > 10:
        color_area = 'red'
    ax9.fill_between([df_ultimos.iloc[i]['timestamp_utc']], 
                      [df_ultimos.iloc[i]['Demanda_m3_hr']], 
                      [df_ultimos.iloc[i]['Demanda_pred']],
                      alpha=0.3, color=color_area)

ax9.set_title('🔍 Detalle: Últimos 3 Días del Período de Testing\n(Verde: error<5%, Naranja: error 5-10%, Rojo: error>10%)', 
              fontsize=12, fontweight='bold')
ax9.set_xlabel('Fecha y Hora', fontsize=10)
ax9.set_ylabel('Demanda (m³/hr)', fontsize=10)
ax9.legend(loc='upper right', fontsize=9)
ax9.grid(True, alpha=0.3)
ax9.tick_params(axis='x', rotation=45)

# ============================================================================
# Título general y información
# ============================================================================
fig.suptitle(f'EVALUACIÓN DE CALIDAD DEL MODELO - PERÍODO DE TESTING\n'
             f'R² = {r2:.4f} | RMSE = {rmse:,.0f} m³/hr | MAE = {mae:,.0f} m³/hr | MAPE = {mape:.2f}%\n'
             f'Período: {df_test["timestamp_utc"].min().strftime("%Y-%m-%d")} a {df_test["timestamp_utc"].max().strftime("%Y-%m-%d")} ({len(df_test):,} registros)',
             fontsize=16, fontweight='bold', y=0.98)

# ============================================================================
# Guardar figura
# ============================================================================
output_dir = Path('outputs/figures')
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / 'evaluacion_calidad_prediccion.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Gráfico guardado: {output_path}")

# ============================================================================
# Estadísticas adicionales
# ============================================================================
print("\n" + "="*80)
print("ESTADÍSTICAS DETALLADAS")
print("="*80)

print("\n📊 Distribución de Errores:")
print(f"   Percentil 5%:  {np.percentile(df_test['error_abs'], 5):>8,.0f} m³/hr")
print(f"   Percentil 25%: {np.percentile(df_test['error_abs'], 25):>8,.0f} m³/hr")
print(f"   Mediana:       {np.percentile(df_test['error_abs'], 50):>8,.0f} m³/hr")
print(f"   Percentil 75%: {np.percentile(df_test['error_abs'], 75):>8,.0f} m³/hr")
print(f"   Percentil 95%: {np.percentile(df_test['error_abs'], 95):>8,.0f} m³/hr")

print("\n🎯 Precisión por Rangos:")
errores_5pct = (df_test['error_pct'] < 5).sum()
errores_10pct = (df_test['error_pct'] < 10).sum()
errores_15pct = (df_test['error_pct'] < 15).sum()

print(f"   Error < 5%:  {errores_5pct:>6,} registros ({errores_5pct/len(df_test)*100:>5.1f}%)")
print(f"   Error < 10%: {errores_10pct:>6,} registros ({errores_10pct/len(df_test)*100:>5.1f}%)")
print(f"   Error < 15%: {errores_15pct:>6,} registros ({errores_15pct/len(df_test)*100:>5.1f}%)")

print("\n⏰ Horarios de Inflexión (Test Set):")
hora_max_real = patron_hora.loc[patron_hora['Demanda_m3_hr'].idxmax(), 'hour']
hora_max_pred = patron_hora.loc[patron_hora['Demanda_pred'].idxmax(), 'hour']
hora_min_real = patron_hora.loc[patron_hora['Demanda_m3_hr'].idxmin(), 'hour']
hora_min_pred = patron_hora.loc[patron_hora['Demanda_pred'].idxmin(), 'hour']

print(f"   Máximo Real:     {int(hora_max_real):02d}:00 → {patron_hora.loc[patron_hora['Demanda_m3_hr'].idxmax(), 'Demanda_m3_hr']:,.0f} m³/hr")
print(f"   Máximo Predicho: {int(hora_max_pred):02d}:00 → {patron_hora.loc[patron_hora['Demanda_pred'].idxmax(), 'Demanda_pred']:,.0f} m³/hr")
print(f"   {'✅ MATCH' if hora_max_real == hora_max_pred else f'⚠️  DIFERENCIA: {abs(hora_max_real - hora_max_pred):.0f} horas'}")
print()
print(f"   Mínimo Real:     {int(hora_min_real):02d}:00 → {patron_hora.loc[patron_hora['Demanda_m3_hr'].idxmin(), 'Demanda_m3_hr']:,.0f} m³/hr")
print(f"   Mínimo Predicho: {int(hora_min_pred):02d}:00 → {patron_hora.loc[patron_hora['Demanda_pred'].idxmin(), 'Demanda_pred']:,.0f} m³/hr")
print(f"   {'✅ MATCH' if hora_min_real == hora_min_pred else f'⚠️  DIFERENCIA: {abs(hora_min_real - hora_min_pred):.0f} horas'}")

print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)
print(f"\n📁 Gráfico guardado en: {output_path}")
print(f"📊 Visualización completa con 9 gráficos comparativos")
print(f"🎯 Calidad del modelo: {'EXCELENTE' if r2 > 0.95 else 'MUY BUENA' if r2 > 0.90 else 'BUENA' if r2 > 0.85 else 'ACEPTABLE'}")

plt.show()
