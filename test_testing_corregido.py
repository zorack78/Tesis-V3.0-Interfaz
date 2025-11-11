"""
Script de prueba para verificar las correcciones en la evaluación testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*80)
print("PRUEBA: Evaluación Testing - Agosto-Septiembre 2025")
print("="*80)

# Cargar modelo y features
model = pickle.load(open('models/demanda/demanda_xgboost_model.pkl', 'rb'))
features = [line.strip() for line in open('models/demanda/features.txt')]

# Cargar datos
df = pd.read_csv('data/processed/data_processed_demanda_valid.csv')
df.columns = df.columns.str.strip()
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])

# Crear LAGs
df['Demanda_lag_1h'] = df['Demanda_m3_hr'].shift(1)
df['Demanda_lag_2h'] = df['Demanda_m3_hr'].shift(2)
df['Demanda_lag_24h'] = df['Demanda_m3_hr'].shift(24)
df['Demanda_lag_168h'] = df['Demanda_m3_hr'].shift(168)
df['Demanda_rolling_mean_6h'] = df['Demanda_m3_hr'].rolling(6, min_periods=1).mean()
df['Demanda_rolling_std_6h'] = df['Demanda_m3_hr'].rolling(6, min_periods=1).std()
df['Demanda_rolling_mean_24h'] = df['Demanda_m3_hr'].rolling(24, min_periods=1).mean()
df['Demanda_rolling_std_24h'] = df['Demanda_m3_hr'].rolling(24, min_periods=1).std()
df['Demanda_diff_1h'] = df['Demanda_m3_hr'].diff()
df['Demanda_diff_24h'] = df['Demanda_m3_hr'].diff(24)
df['Demanda_ratio_vs_24h'] = df['Demanda_m3_hr'] / (df['Demanda_lag_24h'] + 1)

# Limpiar NaN
df_clean = df.dropna(subset=['Demanda_m3_hr', 'Demanda_lag_1h', 
                               'Demanda_lag_24h', 'Demanda_lag_168h']).reset_index(drop=True)

print(f"\n✅ Datos cargados y procesados: {len(df_clean)} registros")

# FILTRAR AGOSTO-SEPTIEMBRE 2025
df_test = df_clean[
    (df_clean['timestamp_utc'] >= '2025-08-01') &
    (df_clean['timestamp_utc'] <= '2025-09-30')
].copy()

print(f"\n📅 Período Agosto-Septiembre 2025:")
print(f"   Inicio: {df_test['timestamp_utc'].min()}")
print(f"   Fin: {df_test['timestamp_utc'].max()}")
print(f"   Registros: {len(df_test)}")

# FILTRAR OUTLIERS
print(f"\n🧹 Filtrando outliers...")
print(f"   Demanda antes del filtro:")
print(f"     Min: {df_test['Demanda_m3_hr'].min():,.0f} m³/hr")
print(f"     Max: {df_test['Demanda_m3_hr'].max():,.0f} m³/hr")
negativos = (df_test['Demanda_m3_hr'] < 0).sum()
print(f"     Negativos: {negativos} ({negativos/len(df_test)*100:.2f}%)")

df_test = df_test[
    (df_test['Demanda_m3_hr'] >= 0) &
    (df_test['Demanda_m3_hr'] <= 30000)
].reset_index(drop=True)

print(f"\n   Demanda después del filtro:")
print(f"     Min: {df_test['Demanda_m3_hr'].min():,.0f} m³/hr")
print(f"     Max: {df_test['Demanda_m3_hr'].max():,.0f} m³/hr")
print(f"     Registros: {len(df_test)}")

# Predecir y calcular métricas
X_test = df_test[features]
y_test = df_test['Demanda_m3_hr']
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\n" + "="*80)
print("MÉTRICAS DE CALIDAD")
print("="*80)
print(f"\n📊 Resultados:")
calidad_r2 = '✅ EXCELENTE' if r2 > 0.95 else '⚠️ BUENO' if r2 > 0.90 else '❌ MEJORABLE'
calidad_mape = '✅ EXCELENTE' if mape < 5 else '⚠️ BUENO' if mape < 10 else '❌ MEJORABLE'

print(f"   R²: {r2:.4f} {calidad_r2}")
print(f"   RMSE: {rmse:,.0f} m³/hr")
print(f"   MAE: {mae:,.0f} m³/hr")
print(f"   MAPE: {mape:.2f}% {calidad_mape}")

print(f"\n✅ PRUEBA COMPLETADA")
print("="*80)
print("\n💡 Cambios aplicados:")
print("   ✅ Período: Solo Agosto-Septiembre 2025")
print("   ✅ Outliers: Filtrados (0-30,000 m³/hr)")
print("   ✅ Fechas: Incluidas para eje X")
print("="*80)
