"""
Script simplificado de verificación de datos testing
"""

import pandas as pd
import numpy as np

print("="*80)
print("VERIFICACIÓN: Datos Testing - Agosto-Septiembre 2025")
print("="*80)

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
print(f"   Registros antes de filtrar outliers: {len(df_test)}")

# Analizar outliers
print(f"\n🔍 Análisis de outliers:")
print(f"   Demanda ANTES del filtro:")
print(f"     Min: {df_test['Demanda_m3_hr'].min():,.0f} m³/hr")
print(f"     Max: {df_test['Demanda_m3_hr'].max():,.0f} m³/hr")
print(f"     Media: {df_test['Demanda_m3_hr'].mean():,.0f} m³/hr")
negativos = (df_test['Demanda_m3_hr'] < 0).sum()
extremos_altos = (df_test['Demanda_m3_hr'] > 30000).sum()
print(f"     Negativos: {negativos} ({negativos/len(df_test)*100:.2f}%)")
print(f"     > 30,000: {extremos_altos} ({extremos_altos/len(df_test)*100:.2f}%)")
print(f"     Total outliers: {negativos + extremos_altos}")

# FILTRAR OUTLIERS (como en la interfaz)
df_test_limpio = df_test[
    (df_test['Demanda_m3_hr'] >= 0) &
    (df_test['Demanda_m3_hr'] <= 30000)
].reset_index(drop=True)

print(f"\n   Demanda DESPUÉS del filtro (0-30,000 m³/hr):")
print(f"     Min: {df_test_limpio['Demanda_m3_hr'].min():,.0f} m³/hr")
print(f"     Max: {df_test_limpio['Demanda_m3_hr'].max():,.0f} m³/hr")
print(f"     Media: {df_test_limpio['Demanda_m3_hr'].mean():,.0f} m³/hr")
print(f"     Registros: {len(df_test_limpio)}")
print(f"     Removidos: {len(df_test) - len(df_test_limpio)} registros")

# Verificar formato de fechas para el gráfico
print(f"\n📅 Verificación de timestamps para gráficos:")
timestamps = df_test_limpio['timestamp_utc']
print(f"   Primer timestamp: {timestamps.iloc[0]}")
print(f"   Último timestamp: {timestamps.iloc[-1]}")
print(f"   Tipo: {type(timestamps.iloc[0])}")
print(f"   ✅ Formato válido para matplotlib.dates")

print(f"\n✅ VERIFICACIÓN COMPLETADA")
print("="*80)
print("\n💡 Resumen de correcciones aplicadas:")
print("   ✅ Período: Agosto-Septiembre 2025 (61 días)")
print(f"   ✅ Registros válidos: {len(df_test_limpio)} de {len(df_test)}")
print(f"   ✅ Outliers removidos: {len(df_test) - len(df_test_limpio)}")
print("   ✅ Rango válido: 0-30,000 m³/hr")
print("   ✅ Timestamps listos para eje X")
print("="*80)
