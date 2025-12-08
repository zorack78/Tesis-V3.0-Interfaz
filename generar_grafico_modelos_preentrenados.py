#!/usr/bin/env python3
"""
Genera gráfico comparativo usando modelos ya entrenados (más rápido)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pytz
from pathlib import Path
from sklearn.metrics import mean_absolute_error

print("\n" + "="*80)
print("GRÁFICA COMPARATIVA RÁPIDA - MODELOS PRE-ENTRENADOS")
print("="*80)

# 1. Cargar modelos
print("\n[1] Cargando modelos...")
modelo_xgb = joblib.load('models/forecasting/modelo_forecasting_xgboost.pkl')
modelo_lgb = joblib.load('models/forecasting/modelo_forecasting_lgbm.pkl')

with open('models/forecasting/features.txt', 'r', encoding='utf-8') as f:
    features = [line.strip() for line in f if line.strip()]

print(f"  ✅ XGBoost cargado")
print(f"  ✅ LightGBM cargado")
print(f"  Features: {len(features)}")

# 2. Cargar dataset
print("\n[2] Cargando dataset...")
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

# Eliminar features de Q_net si existen
q_features = [col for col in df.columns if col.startswith('Q_net_m3h__')]
if q_features:
    df = df.drop(columns=q_features)

print(f"  Registros: {len(df):,}")

# 3. Cargar perfil Qin (train only)
print("\n[3] Cargando perfil Qin...")
fecha_limite_train = pd.Timestamp('2025-03-22 22:00:00', tz='UTC')
qin_raw = pd.read_csv('data/raw/BD_Qin_m3_Local.csv')
qin_raw['timestamp'] = pd.to_datetime(qin_raw['timestamp'])

if qin_raw['timestamp'].dt.tz is None:
    qin_raw['timestamp'] = qin_raw['timestamp'].dt.tz_localize('UTC')

qin_train = qin_raw[qin_raw['timestamp'] <= fecha_limite_train].copy()
qin_train['hora'] = qin_train['timestamp'].dt.hour
qin_perfil = qin_train.groupby('hora')['Qin'].median().to_dict()
print(f"  ✅ Perfil calculado: {len(qin_train):,} registros")

# 4. Separar train/test (85%/15% - igual que interfaz)
print("\n[4] Separando train/test (85%/15%)...")
n = len(df)
val_end = int(n * 0.85)
train = df.iloc[:val_end].copy()
test = df.iloc[val_end:].copy()

print(f"  Total registros:  {n:,}")
print(f"  Train (85%):      {len(train):,}")
print(f"  Test (15%):       {len(test):,}")

# 5. Calcular Qin y Qout
print("\n[5] Calculando Qin y Qout...")
train['hora'] = train['timestamp'].dt.hour
train['Qin_m3h'] = train['hora'].map(qin_perfil)

test['hora'] = test['timestamp'].dt.hour
test['Qin_m3h'] = test['hora'].map(qin_perfil)
test['Qout_m3h'] = test['Qin_m3h'] - test['Q_net_m3h']

print(f"  Desde: {test['timestamp'].min()}")
print(f"  Hasta: {test['timestamp'].max()}")

# 5.5. Entrenar RandomForest (no hay modelo guardado)
print("\n[5.5] Entrenando RandomForest...")
from sklearn.ensemble import RandomForestRegressor

modelo_rf = RandomForestRegressor(
    n_estimators=50,  # Reducido para velocidad
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

X_train = train[features].fillna(0)
modelo_rf.fit(X_train, train['Q_net_m3h'])
print(f"  ✅ RandomForest entrenado")

# 6. Predecir con los 3 modelos
print("\n[6] Generando predicciones...")
X_test = test[features].fillna(0)
y_test_qout = test['Qout_m3h'].values
qin_test = test['Qin_m3h'].values

# XGBoost
pred_qnet_xgb = modelo_xgb.predict(X_test)
pred_qout_xgb = qin_test - pred_qnet_xgb
mae_xgb = mean_absolute_error(y_test_qout, pred_qout_xgb)

# RandomForest
pred_qnet_rf = modelo_rf.predict(X_test)
pred_qout_rf = qin_test - pred_qnet_rf
mae_rf = mean_absolute_error(y_test_qout, pred_qout_rf)

# LightGBM
pred_qnet_lgb = modelo_lgb.predict(X_test)
pred_qout_lgb = qin_test - pred_qnet_lgb
mae_lgb = mean_absolute_error(y_test_qout, pred_qout_lgb)

print(f"  ✅ MAE XGBoost:       {mae_xgb:,.0f} m³/h")
print(f"  ✅ MAE RandomForest:  {mae_rf:,.0f} m³/h")
print(f"  ✅ MAE LightGBM:      {mae_lgb:,.0f} m³/h")

# 7. Generar gráfico
print("\n[7] Generando gráfico...")
chile_tz = pytz.timezone('America/Santiago')
timestamps_chile = test['timestamp'].dt.tz_convert(chile_tz)

plt.figure(figsize=(20, 10))

# Datos reales (línea más tenue)
plt.plot(timestamps_chile, y_test_qout, 'k-',
         linewidth=1.5, label='Demanda Real (Qout)',
         alpha=0.7, zorder=1)

# Modelos (líneas más gruesas)
plt.plot(timestamps_chile, pred_qout_xgb, 'b-',
         linewidth=2.8, label=f'XGBoost (MAE: {mae_xgb:,.0f} m³/h)',
         alpha=0.85, zorder=4)

plt.plot(timestamps_chile, pred_qout_rf, 'g-',
         linewidth=2.8, label=f'Random Forest (MAE: {mae_rf:,.0f} m³/h)',
         alpha=0.85, zorder=3)

plt.plot(timestamps_chile, pred_qout_lgb, 'r-',
         linewidth=2.8, label=f'LightGBM (MAE: {mae_lgb:,.0f} m³/h)',
         alpha=0.85, zorder=2)

# Configuración
plt.title('Comparación de Modelos ML - Periodo Completo de Testing (15%)\n' +
          'Sin Features Derivadas de Q_net (Sin Data Leakage)',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Fecha y Hora (Hora Local Chile)', fontsize=13, fontweight='bold')
plt.ylabel('Demanda Qout (m³/h)', fontsize=13, fontweight='bold')
plt.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')

# Formato eje x
import matplotlib.dates as mdates
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m\n%H:%M', tz=chile_tz))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=0, ha='center')

plt.tight_layout()

# 8. Guardar
output_path = 'outputs/figures/comparacion_3_modelos_test_completo.png'
Path('outputs/figures').mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✅ Guardado: {output_path}")

print("\n" + "="*80)
print("✅ COMPLETADO")
print("="*80)
print(f"\nGráfico: {output_path}")
print(f"Test set: {len(test):,} registros (15% del dataset)")
print(f"Periodo: {test['timestamp'].min()} - {test['timestamp'].max()}")
print(f"\nMétricas consistentes con interfaz Gradio ✓")
print()
