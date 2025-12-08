"""
Script rápido para generar gráfica comparativa de modelos
usando modelos ya entrenados en la interfaz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from datetime import timedelta
import pytz

print("\n" + "="*80)
print("GRÁFICA COMPARATIVA - 3 MODELOS ML (RÁPIDO)")
print("="*80)

# 1. Cargar modelo XGBoost
print("\n[1] Cargando modelo XGBoost...")
model_xgb = joblib.load('models/forecasting/modelo_forecasting_xgboost.pkl')
with open('models/forecasting/features.txt', 'r', encoding='utf-8') as f:
    features = [line.strip() for line in f if line.strip()]
print(f"  ✅ XGBoost cargado ({len(features)} features)")

# 2. Cargar dataset
print("\n[2] Cargando dataset...")
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

# Eliminar features de Q_net si existen
q_features = [col for col in df.columns if col.startswith('Q_net_m3h__')]
if q_features:
    df = df.drop(columns=q_features)

# 3. Cargar perfil Qin (train only)
print("\n[3] Cargando perfil Qin...")
fecha_limite_train = pd.Timestamp('2025-03-22 22:00:00', tz='UTC')
qin_raw = pd.read_csv('data/raw/BD_Qin_m3_Local.csv')
qin_raw['timestamp'] = pd.to_datetime(qin_raw['timestamp'])

# Hacer timezone aware
if qin_raw['timestamp'].dt.tz is None:
    qin_raw['timestamp'] = qin_raw['timestamp'].dt.tz_localize('UTC')

qin_train = qin_raw[qin_raw['timestamp'] <= fecha_limite_train].copy()
qin_train['hora'] = qin_train['timestamp'].dt.hour
qin_perfil = qin_train.groupby('hora')['Qin'].median().to_dict()
print(f"  ✅ Perfil calculado: {len(qin_train):,} registros")

# 4. Separar train/test usando mismo método que interfaz (85%/15%)
print("\n[4] Separando train/test (85%/15% - igual que interfaz)...")
n = len(df)
val_end = int(n * 0.85)

train = df.iloc[:val_end].copy()
test = df.iloc[val_end:].copy()

print(f"  Total registros:  {n:,}")
print(f"  Train (85%):      {len(train):,}")
print(f"  Test (15%):       {len(test):,}")

# 5. Calcular Qin y Qout para train y test
print("\n[5] Calculando Qin y Qout...")
train['hora'] = train['timestamp'].dt.hour
train['Qin_m3h'] = train['hora'].map(qin_perfil)

test['hora'] = test['timestamp'].dt.hour
test['Qin_m3h'] = test['hora'].map(qin_perfil)
test['Qout_m3h'] = test['Qin_m3h'] - test['Q_net_m3h']

# 6. Usar TODO el test set (igual que interfaz)
print("\n[6] Usando test set completo (igual que interfaz)...")
test_semana = test.copy()

print(f"  Desde: {test_semana['timestamp'].min()}")
print(f"  Hasta: {test_semana['timestamp'].max()}")
print(f"  Registros: {len(test_semana):,}")

# Predecir con XGBoost
print("\n[7] Generando predicciones XGBoost...")
X_test = test_semana[features].fillna(0)
y_test_qout = test_semana['Qout_m3h'].values
qin_test = test_semana['Qin_m3h'].values

pred_qnet_xgb = model_xgb.predict(X_test)
pred_qout_xgb = qin_test - pred_qnet_xgb

from sklearn.metrics import mean_absolute_error
mae_xgb = mean_absolute_error(y_test_qout, pred_qout_xgb)
print(f"  ✅ MAE XGBoost: {mae_xgb:,.0f} m³/h")

# 8. Entrenar modelos RF y LightGBM reales (igual que interfaz)
print("\n[8] Entrenando modelos RF y LightGBM...")
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# RandomForest
print("  Entrenando Random Forest...")
modelo_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
X_train = train[features].fillna(0)
modelo_rf.fit(X_train, train['Q_net_m3h'])
pred_qnet_rf = modelo_rf.predict(X_test)
pred_qout_rf = qin_test - pred_qnet_rf
mae_rf = mean_absolute_error(y_test_qout, pred_qout_rf)

# LightGBM
print("  Entrenando LightGBM...")
modelo_lgb = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=12,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1,
    force_col_wise=True
)
modelo_lgb.fit(X_train, train['Q_net_m3h'])
pred_qnet_lgb = modelo_lgb.predict(X_test)
pred_qout_lgb = qin_test - pred_qnet_lgb
mae_lgb = mean_absolute_error(y_test_qout, pred_qout_lgb)

print(f"  ✅ MAE RandomForest (entrenado): {mae_rf:,.0f} m³/h")
print(f"  ✅ MAE LightGBM (entrenado): {mae_lgb:,.0f} m³/h")

# 8. Convertir a hora Chile
chile_tz = pytz.timezone('America/Santiago')
timestamps_chile = test_semana['timestamp'].dt.tz_convert(chile_tz)

# 9. Crear gráfico
print("\n[8] Generando gráfico...")
plt.figure(figsize=(16, 9))

# Real (más tenue, más delgada)
plt.plot(timestamps_chile, y_test_qout, 'k-', 
         linewidth=1.5, label='Real', alpha=0.4, zorder=1)

# Predicciones (más gruesas, más opacas)
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
plt.title('Comparación de Modelos ML - Periodo Completo de Testing\n' +
          'Sin Features Derivadas de Q_net (Sin Data Leakage)',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Fecha y Hora (Hora Local Chile)', fontsize=13, fontweight='bold')
plt.ylabel('Demanda Qout (m³/h)', fontsize=13, fontweight='bold')
plt.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')

# Formato eje x
import matplotlib.dates as mdates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))
plt.xticks(rotation=0, ha='center')

plt.tight_layout()

# 10. Guardar
output_path = 'outputs/figures/comparacion_3_modelos_test_completo.png'
Path('outputs/figures').mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✅ Guardado: {output_path}")

print("\n" + "="*80)
print("✅ COMPLETADO")
print("="*80)
print(f"\nGráfico: {output_path}")
print(f"Valores todos positivos (Demanda Qout)")
print(f"Línea real más tenue, modelos más gruesos")
print()
