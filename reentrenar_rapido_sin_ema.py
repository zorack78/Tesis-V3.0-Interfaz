"""
Script rápido para reentrenar modelo SIN features EMA contaminadas
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

print("\n" + "="*80)
print("REENTRENAMIENTO RÁPIDO - SIN FEATURES EMA")
print("="*80)

# 1. Cargar dataset completo
print("\n[1] Cargando dataset...")
df = pd.read_csv('data/processed/dataset_features_completo.csv')
print(f"  Total registros: {len(df):,}")

# 2. Eliminar TODAS las features derivadas de Q_net (EMAs, LAGs, diffs)
print("\n[2] Eliminando features derivadas de Q_net...")
q_features = [col for col in df.columns if col.startswith('Q_net_m3h__')]
if q_features:
    print(f"  Eliminando {len(q_features)} features: {q_features}")
    df = df.drop(columns=q_features)
else:
    print("  ✅ No se encontraron features derivadas de Q_net")

# 3. Separar train/test
print("\n[3] Separando train/test...")
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
fecha_corte = pd.Timestamp('2025-03-23 23:59:59', tz='UTC')

train = df[df['timestamp'] <= fecha_corte].copy()
test = df[df['timestamp'] > fecha_corte].copy()

print(f"  Train: {len(train):,} registros hasta {train['timestamp'].max()}")
print(f"  Test:  {len(test):,} registros desde {test['timestamp'].min()}")

# 4. Preparar X, y
print("\n[4] Preparando features...")
target = 'Q_net_m3h'
excluir = ['timestamp', target]

# Seleccionar solo columnas numéricas
features = [col for col in train.columns if col not in excluir]
features = [col for col in features if train[col].dtype in ['int64', 'float64', 'int32', 'float32']]

X_train = train[features].fillna(0)
y_train = train[target]
X_test = test[features].fillna(0)
y_test = test[target]

print(f"  Features totales: {len(features)}")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")

# 5. Entrenar XGBoost
print("\n[5] Entrenando XGBoost...")
modelo = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

modelo.fit(X_train, y_train, verbose=False)
print("  ✅ Entrenamiento completado")

# 6. Evaluar
print("\n[6] Evaluando modelo...")
y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print(f"  Train MAE: {mae_train:,.1f} m³/h")
print(f"  Test MAE:  {mae_test:,.1f} m³/h")
print(f"  Test RMSE: {rmse_test:,.1f} m³/h")
print(f"  Test R²:   {r2_test:.4f}")

# 7. Guardar modelo
print("\n[7] Guardando modelo...")
modelo_path = Path('models/forecasting')
modelo_path.mkdir(parents=True, exist_ok=True)

joblib.dump(modelo, modelo_path / 'modelo_forecasting_xgboost.pkl')
print(f"  ✅ Modelo guardado")

# 8. Guardar features
with open(modelo_path / 'features.txt', 'w') as f:
    f.write('\n'.join(features))
print(f"  ✅ Features guardadas ({len(features)} features)")

# 9. Guardar métricas
metricas = {
    'mae_train': float(mae_train),
    'mae_test': float(mae_test),
    'rmse_test': float(rmse_test),
    'r2': float(r2_test),
    'n_features': len(features),
    'modelo': 'XGBoost sin EMAs',
    'nota': 'Reentrenado sin features EMA para eliminar data leakage'
}

with open(modelo_path / 'metricas.json', 'w') as f:
    json.dump(metricas, f, indent=2)
print(f"  ✅ Métricas guardadas")

# 10. Verificar Sep 25, 2025
print("\n[8] Verificando predicción Sep 25, 2025 08:00...")
fecha_problema = pd.Timestamp('2025-09-25 08:00:00', tz='UTC')
mask = test['timestamp'] == fecha_problema
if mask.any():
    idx = test[mask].index[0]
    real = test.loc[idx, target]
    # Buscar índice en y_pred_test
    test_loc = test.index.tolist().index(idx)
    pred = y_pred_test[test_loc]
    print(f"  Valor real:      {real:,.1f} m³/h")
    print(f"  Predicción:      {pred:,.1f} m³/h")
    print(f"  Error absoluto:  {abs(real - pred):,.1f} m³/h")
else:
    print("  ⚠️ Fecha no encontrada en test set")

print("\n" + "="*80)
print("✅ REENTRENAMIENTO COMPLETADO")
print("="*80)
print("\nResumen:")
print(f"  • Features EMA eliminadas")
print(f"  • Modelo reentrenado con {len(features)} features")
print(f"  • MAE test: {mae_test:,.1f} m³/h")
print(f"  • R² test: {r2_test:.4f}")
print("="*80)
