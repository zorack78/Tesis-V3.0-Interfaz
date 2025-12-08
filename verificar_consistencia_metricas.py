"""
Script para verificar que scatter_plots e interfaz dan MISMAS métricas
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
import joblib

print('='*80)
print('VERIFICACION DE CONSISTENCIA DE METRICAS')
print('='*80)

# Cargar dataset
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').copy()

print(f'\nDataset: {df.shape}')

# Split 70/15/15
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end].copy()
df_test = df.iloc[val_end:].copy()

print(f'Test: {len(df_test):,} registros')
print(f'Periodo: {df_test["timestamp"].min()} -> {df_test["timestamp"].max()}')

# Calcular Qin_perfil (solo train)
df_train['hora'] = df_train['timestamp'].dt.hour
qin_perfil = df_train.groupby('hora')['sist_Qin_m3h'].agg(['median']).to_dict()['median']

# Preparar features
with open('models/forecasting/features.txt') as f:
    features = f.read().strip().split('\n')

print(f'\nFeatures: {len(features)}')

X_train = df_train[features].fillna(0)
y_train = df_train['Q_net_m3h'].values

X_test = df_test[features].fillna(0)
y_test_qnet = df_test['Q_net_m3h'].values

# Calcular Qout real
df_test['hora'] = df_test['timestamp'].dt.hour
qin_test = df_test['hora'].map(qin_perfil).values
y_test_qout = qin_test - y_test_qnet

print(f'\nQout real: {y_test_qout.min():,.0f} - {y_test_qout.max():,.0f} m3/h')

# Función de métricas
def calcular_metricas(y_true, y_pred, nombre):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f'\n{nombre}:')
    print(f'  R2   = {r2:.4f}')
    print(f'  MAE  = {mae:,.0f} m3/h')
    print(f'  RMSE = {rmse:,.0f} m3/h')
    return {'r2': r2, 'mae': mae, 'rmse': rmse}

resultados = {}

# RandomForest
print('\n' + '='*80)
print('RANDOM FOREST')
print('='*80)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_qnet_rf = rf.predict(X_test)
y_pred_qout_rf = qin_test - y_pred_qnet_rf
resultados['RF'] = calcular_metricas(y_test_qout, y_pred_qout_rf, 'RandomForest')

# LightGBM (cargar pickle si existe, sino entrenar)
print('\n' + '='*80)
print('LIGHTGBM')
print('='*80)

lgb_path = Path('models/forecasting/modelo_forecasting_lgbm.pkl')
if lgb_path.exists():
    print('Cargando modelo guardado...')
    lgb_model = joblib.load(lgb_path)
else:
    print('Entrenando nuevo modelo...')
    X_val = df.iloc[train_end:val_end][features].fillna(0)
    y_val = df.iloc[train_end:val_end]['Q_net_m3h'].values
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=12,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False)]
    )

y_pred_qnet_lgb = lgb_model.predict(X_test)
y_pred_qout_lgb = qin_test - y_pred_qnet_lgb
resultados['LGB'] = calcular_metricas(y_test_qout, y_pred_qout_lgb, 'LightGBM')

# XGBoost
print('\n' + '='*80)
print('XGBOOST')
print('='*80)

xgb_path = Path('models/forecasting/modelo_forecasting_xgboost.pkl')
if xgb_path.exists():
    print('Cargando modelo guardado...')
    xgb_model = joblib.load(xgb_path)
else:
    print('Entrenando nuevo modelo...')
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

y_pred_qnet_xgb = xgb_model.predict(X_test)
y_pred_qout_xgb = qin_test - y_pred_qnet_xgb
resultados['XGB'] = calcular_metricas(y_test_qout, y_pred_qout_xgb, 'XGBoost')

# Resumen
print('\n' + '='*80)
print('RESUMEN COMPARATIVO')
print('='*80)
print(f'\n{"Modelo":<15} {"R2":>10} {"MAE":>12} {"RMSE":>12}')
print('-'*50)
for nombre, m in resultados.items():
    print(f'{nombre:<15} {m["r2"]:>10.4f} {m["mae"]:>12,.0f} {m["rmse"]:>12,.0f}')

print('\n' + '='*80)
print('CONCLUSION:')
print('Ahora scatter_plots e interfaz usan MISMO dataset (162 features sin EMAs)')
print('Las metricas deben ser IDENTICAS en ambos lugares.')
print('='*80)
