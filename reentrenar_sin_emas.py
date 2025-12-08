"""
Reentrenar modelos LightGBM y XGBoost sin EMAs (162 features)
Compatible con NumPy < 2.0 (para Anaconda)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime

print('='*80)
print('REENTRENAMIENTO DE MODELOS SIN EMAs')
print(f'NumPy version: {np.__version__}')
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
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

print(f'Train: {len(df_train):,} ({len(df_train)/n:.1%})')
print(f'Val:   {len(df_val):,} ({len(df_val)/n:.1%})')
print(f'Test:  {len(df_test):,} ({len(df_test)/n:.1%})')

# Features
with open('models/forecasting/features.txt') as f:
    features = f.read().strip().split('\n')

print(f'\nFeatures: {len(features)}')

X_train = df_train[features].fillna(0)
y_train = df_train['Q_net_m3h'].values

X_val = df_val[features].fillna(0)
y_val = df_val['Q_net_m3h'].values

X_test = df_test[features].fillna(0)
y_test = df_test['Q_net_m3h'].values

# Calcular Qin_perfil para convertir a Qout
df_train['hora'] = df_train['timestamp'].dt.hour
qin_perfil = df_train.groupby('hora')['sist_Qin_m3h'].agg(['median']).to_dict()['median']

df_test['hora'] = df_test['timestamp'].dt.hour
qin_test = df_test['hora'].map(qin_perfil).values
y_test_qout = qin_test - y_test

def calcular_metricas(y_true, y_pred_qnet, qin, nombre):
    y_pred_qout = qin - y_pred_qnet
    r2 = r2_score(y_true, y_pred_qout)
    mae = mean_absolute_error(y_true, y_pred_qout)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_qout))
    
    print(f'\n{nombre}:')
    print(f'  R2   = {r2:.4f}')
    print(f'  MAE  = {mae:,.0f} m3/h')
    print(f'  RMSE = {rmse:,.0f} m3/h')
    
    return r2, mae, rmse

# ============================================================================
# LIGHTGBM
# ============================================================================

print('\n' + '='*80)
print('ENTRENANDO LIGHTGBM')
print('='*80)

modelo_lgb = lgb.LGBMRegressor(
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

modelo_lgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(20, verbose=False)]
)

y_pred_lgb = modelo_lgb.predict(X_test)
r2_lgb, mae_lgb, rmse_lgb = calcular_metricas(y_test_qout, y_pred_lgb, qin_test, 'LightGBM')

# Guardar modelo
output_path_lgb = Path('models/forecasting/modelo_forecasting_lgbm.pkl')
metadata_lgb = {
    'modelo': 'LightGBM',
    'fecha_entrenamiento': datetime.now().isoformat(),
    'n_features': len(features),
    'features_sin_emas': True,
    'metricas_test': {
        'r2': r2_lgb,
        'mae': mae_lgb,
        'rmse': rmse_lgb
    },
    'parametros': modelo_lgb.get_params()
}

joblib.dump({'modelo': modelo_lgb, 'metadata': metadata_lgb}, output_path_lgb, protocol=4)
print(f'\nGuardado: {output_path_lgb}')
print('   (Protocolo 4 - compatible con NumPy < 2.0)')

# ============================================================================
# XGBOOST
# ============================================================================

print('\n' + '='*80)
print('ENTRENANDO XGBOOST')
print('='*80)

modelo_xgb = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

modelo_xgb.fit(X_train, y_train)

y_pred_xgb = modelo_xgb.predict(X_test)
r2_xgb, mae_xgb, rmse_xgb = calcular_metricas(y_test_qout, y_pred_xgb, qin_test, 'XGBoost')

# Guardar modelo
output_path_xgb = Path('models/forecasting/modelo_forecasting_xgboost.pkl')
metadata_xgb = {
    'modelo': 'XGBoost',
    'fecha_entrenamiento': datetime.now().isoformat(),
    'n_features': len(features),
    'features_sin_emas': True,
    'metricas_test': {
        'r2': r2_xgb,
        'mae': mae_xgb,
        'rmse': rmse_xgb
    },
    'parametros': modelo_xgb.get_params()
}

joblib.dump({'modelo': modelo_xgb, 'metadata': metadata_xgb}, output_path_xgb, protocol=4)
print(f'\nGuardado: {output_path_xgb}')
print('   (Protocolo 4 - compatible con NumPy < 2.0)')

# ============================================================================
# RESUMEN
# ============================================================================

print('\n' + '='*80)
print('RESUMEN FINAL')
print('='*80)

print(f'\n{"Modelo":<15} {"R2":>10} {"MAE":>12} {"RMSE":>12}')
print('-'*50)
print(f'{"LightGBM":<15} {r2_lgb:>10.4f} {mae_lgb:>12,.0f} {rmse_lgb:>12,.0f}')
print(f'{"XGBoost":<15} {r2_xgb:>10.4f} {mae_xgb:>12,.0f} {rmse_xgb:>12,.0f}')

print('\n' + '='*80)
print('COMPLETADO!')
print('Modelos reentrenados con dataset SIN EMAs (162 features)')
print('='*80)
