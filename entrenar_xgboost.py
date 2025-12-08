"""
Entrenar solo XGBoost con NumPy 1.24 (compatible con Anaconda)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime

print('='*80)
print('ENTRENAMIENTO DE XGBOOST')
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

X_test = df_test[features].fillna(0)
y_test = df_test['Q_net_m3h'].values

# Calcular Qin_perfil para convertir a Qout
df_train['hora'] = df_train['timestamp'].dt.hour
qin_perfil = df_train.groupby('hora')['sist_Qin_m3h'].agg(['median']).to_dict()['median']

df_test['hora'] = df_test['timestamp'].dt.hour
qin_test = df_test['hora'].map(qin_perfil).values
y_test_qout = qin_test - y_test

print('\nEntrenando XGBoost...')

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
y_pred_qout = qin_test - y_pred_xgb

r2 = r2_score(y_test_qout, y_pred_qout)
mae = mean_absolute_error(y_test_qout, y_pred_qout)
rmse = np.sqrt(mean_squared_error(y_test_qout, y_pred_qout))

print(f'\nXGBoost:')
print(f'  R2   = {r2:.4f}')
print(f'  MAE  = {mae:,.0f} m3/h')
print(f'  RMSE = {rmse:,.0f} m3/h')

# Guardar modelo
output_path = Path('models/forecasting/modelo_forecasting_xgboost.pkl')
metadata = {
    'modelo': 'XGBoost',
    'fecha_entrenamiento': datetime.now().isoformat(),
    'n_features': len(features),
    'features_sin_emas': True,
    'metricas_test': {
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    },
    'parametros': modelo_xgb.get_params()
}

joblib.dump({'modelo': modelo_xgb, 'metadata': metadata}, output_path, protocol=4)
print(f'\nGuardado: {output_path}')
print('   (Protocolo 4 - compatible con NumPy < 2.0)')

print('\n' + '='*80)
print('COMPLETADO!')
print('='*80)
