import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Cargar datos
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Splits
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
df_test = df.iloc[val_end:].copy()
df_train = df.iloc[:train_end].copy()

# Perfil Qin (solo train)
qin_perfil = df_train.groupby('hora')['sist_Qin_m3h'].median().to_dict()
qin_test = df_test['hora'].map(qin_perfil).values

# Preparar test
y_test_qnet = df_test['Q_net_m3h'].values
y_test_qout = qin_test - y_test_qnet

# Cargar features y modelo
with open('models/forecasting/features.txt') as f:
    features = [line.strip() for line in f]

X_test = df_test[features].fillna(0).values

# Cargar y predecir XGBoost
modelo_xgb = joblib.load('models/forecasting/modelo_forecasting_xgboost.pkl')
y_pred_qnet = modelo_xgb.predict(X_test)
y_pred_qout = qin_test - y_pred_qnet

# Calcular métricas
mae = mean_absolute_error(y_test_qout, y_pred_qout)
rmse = np.sqrt(mean_squared_error(y_test_qout, y_pred_qout))
r2 = r2_score(y_test_qout, y_pred_qout)

print('='*80)
print('ANÁLISIS R² XGBoost - TODO EL TEST SET')
print('='*80)
print(f'\n📊 Dataset Info:')
print(f'   Total test: {len(y_test_qout)} registros')
print(f'   Periodo: {df_test["timestamp"].min()} → {df_test["timestamp"].max()}')

print(f'\n📈 Qout Real (y_test):')
print(f'   Mean: {y_test_qout.mean():,.0f} m³/h')
print(f'   Std:  {y_test_qout.std():,.0f} m³/h')
print(f'   Min:  {y_test_qout.min():,.0f} m³/h')
print(f'   Max:  {y_test_qout.max():,.0f} m³/h')

print(f'\n🤖 Qout Predicho (y_pred):')
print(f'   Mean: {y_pred_qout.mean():,.0f} m³/h')
print(f'   Std:  {y_pred_qout.std():,.0f} m³/h')
print(f'   Min:  {y_pred_qout.min():,.0f} m³/h')
print(f'   Max:  {y_pred_qout.max():,.0f} m³/h')

print(f'\n📉 Métricas XGBoost (Qout):')
print(f'   MAE:  {mae:,.1f} m³/h')
print(f'   RMSE: {rmse:,.1f} m³/h')
print(f'   R²:   {r2:.6f}')

print('\n' + '='*80)

# Comparar con semanas específicas
primera_semana_idx = np.arange(0, min(168, len(df_test)))
ultima_semana_idx = np.arange(max(0, len(df_test)-168), len(df_test))

r2_primera = r2_score(y_test_qout[primera_semana_idx], y_pred_qout[primera_semana_idx])
r2_ultima = r2_score(y_test_qout[ultima_semana_idx], y_pred_qout[ultima_semana_idx])

print(f'\n🔍 Comparación por período:')
print(f'   R² Primera semana: {r2_primera:.6f}')
print(f'   R² Última semana:  {r2_ultima:.6f}')
print(f'   R² Test completo:  {r2:.6f}')
print('='*80)
