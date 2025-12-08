"""
VERIFICACIÓN DE CONSISTENCIA DE MÉTRICAS
=========================================
Este script replica EXACTAMENTE lo que hace la interfaz en la pestaña Testing
para verificar que las métricas sean consistentes.

Usa:
- Mismo dataset
- Mismo split (70/15/15)
- Mismo cálculo de Qin_perfil
- Mismos modelos (entrena en memoria como la interfaz)
- Mismo cálculo de Qout
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

print('='*80)
print('VERIFICACION DE CONSISTENCIA - REPLICA INTERFAZ')
print('='*80)

# ==============================================================================
# 1. CARGAR DATASET (IGUAL QUE INTERFAZ)
# ==============================================================================

print('\n[1] Cargando dataset...')

df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').copy()

print(f'✅ Dataset: {df.shape}')

# ==============================================================================
# 2. SPLIT 70/15/15 (IGUAL QUE INTERFAZ)
# ==============================================================================

print('\n2️⃣ Split 70/15/15...')

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

print(f'✅ Train: {len(df_train):,} ({len(df_train)/n:.1%})')
print(f'✅ Val:   {len(df_val):,} ({len(df_val)/n:.1%})')
print(f'✅ Test:  {len(df_test):,} ({len(df_test)/n:.1%})')
print(f'   Período test: {df_test["timestamp"].min()} → {df_test["timestamp"].max()}')

# ==============================================================================
# 3. CALCULAR Qin_perfil (IGUAL QUE INTERFAZ - SOLO TRAIN)
# ==============================================================================

print('\n3️⃣ Calculando Qin_perfil (solo con datos de TRAIN)...')

# Intentar cargar BD_Qin_m3_UTC.csv
qin_path = Path('data/raw/BD_Qin_m3_UTC.csv')

if qin_path.exists():
    df_qin_raw = pd.read_csv(qin_path)
    df_qin_raw['timestamp'] = pd.to_datetime(df_qin_raw['timestamp'])
    df_qin_raw['hora'] = df_qin_raw['timestamp'].dt.hour
    
    # SOLO datos hasta el final de train
    fecha_limite_train = df_train['timestamp'].max()
    df_qin_train = df_qin_raw[df_qin_raw['timestamp'] <= fecha_limite_train].copy()
    
    qin_perfil = df_qin_train.groupby('hora')['Qin'].agg(['median', 'mean', 'std', 'count'])
    print(f'✅ Qin_perfil desde BD_Qin: {len(df_qin_train):,} registros de TRAIN')
else:
    # Fallback a sist_Qin_m3h
    df_train['hora'] = df_train['timestamp'].dt.hour
    qin_perfil = df_train.groupby('hora')['sist_Qin_m3h'].agg(['median', 'mean', 'std', 'count'])
    print(f'✅ Qin_perfil desde sist_Qin_m3h: {len(df_train):,} registros de TRAIN')

print(f'   Rango Qin: {qin_perfil["median"].min():,.0f} - {qin_perfil["median"].max():,.0f} m³/h')

# ==============================================================================
# 4. PREPARAR FEATURES (IGUAL QUE INTERFAZ)
# ==============================================================================

print('\n4️⃣ Preparando features...')

# Excluir columnas no numéricas y target
exclude = ['timestamp', 'Q_net_m3h', 'sist_Qin_m3h', 'sist_Vtotal_m3']
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features = [col for col in numeric_cols if col not in exclude]

print(f'✅ Features: {len(features)}')

X_train = df_train[features].fillna(0)
y_train = df_train['Q_net_m3h'].values

X_val = df_val[features].fillna(0)
y_val = df_val['Q_net_m3h'].values

X_test = df_test[features].fillna(0)
y_test_qnet = df_test['Q_net_m3h'].values

# ==============================================================================
# 5. CALCULAR Qout REAL (IGUAL QUE INTERFAZ)
# ==============================================================================

print('\n5️⃣ Calculando Qout real...')

df_test['hora'] = df_test['timestamp'].dt.hour
qin_por_hora = df_test['hora'].map(qin_perfil['median']).values

# Qout = Qin_perfil - Q_net
y_test_qout = qin_por_hora - y_test_qnet

print(f'✅ Qout real calculado')
print(f'   Rango: {y_test_qout.min():,.0f} - {y_test_qout.max():,.0f} m³/h')
print(f'   Media: {y_test_qout.mean():,.0f} m³/h')

# ==============================================================================
# 6. ENTRENAR Y EVALUAR MODELOS (IGUAL QUE INTERFAZ)
# ==============================================================================

def calcular_metricas(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
    }

resultados = {}

# --- RandomForest ---
print('\n6️⃣ Entrenando RandomForest (como interfaz)...')
modelo_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
modelo_rf.fit(X_train, y_train)
y_pred_qnet_rf = modelo_rf.predict(X_test)
y_pred_qout_rf = qin_por_hora - y_pred_qnet_rf
resultados['RandomForest'] = calcular_metricas(y_test_qout, y_pred_qout_rf)
print(f'   ✅ R²={resultados["RandomForest"]["r2"]:.4f}, MAE={resultados["RandomForest"]["mae"]:,.0f}')

# --- LightGBM ---
print('\n7️⃣ Entrenando LightGBM (como interfaz)...')
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
y_pred_qnet_lgb = modelo_lgb.predict(X_test)
y_pred_qout_lgb = qin_por_hora - y_pred_qnet_lgb
resultados['LightGBM'] = calcular_metricas(y_test_qout, y_pred_qout_lgb)
print(f'   ✅ R²={resultados["LightGBM"]["r2"]:.4f}, MAE={resultados["LightGBM"]["mae"]:,.0f}')

# --- XGBoost ---
print('\n8️⃣ Cargando/Entrenando XGBoost...')
# Intentar cargar modelo pickle
xgb_path = Path('models/forecasting/modelo_forecasting_xgboost.pkl')
if xgb_path.exists():
    import joblib
    try:
        modelo_xgb = joblib.load(xgb_path)
        print('   ✅ XGBoost cargado desde pickle')
        usar_pickle = True
    except:
        print('   ⚠️  Error cargando pickle, entrenando nuevo...')
        usar_pickle = False
else:
    print('   ⚠️  Pickle no encontrado, entrenando nuevo...')
    usar_pickle = False

if not usar_pickle:
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

y_pred_qnet_xgb = modelo_xgb.predict(X_test)
y_pred_qout_xgb = qin_por_hora - y_pred_qnet_xgb
resultados['XGBoost'] = calcular_metricas(y_test_qout, y_pred_qout_xgb)
print(f'   ✅ R²={resultados["XGBoost"]["r2"]:.4f}, MAE={resultados["XGBoost"]["mae"]:,.0f}')

# ==============================================================================
# 7. REPORTE FINAL
# ==============================================================================

print('\n' + '='*80)
print('RESULTADOS FINALES (DEBERIA COINCIDIR CON INTERFAZ)')
print('='*80)

print(f'\n{"Modelo":<15} {"R2":>10} {"MAE":>15} {"RMSE":>15} {"MAPE":>10}')
print('-'*70)
for nombre, metrics in resultados.items():
    print(f'{nombre:<15} {metrics["r2"]:>10.4f} {metrics["mae"]:>15,.0f} '
          f'{metrics["rmse"]:>15,.0f} {metrics["mape"]:>10.2f}%')

print(f'\nPeriodo de test:')
print(f'   Inicio: {df_test["timestamp"].min()}')
print(f'   Fin:    {df_test["timestamp"].max()}')
print(f'   Registros: {len(df_test):,}')

print(f'\nFeatures utilizadas: {len(features)}')
print(f'   (mismas que usa la interfaz)')

print('\n' + '='*80)
print('VERIFICACION COMPLETADA')
print('='*80)
print('\nEstos valores deberian ser IDENTICOS a los de la interfaz.')
print('Si no lo son, hay inconsistencia en el codigo.')
