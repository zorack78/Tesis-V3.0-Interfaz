"""
REENTRENAMIENTO DE MODELOS CON NUEVAS FEATURES EMA
===================================================
Entrena LightGBM y XGBoost con el dataset actualizado que incluye
features EMA de variables externas (sin data leakage).

Autor: Sistema de Análisis
Fecha: 2025-12-08
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb

print('='*80)
print('REENTRENAMIENTO CON FEATURES EMA EXTERNAS')
print('='*80)

# ==============================================================================
# 1. CARGAR DATASET CON NUEVAS FEATURES
# ==============================================================================

print('\n1️⃣ Cargando dataset con features EMA...')
print('-'*80)

df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f'✅ Dataset cargado: {df.shape}')
print(f'   Registros: {len(df):,}')
print(f'   Features: {df.shape[1]:,}')

# Contar nuevas features EMA
ema_features = [col for col in df.columns if '_ema_' in col]
print(f'\n📊 Features EMA encontradas: {len(ema_features)}')
for feat in sorted(ema_features):
    print(f'   • {feat}')

# ==============================================================================
# 2. SPLIT 70/15/15
# ==============================================================================

print('\n2️⃣ Split de datos (70/15/15)...')
print('-'*80)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

print(f'✅ Train: {len(df_train):,} registros ({len(df_train)/n:.1%})')
print(f'✅ Val:   {len(df_val):,} registros ({len(df_val)/n:.1%})')
print(f'✅ Test:  {len(df_test):,} registros ({len(df_test)/n:.1%})')

print(f'\n📅 Períodos:')
print(f'   Train: {df_train["timestamp"].min()} → {df_train["timestamp"].max()}')
print(f'   Val:   {df_val["timestamp"].min()} → {df_val["timestamp"].max()}')
print(f'   Test:  {df_test["timestamp"].min()} → {df_test["timestamp"].max()}')

# ==============================================================================
# 3. PREPARAR FEATURES Y TARGET
# ==============================================================================

print('\n3️⃣ Preparando features y target...')
print('-'*80)

# Target
target = 'Q_net_m3h'

# Excluir columnas que no son features
exclude = ['timestamp', 'Q_net_m3h', 'sist_Qin_m3h', 'sist_Vtotal_m3']

# Solo features numéricas (excluir texto)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in exclude]

print(f'✅ Target: {target}')
print(f'✅ Features: {len(feature_cols)}')

# Verificar cuántas features EMA están en el set final
ema_in_features = [f for f in feature_cols if '_ema_' in f]
print(f'✅ Features EMA activas: {len(ema_in_features)}')

# Preparar matrices
X_train = df_train[feature_cols].fillna(0)
y_train = df_train[target]

X_val = df_val[feature_cols].fillna(0)
y_val = df_val[target]

X_test = df_test[feature_cols].fillna(0)
y_test = df_test[target]

print(f'\n📐 Dimensiones:')
print(f'   X_train: {X_train.shape}')
print(f'   X_val:   {X_val.shape}')
print(f'   X_test:  {X_test.shape}')

# ==============================================================================
# 4. ENTRENAR LIGHTGBM
# ==============================================================================

print('\n4️⃣ Entrenando LightGBM con features EMA...')
print('-'*80)

modelo_lgb = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=12,
    learning_rate=0.05,
    num_leaves=50,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print('   Entrenando...')
modelo_lgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(20, verbose=False)]
)

# Predicción
y_pred_lgb_val = modelo_lgb.predict(X_val)
y_pred_lgb_test = modelo_lgb.predict(X_test)

# Métricas
r2_val_lgb = r2_score(y_val, y_pred_lgb_val)
mae_val_lgb = mean_absolute_error(y_val, y_pred_lgb_val)
rmse_val_lgb = np.sqrt(mean_squared_error(y_val, y_pred_lgb_val))

r2_test_lgb = r2_score(y_test, y_pred_lgb_test)
mae_test_lgb = mean_absolute_error(y_test, y_pred_lgb_test)
rmse_test_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb_test))

print(f'\n   📊 Métricas VALIDACIÓN:')
print(f'      R²:   {r2_val_lgb:.4f}')
print(f'      MAE:  {mae_val_lgb:,.2f} m³/h')
print(f'      RMSE: {rmse_val_lgb:,.2f} m³/h')

print(f'\n   📊 Métricas TEST:')
print(f'      R²:   {r2_test_lgb:.4f}')
print(f'      MAE:  {mae_test_lgb:,.2f} m³/h')
print(f'      RMSE: {rmse_test_lgb:,.2f} m³/h')

# ==============================================================================
# 5. ENTRENAR XGBOOST
# ==============================================================================

print('\n5️⃣ Entrenando XGBoost con features EMA...')
print('-'*80)

modelo_xgb = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=12,  # Aumentado de 7 a 12 (como LightGBM)
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print('   Entrenando...')
modelo_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Predicción
y_pred_xgb_val = modelo_xgb.predict(X_val)
y_pred_xgb_test = modelo_xgb.predict(X_test)

# Métricas
r2_val_xgb = r2_score(y_val, y_pred_xgb_val)
mae_val_xgb = mean_absolute_error(y_val, y_pred_xgb_val)
rmse_val_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb_val))

r2_test_xgb = r2_score(y_test, y_pred_xgb_test)
mae_test_xgb = mean_absolute_error(y_test, y_pred_xgb_test)
rmse_test_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb_test))

print(f'\n   📊 Métricas VALIDACIÓN:')
print(f'      R²:   {r2_val_xgb:.4f}')
print(f'      MAE:  {mae_val_xgb:,.2f} m³/h')
print(f'      RMSE: {rmse_val_xgb:,.2f} m³/h')

print(f'\n   📊 Métricas TEST:')
print(f'      R²:   {r2_test_xgb:.4f}')
print(f'      MAE:  {mae_test_xgb:,.2f} m³/h')
print(f'      RMSE: {rmse_test_xgb:,.2f} m³/h')

# ==============================================================================
# 6. COMPARACIÓN CON MODELOS ANTERIORES
# ==============================================================================

print('\n6️⃣ Comparación con modelos anteriores...')
print('-'*80)

# Cargar métricas anteriores
try:
    import json
    with open('outputs/metricas_oficiales_todos_modelos.json', 'r') as f:
        metricas_anteriores = json.load(f)
    
    r2_anterior_lgb = metricas_anteriores['metricas']['LightGBM']['Q_net']['r2']
    mae_anterior_lgb = metricas_anteriores['metricas']['LightGBM']['Q_net']['mae']
    
    r2_anterior_xgb = metricas_anteriores['metricas']['XGBoost']['Q_net']['r2']
    mae_anterior_xgb = metricas_anteriores['metricas']['XGBoost']['Q_net']['mae']
    
    print(f'\n📊 LightGBM:')
    print(f'   Anterior:  R² = {r2_anterior_lgb:.4f}, MAE = {mae_anterior_lgb:,.0f} m³/h')
    print(f'   Con EMAs:  R² = {r2_test_lgb:.4f}, MAE = {mae_test_lgb:,.0f} m³/h')
    mejora_r2_lgb = (r2_test_lgb - r2_anterior_lgb) / r2_anterior_lgb * 100
    mejora_mae_lgb = (mae_anterior_lgb - mae_test_lgb) / mae_anterior_lgb * 100
    print(f'   Cambio:    R² {mejora_r2_lgb:+.2f}%, MAE {mejora_mae_lgb:+.2f}%')
    
    print(f'\n📊 XGBoost:')
    print(f'   Anterior:  R² = {r2_anterior_xgb:.4f}, MAE = {mae_anterior_xgb:,.0f} m³/h')
    print(f'   Con EMAs:  R² = {r2_test_xgb:.4f}, MAE = {mae_test_xgb:,.0f} m³/h')
    mejora_r2_xgb = (r2_test_xgb - r2_anterior_xgb) / r2_anterior_xgb * 100
    mejora_mae_xgb = (mae_anterior_xgb - mae_test_xgb) / mae_anterior_xgb * 100
    print(f'   Cambio:    R² {mejora_r2_xgb:+.2f}%, MAE {mejora_mae_xgb:+.2f}%')
    
except Exception as e:
    print(f'⚠️  No se pudieron cargar métricas anteriores: {str(e)}')

# ==============================================================================
# 7. ANÁLISIS DE IMPORTANCIA DE FEATURES EMA
# ==============================================================================

print('\n7️⃣ Importancia de features EMA...')
print('-'*80)

# Feature importance de LightGBM
importance_lgb = pd.DataFrame({
    'feature': feature_cols,
    'importance': modelo_lgb.feature_importances_
}).sort_values('importance', ascending=False)

# Top features EMA
top_ema = importance_lgb[importance_lgb['feature'].str.contains('_ema_')].head(10)

print(f'\n📊 Top 10 features EMA más importantes (LightGBM):')
for idx, row in top_ema.iterrows():
    print(f'   {row["feature"]:<35} {row["importance"]:>8.0f}')

# ==============================================================================
# 8. GUARDAR MODELOS
# ==============================================================================

print('\n8️⃣ Guardando modelos actualizados...')
print('-'*80)

Path('models/forecasting').mkdir(parents=True, exist_ok=True)

# Backup de modelos anteriores
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_path = Path('models/forecasting/backup')
backup_path.mkdir(exist_ok=True)

for modelo_file in ['modelo_forecasting_lgbm.pkl', 'modelo_forecasting_xgboost.pkl']:
    if Path(f'models/forecasting/{modelo_file}').exists():
        import shutil
        shutil.copy(
            f'models/forecasting/{modelo_file}',
            f'{backup_path}/{modelo_file.replace(".pkl", f"_{timestamp}.pkl")}'
        )
        print(f'   💾 Backup: {modelo_file}')

# Guardar nuevos modelos
joblib.dump(modelo_lgb, 'models/forecasting/modelo_forecasting_lgbm.pkl')
joblib.dump(modelo_xgb, 'models/forecasting/modelo_forecasting_xgboost.pkl')

print(f'\n✅ Modelos guardados:')
print(f'   • models/forecasting/modelo_forecasting_lgbm.pkl')
print(f'   • models/forecasting/modelo_forecasting_xgboost.pkl')

# Guardar lista de features
with open('models/forecasting/features_con_ema.txt', 'w') as f:
    for feat in feature_cols:
        f.write(f'{feat}\n')

print(f'   • models/forecasting/features_con_ema.txt ({len(feature_cols)} features)')

# ==============================================================================
# 9. RESUMEN FINAL
# ==============================================================================

print('\n' + '='*80)
print('✅ REENTRENAMIENTO COMPLETADO')
print('='*80)

print(f'\n🎯 RESULTADOS FINALES (Test Set):')
print(f'\n{"Modelo":<15} {"R²":>10} {"MAE":>15} {"RMSE":>15}')
print('-'*55)
print(f'{"LightGBM":<15} {r2_test_lgb:>10.4f} {mae_test_lgb:>15,.0f} {rmse_test_lgb:>15,.0f}')
print(f'{"XGBoost":<15} {r2_test_xgb:>10.4f} {mae_test_xgb:>15,.0f} {rmse_test_xgb:>15,.0f}')

print(f'\n📊 Features totales: {len(feature_cols)}')
print(f'   • Features EMA: {len(ema_in_features)}')
print(f'   • Otras features: {len(feature_cols) - len(ema_in_features)}')

print('\n' + '='*80)
