"""
Script para entrenar modelo LightGBM y guardarlo como modelo por defecto
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("❌ LightGBM no está instalado")
    print("   Instalar con: pip install lightgbm")
    exit(1)

print("\n" + "="*80)
print("ENTRENAMIENTO DE MODELO LIGHTGBM")
print("="*80)

# 1. Cargar features
print("\n[1] Cargando features...")
with open('models/forecasting/features.txt', 'r', encoding='utf-8') as f:
    features = [line.strip() for line in f if line.strip()]
print(f"  ✅ {len(features)} features")

# 2. Cargar dataset
print("\n[2] Cargando dataset...")
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').copy()

# Eliminar features de Q_net si existen
q_features = [col for col in df.columns if col.startswith('Q_net_m3h__')]
if q_features:
    print(f"  ⚠️  Eliminando {len(q_features)} features de Q_net")
    df = df.drop(columns=q_features)

# Agregar features categóricas si faltan
for col in features:
    if col not in df.columns:
        if 'cal_' in col:
            df[col] = 0  # Features calendario vacías

print(f"  ✅ Dataset: {len(df):,} registros")

# 3. Split train/val/test
print("\n[3] Separando datos...")
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end]
df_val = df.iloc[train_end:val_end]
df_test = df.iloc[val_end:]

X_train = df_train[features].fillna(0)
y_train = df_train['Q_net_m3h']
X_val = df_val[features].fillna(0)
y_val = df_val['Q_net_m3h']
X_test = df_test[features].fillna(0)
y_test = df_test['Q_net_m3h']

print(f"  Train: {len(X_train):,}")
print(f"  Val:   {len(X_val):,}")
print(f"  Test:  {len(X_test):,}")

# 4. Entrenar LightGBM
print("\n[4] Entrenando LightGBM...")
modelo = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=12,
    learning_rate=0.05,
    num_leaves=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

modelo.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(30, verbose=False)]
)

print("  ✅ Entrenamiento completado")

# 5. Evaluar
print("\n[5] Evaluando modelo...")
y_pred_train = modelo.predict(X_train)
y_pred_val = modelo.predict(X_val)
y_pred_test = modelo.predict(X_test)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_val = mean_absolute_error(y_val, y_pred_val)
mae_test = mean_absolute_error(y_test, y_pred_test)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print(f"  Train MAE: {mae_train:,.1f} m³/h")
print(f"  Val MAE:   {mae_val:,.1f} m³/h")
print(f"  Test MAE:  {mae_test:,.1f} m³/h")
print(f"  Test RMSE: {rmse_test:,.1f} m³/h")
print(f"  Test R²:   {r2_test:.4f}")

# 6. Guardar modelo
print("\n[6] Guardando modelo...")
output_path = 'models/forecasting/modelo_forecasting_lgbm.pkl'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
joblib.dump(modelo, output_path)
print(f"  ✅ Guardado: {output_path}")

# 7. Comparar con XGBoost si existe
xgb_path = 'models/forecasting/modelo_forecasting_xgboost.pkl'
if Path(xgb_path).exists():
    print("\n[7] Comparando con XGBoost existente...")
    modelo_xgb = joblib.load(xgb_path)
    y_pred_xgb = modelo_xgb.predict(X_test)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    
    print(f"  XGBoost:  MAE={mae_xgb:,.1f}, R²={r2_xgb:.4f}")
    print(f"  LightGBM: MAE={mae_test:,.1f}, R²={r2_test:.4f}")
    
    if mae_test < mae_xgb:
        print(f"  ✅ LightGBM es {((mae_xgb-mae_test)/mae_xgb*100):.1f}% mejor")
    else:
        print(f"  ⚠️ XGBoost es {((mae_test-mae_xgb)/mae_test*100):.1f}% mejor")

print("\n" + "="*80)
print("✅ COMPLETADO")
print("="*80)
print("\nEl modelo LightGBM ahora se usará por defecto en:")
print("  • Predicción 24 horas")
print("  • Predicción 72 horas")
print("  • Predicción 72 horas Multi-Modelo")
print()
