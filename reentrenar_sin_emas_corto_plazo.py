"""
Reentrenamiento SIN FEATURES DE CORTO PLAZO
============================================
Elimina EMAs de 6h, 12h, 24h que contienen datos del test set.
Solo usa lag_168h (semana anterior) para información histórica.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Cargar features originales
with open('models/forecasting/features.txt') as f:
    features_originales = [l.strip() for l in f.readlines()]

# ELIMINAR features que usan datos del test set
features_prohibidas = [
    'Q_net_m3h__ema_win_6h',
    'Q_net_m3h__ema_win_12h', 
    'Q_net_m3h__ema_win_24h'
]

features_limpias = [f for f in features_originales if f not in features_prohibidas]

print("="*80)
print("REENTRENAMIENTO SIN EMAs DE CORTO PLAZO")
print("="*80)
print(f"\nFeatures originales: {len(features_originales)}")
print(f"Features eliminadas: {len(features_prohibidas)}")
print(f"Features limpias: {len(features_limpias)}")
print(f"\nFeatures eliminadas:")
for f in features_prohibidas:
    print(f"  ❌ {f}")

# Cargar dataset
print("\n📂 Cargando dataset...")
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').copy()

# Agregar features categóricas manualmente
print("📊 Creando features categóricas...")

# Funciones auxiliares de la interfaz
def agregar_features_categoricas(df):
    """Replica la función de la interfaz"""
    # temp_nivel
    df['temp_nivel_Frio'] = (df['clima_temp_c'] < 10.5).astype(int)
    df['temp_nivel_Normal'] = ((df['clima_temp_c'] >= 10.5) & (df['clima_temp_c'] < 16.3)).astype(int)
    df['temp_nivel_Calor'] = (df['clima_temp_c'] >= 16.3).astype(int)
    
    # periodo_dia
    df['hora'] = df['timestamp'].dt.hour
    df['periodo_dia_Madrugada'] = df['hora'].isin([0,1,2,3,4,5]).astype(int)
    df['periodo_dia_Manana_critica'] = df['hora'].isin([6,7,8,9]).astype(int)
    df['periodo_dia_Dia'] = df['hora'].isin([10,11,12,13,14,15,16,17]).astype(int)
    df['periodo_dia_Noche'] = df['hora'].isin([18,19,20,21]).astype(int)
    df['periodo_dia_Noche_tardia'] = df['hora'].isin([22,23]).astype(int)
    
    # Regímenes (binarios simples)
    df['regimen_equilibrio'] = (df['Q_net_m3h'].abs() < 500).astype(int)
    
    # Hora bisagra
    df['es_hora_bisagra'] = df['hora'].isin([6,7,8,9,10,12,18,19,20]).astype(int)
    
    # Interacciones
    df['temp_x_hora'] = df['clima_temp_c'] * df['hora']
    df['delta_temp_6h_x_hora'] = df['clima_temp_delta_6h'] * df['hora']
    
    # Fin de semana
    df['dia_semana'] = df['timestamp'].dt.dayofweek
    df['es_fin_de_semana'] = (df['dia_semana'] >= 5).astype(int)
    df['temp_x_finde'] = df['clima_temp_c'] * df['es_fin_de_semana']
    
    return df

df = agregar_features_categoricas(df)

# Verificar features
missing = [f for f in features_limpias if f not in df.columns]
if missing:
    print(f"\n❌ Features faltantes: {missing}")
    exit(1)

print("✅ Todas las features disponibles")

# Split
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

X_train = df_train[features_limpias]
y_train = df_train['Q_net_m3h']
X_val = df_val[features_limpias]
y_val = df_val['Q_net_m3h']
X_test = df_test[features_limpias]
y_test = df_test['Q_net_m3h']

print(f"\n📊 Split:")
print(f"  Train: {len(X_train):,} registros (hasta {df_train['timestamp'].max()})")
print(f"  Val:   {len(X_val):,} registros")
print(f"  Test:  {len(X_test):,} registros (desde {df_test['timestamp'].min()})")

# Rellenar NaN
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)
X_test = X_test.fillna(0)

# ==========================================
# ENTRENAR XGBOOST
# ==========================================
print("\n" + "="*80)
print("ENTRENANDO XGBOOST (SIN EMAs CORTO PLAZO)")
print("="*80)

modelo_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=8,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

eval_set = [(X_val, y_val)]
modelo_xgb.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=False
)

y_pred_xgb = modelo_xgb.predict(X_test)

# Métricas
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"\n✅ XGBoost entrenado:")
print(f"  R² = {r2_xgb:.4f}")
print(f"  MAE = {mae_xgb:,.0f} m³/hr")
print(f"  RMSE = {rmse_xgb:,.0f} m³/hr")

# Guardar modelo SIN EMAs
output_dir = Path('models/forecasting_sin_emas')
output_dir.mkdir(exist_ok=True)

joblib.dump(modelo_xgb, output_dir / 'modelo_xgboost_sin_emas.pkl')
with open(output_dir / 'features.txt', 'w') as f:
    f.write('\n'.join(features_limpias))

metricas = {
    'r2': r2_xgb,
    'mae': mae_xgb,
    'rmse': rmse_xgb,
    'n_features': len(features_limpias),
    'features_eliminadas': features_prohibidas
}
with open(output_dir / 'metricas.json', 'w') as f:
    json.dump(metricas, f, indent=2)

print(f"\n💾 Modelo guardado en: {output_dir}")

# ==========================================
# ENTRENAR RANDOMFOREST
# ==========================================
print("\n" + "="*80)
print("ENTRENANDO RANDOMFOREST (SIN EMAs CORTO PLAZO)")
print("="*80)

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
y_pred_rf = modelo_rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"\n✅ RandomForest entrenado:")
print(f"  R² = {r2_rf:.4f}")
print(f"  MAE = {mae_rf:,.0f} m³/hr")
print(f"  RMSE = {rmse_rf:,.0f} m³/hr")

joblib.dump(modelo_rf, output_dir / 'modelo_rf_sin_emas.pkl')

# ==========================================
# COMPARAR CON MODELOS ORIGINALES
# ==========================================
print("\n" + "="*80)
print("COMPARACIÓN: CON vs SIN EMAs")
print("="*80)

# Cargar métricas originales
try:
    with open('outputs/metricas_ML/metricas_reales.json') as f:
        metricas_con_emas = json.load(f)['modelos']
    
    print("\nXGBoost:")
    print(f"  CON EMAs:  R²={metricas_con_emas['XGBoost V3.0 (Actual)']['r2']:.4f}, MAE={metricas_con_emas['XGBoost V3.0 (Actual)']['mae']:,.0f}")
    print(f"  SIN EMAs:  R²={r2_xgb:.4f}, MAE={mae_xgb:,.0f}")
    print(f"  Diferencia MAE: {mae_xgb - metricas_con_emas['XGBoost V3.0 (Actual)']['mae']:+,.0f} m³/hr")
    
    print("\nRandomForest:")
    print(f"  CON EMAs:  R²={metricas_con_emas['RandomForest']['r2']:.4f}, MAE={metricas_con_emas['RandomForest']['mae']:,.0f}")
    print(f"  SIN EMAs:  R²={r2_rf:.4f}, MAE={mae_rf:,.0f}")
    print(f"  Diferencia MAE: {mae_rf - metricas_con_emas['RandomForest']['mae']:+,.0f} m³/hr")
except:
    print("\n⚠️ No se pudieron cargar métricas originales para comparar")

print("\n" + "="*80)
print("✅ REENTRENAMIENTO COMPLETADO")
print("="*80)
print("\nModelos SIN EMAs de corto plazo guardados en:")
print(f"  {output_dir}")
print("\nEstos modelos NO ven datos del test set en sus features.")
print("Ahora ejecuta: python comparar_con_sin_emas.py")
