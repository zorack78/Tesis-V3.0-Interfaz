"""
Entrenamiento del modelo de predicción de DEMANDA con TEMPERATURA
Incluye temperatura horaria como feature adicional
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import json

print("="*80)
print("ENTRENAMIENTO: MODELO DE DEMANDA CON TEMPERATURA")
print("="*80)

# ==================== 1. CARGAR DATOS ====================
print("\n[1/7] Cargando datos...")

# Cargar datos de demanda procesados
demanda_path = Path('data/processed/data_processed_demanda_valid.csv')
df = pd.read_csv(demanda_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
df.set_index('timestamp_utc', inplace=True)

print(f"   ✅ Datos de demanda: {len(df)} registros")

# Cargar datos de clima
clima_path = Path('data/raw/BD_Clima2024a202509_UTC.csv')
df_clima = pd.read_csv(clima_path)
df_clima['timestamp'] = pd.to_datetime(df_clima['timestamp'])
df_clima.set_index('timestamp', inplace=True)

print(f"   ✅ Datos de clima: {len(df_clima)} registros")

# ==================== 2. MERGE CON TEMPERATURA ====================
print("\n[2/7] Merging datos con temperatura...")

# Join por timestamp
df = df.join(df_clima[['temp', 'HR', 'mmhr']], how='inner')

# Verificar merge
print(f"   ✅ Registros después del merge: {len(df)}")
print(f"   ✅ Columna 'temp' agregada (temperatura en °C)")

# Verificar valores nulos
nulos_temp = df['temp'].isna().sum()
if nulos_temp > 0:
    print(f"   ⚠️  {nulos_temp} valores nulos en temperatura - eliminando...")
    df = df.dropna(subset=['temp'])
    print(f"   ✅ Registros finales: {len(df)}")
else:
    print(f"   ✅ Sin valores nulos en temperatura")

# ==================== 3. FILTRAR OUTLIERS ====================
print("\n[3/7] Filtrando outliers en Demanda...")

antes_filtro = len(df)
df = df[
    (df['Demanda_m3_hr'] >= 0) &
    (df['Demanda_m3_hr'] <= 30000)
].copy()
outliers_removidos = antes_filtro - len(df)

print(f"   ✅ Outliers removidos: {outliers_removidos} ({outliers_removidos/antes_filtro*100:.2f}%)")
print(f"   ✅ Datos limpios: {len(df)} registros")
print(f"   📊 Demanda rango: [{df['Demanda_m3_hr'].min():.0f}, {df['Demanda_m3_hr'].max():.0f}] m³/hr")
print(f"   🌡️  Temperatura rango: [{df['temp'].min():.1f}, {df['temp'].max():.1f}] °C")

# ==================== 4. PREPARAR FEATURES ====================
print("\n[4/7] Preparando features con temperatura...")

# Crear LAGs de DEMANDA si no existen
print("   📐 Creando LAGs de Demanda...")

df['Demanda_lag_1h'] = df['Demanda_m3_hr'].shift(1)
df['Demanda_lag_2h'] = df['Demanda_m3_hr'].shift(2)
df['Demanda_lag_24h'] = df['Demanda_m3_hr'].shift(24)
df['Demanda_lag_168h'] = df['Demanda_m3_hr'].shift(168)

# Rolling statistics de Demanda
df['Demanda_rolling_mean_6h'] = df['Demanda_m3_hr'].rolling(window=6, min_periods=1).mean()
df['Demanda_rolling_std_6h'] = df['Demanda_m3_hr'].rolling(window=6, min_periods=1).std()
df['Demanda_rolling_mean_24h'] = df['Demanda_m3_hr'].rolling(window=24, min_periods=1).mean()
df['Demanda_rolling_std_24h'] = df['Demanda_m3_hr'].rolling(window=24, min_periods=1).std()

# Diff
df['Demanda_diff_1h'] = df['Demanda_m3_hr'].diff()
df['Demanda_diff_24h'] = df['Demanda_m3_hr'].diff(24)

# Ratio
df['Demanda_ratio_vs_24h'] = df['Demanda_m3_hr'] / (df['Demanda_lag_24h'] + 1)

print("   ✅ Features LAG creadas")

# Lista de features del modelo ORIGINAL (sin temperatura)
features_originales = [
    # Temporales básicas
    'hour', 'day_of_week', 'month', 'is_weekend',
    # Cíclicas
    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
    # Eventos
    'feriado', 'temporada_turistica_alta',
    # LAGs de Demanda
    'Demanda_lag_1h', 'Demanda_lag_2h', 'Demanda_lag_24h', 'Demanda_lag_168h',
    # Rolling de Demanda
    'Demanda_rolling_mean_6h', 'Demanda_rolling_std_6h',
    'Demanda_rolling_mean_24h', 'Demanda_rolling_std_24h',
    # Diferencias
    'Demanda_diff_1h', 'Demanda_diff_24h',
    # Ratio
    'Demanda_ratio_vs_24h'
]

# AGREGAR TEMPERATURA A LA LISTA DE FEATURES
features_con_temperatura = features_originales + ['temperatura']

# Renombrar columna temp → temperatura para claridad
df['temperatura'] = df['temp']

print(f"   ✅ Features originales: {len(features_originales)}")
print(f"   ✅ Features con temperatura: {len(features_con_temperatura)}")
print(f"   📋 Nueva feature: temperatura (°C)")

# ==================== 5. SPLIT DE DATOS ====================
print("\n[5/7] Dividiendo datos (70% train, 15% val, 15% test)...")

# Verificar que no hay NaN en features
df_clean = df[features_con_temperatura + ['Demanda_m3_hr']].dropna()
print(f"   ✅ Registros sin NaN: {len(df_clean)}")

X = df_clean[features_con_temperatura]
y = df_clean['Demanda_m3_hr']

# Split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, shuffle=False
)

# Split temp: 50% val, 50% test (15% y 15% del total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, shuffle=False
)

print(f"   ✅ Train: {len(X_train)} registros ({len(X_train)/len(X)*100:.1f}%)")
print(f"   ✅ Val:   {len(X_val)} registros ({len(X_val)/len(X)*100:.1f}%)")
print(f"   ✅ Test:  {len(X_test)} registros ({len(X_test)/len(X)*100:.1f}%)")

# ==================== 6. ENTRENAR MODELO XGBOOST ====================
print("\n[6/7] Entrenando modelo XGBoost con temperatura...")

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("   ✅ Modelo entrenado")

# ==================== 7. EVALUAR MODELO ====================
print("\n[7/7] Evaluando modelo...")

# Predicciones
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Métricas train
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

# Métricas validación
val_r2 = r2_score(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)
val_mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100

# Métricas test
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

print("\n" + "="*80)
print("MÉTRICAS DEL MODELO CON TEMPERATURA")
print("="*80)

print(f"\n📊 TRAIN:")
print(f"   R² Score: {train_r2:.4f}")
print(f"   RMSE:     {train_rmse:,.2f} m³/hr")
print(f"   MAE:      {train_mae:,.2f} m³/hr")
print(f"   MAPE:     {train_mape:.2f}%")

print(f"\n📊 VALIDACIÓN:")
print(f"   R² Score: {val_r2:.4f}")
print(f"   RMSE:     {val_rmse:,.2f} m³/hr")
print(f"   MAE:      {val_mae:,.2f} m³/hr")
print(f"   MAPE:     {val_mape:.2f}%")

print(f"\n📊 TEST:")
print(f"   R² Score: {test_r2:.4f}")
print(f"   RMSE:     {test_rmse:,.2f} m³/hr")
print(f"   MAE:      {test_mae:,.2f} m³/hr")
print(f"   MAPE:     {test_mape:.2f}%")

# Feature importance
print("\n" + "="*80)
print("IMPORTANCIA DE FEATURES")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': features_con_temperatura,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📈 Top 10 features más importantes:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:30} : {row['importance']:.4f}")

# Buscar temperatura en el ranking
temp_rank = feature_importance.reset_index(drop=True)
temp_idx = temp_rank[temp_rank['feature'] == 'temperatura'].index[0]
temp_importance = temp_rank.loc[temp_idx, 'importance']

print(f"\n🌡️  Temperatura ranking: #{temp_idx + 1} de {len(features_con_temperatura)}")
print(f"   Importancia: {temp_importance:.4f}")

# ==================== 8. COMPARAR CON MODELO ORIGINAL ====================
print("\n" + "="*80)
print("COMPARACIÓN: MODELO ORIGINAL vs MODELO CON TEMPERATURA")
print("="*80)

# Cargar métricas del modelo original (si existe)
metricas_original_path = Path('models/demanda/metricas.json')
if metricas_original_path.exists():
    with open(metricas_original_path, 'r') as f:
        metricas_original = json.load(f)
    
    print("\n📊 MODELO ORIGINAL (sin temperatura):")
    r2_orig = metricas_original.get('test_r2', 0)
    mape_orig = metricas_original.get('test_mape', 0)
    n_feat_orig = metricas_original.get('n_features', 0)
    print(f"   R² Test:  {r2_orig:.4f}")
    print(f"   MAPE Test: {mape_orig:.2f}%")
    print(f"   Features: {n_feat_orig}")
    
    print("\n📊 MODELO NUEVO (con temperatura):")
    print(f"   R² Test:  {test_r2:.4f}")
    print(f"   MAPE Test: {test_mape:.2f}%")
    print(f"   Features: {len(features_con_temperatura)}")
    
    # Calcular mejora
    mejora_r2 = test_r2 - r2_orig
    mejora_mape = mape_orig - test_mape
    
    print("\n💡 MEJORA:")
    print(f"   R²:   {mejora_r2:+.4f} {'✅' if mejora_r2 > 0 else '❌'}")
    print(f"   MAPE: {mejora_mape:+.2f}% {'✅' if mejora_mape > 0 else '❌'}")
else:
    print("\n⚠️  No se encontraron métricas del modelo original para comparar")

# ==================== 9. GUARDAR MODELO ====================
print("\n" + "="*80)
print("GUARDANDO MODELO Y MÉTRICAS")
print("="*80)

# Crear directorio
output_dir = Path('models/demanda_temperatura')
output_dir.mkdir(parents=True, exist_ok=True)

# Guardar modelo
model_path = output_dir / 'demanda_temperatura_xgboost_model.pkl'
joblib.dump(model, model_path)
print(f"\n✅ Modelo guardado: {model_path}")

# Guardar features
features_path = output_dir / 'features.txt'
with open(features_path, 'w') as f:
    for feature in features_con_temperatura:
        f.write(f"{feature}\n")
print(f"✅ Features guardadas: {features_path}")

# Guardar métricas
metricas = {
    'train_r2': float(train_r2),
    'train_rmse': float(train_rmse),
    'train_mae': float(train_mae),
    'train_mape': float(train_mape),
    'val_r2': float(val_r2),
    'val_rmse': float(val_rmse),
    'val_mae': float(val_mae),
    'val_mape': float(val_mape),
    'test_r2': float(test_r2),
    'test_rmse': float(test_rmse),
    'test_mae': float(test_mae),
    'test_mape': float(test_mape),
    'n_features': len(features_con_temperatura),
    'features': features_con_temperatura,
    'temperatura_importance': float(temp_importance),
    'temperatura_rank': int(temp_idx + 1)
}

metricas_path = output_dir / 'metricas.json'
with open(metricas_path, 'w') as f:
    json.dump(metricas, f, indent=4)
print(f"✅ Métricas guardadas: {metricas_path}")

# Guardar feature importance
importance_path = output_dir / 'feature_importance.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"✅ Feature importance guardada: {importance_path}")

print("\n" + "="*80)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*80)
print(f"\n📁 Archivos generados en: {output_dir}")
print(f"   • demanda_temperatura_xgboost_model.pkl")
print(f"   • features.txt ({len(features_con_temperatura)} features)")
print(f"   • metricas.json")
print(f"   • feature_importance.csv")

print("\n🌡️  Temperatura incluida con éxito")
print(f"   Ranking: #{temp_idx + 1} de {len(features_con_temperatura)}")
print(f"   Importancia: {temp_importance:.4f}")

if test_mape < 5.0:
    print(f"\n🎉 EXCELENTE: MAPE Test = {test_mape:.2f}% (< 5%)")
elif test_mape < 10.0:
    print(f"\n✅ BUENO: MAPE Test = {test_mape:.2f}% (< 10%)")
else:
    print(f"\n⚠️  ACEPTABLE: MAPE Test = {test_mape:.2f}%")

print("\n" + "="*80)
