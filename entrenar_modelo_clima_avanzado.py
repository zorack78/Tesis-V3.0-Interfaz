"""
Entrenamiento del modelo de predicción de DEMANDA con FEATURES CLIMÁTICAS AVANZADAS
Basado en experiencia operacional: los extremos y cambios bruscos son lo importante

Features climáticas implementadas:
1. Temperatura: LAGs (1h, 24h, 168h), rolling (mean/std/max/min 6h y 24h), diffs, extremos
2. Humedad Relativa: LAGs, rolling, extremos de sequedad/humedad
3. Precipitación: acumulados, binarios de lluvia
4. Interacciones: temperatura extrema combinada con hora, sensación térmica
5. Cambios bruscos: detección de variaciones rápidas de temperatura

Comparación con:
- Modelo original (sin temperatura): 21 features
- Modelo simple (temperatura directa): 22 features
- Modelo avanzado (features climáticas): ~35+ features
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
print("ENTRENAMIENTO: MODELO CON FEATURES CLIMÁTICAS AVANZADAS")
print("="*80)
print("\n💡 Basado en experiencia operacional:")
print("   • Lo importante NO es temperatura absoluta")
print("   • Lo importante SON los extremos (frío/calor intenso)")
print("   • Lo importante SON los cambios bruscos")
print("   • Temperatura estable templada → poco impacto")
print("   • Temperatura fuera de rango normal → alto impacto")

# ==================== 1. CARGAR DATOS ====================
print("\n" + "="*80)
print("[1/8] CARGANDO DATOS")
print("="*80)

# Cargar datos de demanda procesados
demanda_path = Path('data/processed/data_processed_demanda_valid.csv')
df = pd.read_csv(demanda_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
df.set_index('timestamp_utc', inplace=True)

print(f"\n✅ Datos de demanda: {len(df)} registros")
print(f"   Rango temporal: {df.index.min()} a {df.index.max()}")

# Cargar datos de clima
clima_path = Path('data/raw/BD_Clima2024a202509_UTC.csv')
df_clima = pd.read_csv(clima_path)
df_clima['timestamp'] = pd.to_datetime(df_clima['timestamp'])
df_clima.set_index('timestamp', inplace=True)

print(f"✅ Datos de clima: {len(df_clima)} registros")
print(f"   Columnas: {list(df_clima.columns)}")

# ==================== 2. MERGE Y ANÁLISIS INICIAL ====================
print("\n" + "="*80)
print("[2/8] MERGE CON DATOS CLIMÁTICOS")
print("="*80)

# Join por timestamp
df = df.join(df_clima[['temp', 'HR', 'mmhr']], how='inner')

print(f"\n✅ Registros después del merge: {len(df)}")
print(f"✅ Columnas climáticas agregadas: temp, HR, mmhr")

# Verificar valores nulos
nulos_clima = df[['temp', 'HR', 'mmhr']].isna().sum()
print(f"\n📊 Valores nulos por columna:")
for col, count in nulos_clima.items():
    print(f"   {col:10}: {count:6} ({count/len(df)*100:5.2f}%)")

# Eliminar filas con NaN en clima
if nulos_clima.sum() > 0:
    df = df.dropna(subset=['temp', 'HR', 'mmhr'])
    print(f"\n✅ Registros después de eliminar NaN: {len(df)}")

# Estadísticas clima
print(f"\n📊 Estadísticas climáticas:")
print(f"   Temperatura: [{df['temp'].min():.1f}, {df['temp'].max():.1f}] °C (media: {df['temp'].mean():.1f})")
print(f"   Humedad:     [{df['HR'].min():.1f}, {df['HR'].max():.1f}] % (media: {df['HR'].mean():.1f})")
print(f"   Precipitación: [{df['mmhr'].min():.1f}, {df['mmhr'].max():.1f}] mm/hr (media: {df['mmhr'].mean():.3f})")

# ==================== 3. CREAR FEATURES CLIMÁTICAS AVANZADAS ====================
print("\n" + "="*80)
print("[3/8] CREANDO FEATURES CLIMÁTICAS AVANZADAS")
print("="*80)

print("\n🌡️ [A] FEATURES DE TEMPERATURA:")

# Renombrar para claridad
df['temperatura'] = df['temp']

# LAGs de temperatura (capturar inercia térmica)
print("   • LAGs (1h, 2h, 24h, 168h)...")
df['temp_lag_1h'] = df['temperatura'].shift(1)
df['temp_lag_2h'] = df['temperatura'].shift(2)
df['temp_lag_24h'] = df['temperatura'].shift(24)
df['temp_lag_168h'] = df['temperatura'].shift(168)

# Rolling statistics (capturar olas de calor/frío)
print("   • Rolling mean/std/max/min (6h y 24h)...")
df['temp_rolling_mean_6h'] = df['temperatura'].rolling(window=6, min_periods=1).mean()
df['temp_rolling_std_6h'] = df['temperatura'].rolling(window=6, min_periods=1).std()
df['temp_rolling_max_6h'] = df['temperatura'].rolling(window=6, min_periods=1).max()
df['temp_rolling_min_6h'] = df['temperatura'].rolling(window=6, min_periods=1).min()

df['temp_rolling_mean_24h'] = df['temperatura'].rolling(window=24, min_periods=1).mean()
df['temp_rolling_std_24h'] = df['temperatura'].rolling(window=24, min_periods=1).std()
df['temp_rolling_max_24h'] = df['temperatura'].rolling(window=24, min_periods=1).max()
df['temp_rolling_min_24h'] = df['temperatura'].rolling(window=24, min_periods=1).min()

# Diferencias (cambios bruscos)
print("   • Diferencias (1h, 24h)...")
df['temp_diff_1h'] = df['temperatura'].diff()
df['temp_diff_24h'] = df['temperatura'].diff(24)

# Cambios bruscos de temperatura (>3°C en 1h = cambio brusco)
print("   • Detección de cambios bruscos...")
df['temp_cambio_brusco'] = (df['temp_diff_1h'].abs() > 3).astype(int)

# Extremos de temperatura (basado en percentiles y experiencia)
print("   • Clasificación de extremos...")
temp_p10 = df['temperatura'].quantile(0.10)  # ~10°C
temp_p90 = df['temperatura'].quantile(0.90)  # ~20°C

df['temp_extremo_frio'] = (df['temperatura'] < temp_p10).astype(int)
df['temp_extremo_calor'] = (df['temperatura'] > temp_p90).astype(int)
df['temp_templado'] = ((df['temperatura'] >= temp_p10) & (df['temperatura'] <= temp_p90)).astype(int)

# Temperatura muy extrema (operacional: <5°C o >30°C)
df['temp_muy_frio'] = (df['temperatura'] < 5).astype(int)
df['temp_muy_calor'] = (df['temperatura'] > 30).astype(int)

print(f"      • Frío extremo (<{temp_p10:.1f}°C): {df['temp_extremo_frio'].sum()} registros ({df['temp_extremo_frio'].sum()/len(df)*100:.1f}%)")
print(f"      • Calor extremo (>{temp_p90:.1f}°C): {df['temp_extremo_calor'].sum()} registros ({df['temp_extremo_calor'].sum()/len(df)*100:.1f}%)")
print(f"      • Muy frío (<5°C): {df['temp_muy_frio'].sum()} registros")
print(f"      • Muy calor (>30°C): {df['temp_muy_calor'].sum()} registros")

print("\n💧 [B] FEATURES DE HUMEDAD RELATIVA:")

# Renombrar
df['humedad'] = df['HR']

# LAGs
print("   • LAGs (1h, 24h)...")
df['HR_lag_1h'] = df['humedad'].shift(1)
df['HR_lag_24h'] = df['humedad'].shift(24)

# Rolling
print("   • Rolling mean/std (24h)...")
df['HR_rolling_mean_24h'] = df['humedad'].rolling(window=24, min_periods=1).mean()
df['HR_rolling_std_24h'] = df['humedad'].rolling(window=24, min_periods=1).std()

# Extremos de humedad
print("   • Clasificación de extremos...")
df['HR_muy_baja'] = (df['humedad'] < 30).astype(int)  # Sequedad extrema
df['HR_muy_alta'] = (df['humedad'] > 90).astype(int)  # Humedad extrema

print(f"      • Sequedad extrema (<30%): {df['HR_muy_baja'].sum()} registros ({df['HR_muy_baja'].sum()/len(df)*100:.1f}%)")
print(f"      • Humedad extrema (>90%): {df['HR_muy_alta'].sum()} registros ({df['HR_muy_alta'].sum()/len(df)*100:.1f}%)")

print("\n🌧️ [C] FEATURES DE PRECIPITACIÓN:")

# Renombrar
df['precipitacion'] = df['mmhr']

# Acumulados
print("   • Acumulados (6h, 24h)...")
df['precip_acum_6h'] = df['precipitacion'].rolling(window=6, min_periods=1).sum()
df['precip_acum_24h'] = df['precipitacion'].rolling(window=24, min_periods=1).sum()

# Binarios
print("   • Detección de lluvia...")
df['hay_lluvia'] = (df['precipitacion'] > 0).astype(int)
df['lluvia_intensa'] = (df['precipitacion'] > 5).astype(int)  # >5mm/hr = intensa

print(f"      • Horas con lluvia: {df['hay_lluvia'].sum()} ({df['hay_lluvia'].sum()/len(df)*100:.1f}%)")
print(f"      • Lluvia intensa (>5mm/hr): {df['lluvia_intensa'].sum()} registros")

print("\n🔗 [D] FEATURES DE INTERACCIÓN:")

# Sensación térmica simplificada (temp + humedad)
print("   • Sensación térmica (temp ajustada por HR)...")
df['sensacion_termica'] = df['temperatura'] - (df['humedad'] - 50) * 0.1

# Temperatura extrema por hora (interacción)
print("   • Temperatura extrema x hora del día...")
df['temp_extrema_manana'] = df['temp_extremo_calor'] * df['is_morning']
df['temp_extrema_tarde'] = df['temp_extremo_calor'] * df['is_afternoon']

# Condiciones adversas combinadas
print("   • Condiciones climáticas adversas combinadas...")
df['condiciones_adversas'] = (
    df['temp_muy_frio'] | 
    df['temp_muy_calor'] | 
    df['HR_muy_baja'] | 
    df['lluvia_intensa']
).astype(int)

print(f"      • Registros con condiciones adversas: {df['condiciones_adversas'].sum()} ({df['condiciones_adversas'].sum()/len(df)*100:.1f}%)")

# ==================== 4. CREAR FEATURES DE DEMANDA (LAGs) ====================
print("\n" + "="*80)
print("[4/8] CREANDO FEATURES DE DEMANDA (LAGs y derivadas)")
print("="*80)

print("\n📐 LAGs de Demanda...")
df['Demanda_lag_1h'] = df['Demanda_m3_hr'].shift(1)
df['Demanda_lag_2h'] = df['Demanda_m3_hr'].shift(2)
df['Demanda_lag_24h'] = df['Demanda_m3_hr'].shift(24)
df['Demanda_lag_168h'] = df['Demanda_m3_hr'].shift(168)

print("📐 Rolling statistics de Demanda...")
df['Demanda_rolling_mean_6h'] = df['Demanda_m3_hr'].rolling(window=6, min_periods=1).mean()
df['Demanda_rolling_std_6h'] = df['Demanda_m3_hr'].rolling(window=6, min_periods=1).std()
df['Demanda_rolling_mean_24h'] = df['Demanda_m3_hr'].rolling(window=24, min_periods=1).mean()
df['Demanda_rolling_std_24h'] = df['Demanda_m3_hr'].rolling(window=24, min_periods=1).std()

print("📐 Diferencias y ratios de Demanda...")
df['Demanda_diff_1h'] = df['Demanda_m3_hr'].diff()
df['Demanda_diff_24h'] = df['Demanda_m3_hr'].diff(24)
df['Demanda_ratio_vs_24h'] = df['Demanda_m3_hr'] / (df['Demanda_lag_24h'] + 1)

print("✅ Features de demanda creadas")

# ==================== 5. LISTA COMPLETA DE FEATURES ====================
print("\n" + "="*80)
print("[5/8] DEFINIENDO LISTA DE FEATURES")
print("="*80)

# Features temporales (originales)
features_temporales = [
    'hour', 'day_of_week', 'month', 'is_weekend',
    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
    'feriado', 'temporada_turistica_alta'
]

# Features de demanda (LAGs)
features_demanda = [
    'Demanda_lag_1h', 'Demanda_lag_2h', 'Demanda_lag_24h', 'Demanda_lag_168h',
    'Demanda_rolling_mean_6h', 'Demanda_rolling_std_6h',
    'Demanda_rolling_mean_24h', 'Demanda_rolling_std_24h',
    'Demanda_diff_1h', 'Demanda_diff_24h', 'Demanda_ratio_vs_24h'
]

# Features climáticas avanzadas
features_clima = [
    # Temperatura base
    'temperatura',
    # LAGs temperatura
    'temp_lag_1h', 'temp_lag_2h', 'temp_lag_24h', 'temp_lag_168h',
    # Rolling temperatura
    'temp_rolling_mean_6h', 'temp_rolling_std_6h', 'temp_rolling_max_6h', 'temp_rolling_min_6h',
    'temp_rolling_mean_24h', 'temp_rolling_std_24h', 'temp_rolling_max_24h', 'temp_rolling_min_24h',
    # Diferencias y cambios
    'temp_diff_1h', 'temp_diff_24h', 'temp_cambio_brusco',
    # Extremos temperatura
    'temp_extremo_frio', 'temp_extremo_calor', 'temp_templado',
    'temp_muy_frio', 'temp_muy_calor',
    # Humedad
    'humedad', 'HR_lag_1h', 'HR_lag_24h',
    'HR_rolling_mean_24h', 'HR_rolling_std_24h',
    'HR_muy_baja', 'HR_muy_alta',
    # Precipitación
    'precipitacion', 'precip_acum_6h', 'precip_acum_24h',
    'hay_lluvia', 'lluvia_intensa',
    # Interacciones
    'sensacion_termica', 'temp_extrema_manana', 'temp_extrema_tarde',
    'condiciones_adversas'
]

# Lista completa
features_completas = features_temporales + features_demanda + features_clima

print(f"\n📊 RESUMEN DE FEATURES:")
print(f"   • Temporales:  {len(features_temporales):2} features")
print(f"   • Demanda:     {len(features_demanda):2} features")
print(f"   • Climáticas:  {len(features_clima):2} features")
print(f"   • TOTAL:       {len(features_completas):2} features")

print(f"\n📋 Features climáticas ({len(features_clima)}):")
for i, feat in enumerate(features_clima, 1):
    print(f"   {i:2}. {feat}")

# ==================== 6. PREPARAR DATOS (3 ESCENARIOS) ====================
print("\n" + "="*80)
print("[6/8] PREPARANDO DATOS - 3 ESCENARIOS")
print("="*80)

print("\n🔍 Analizando outliers en Demanda...")
outliers_negativos = (df['Demanda_m3_hr'] < 0).sum()
outliers_positivos = (df['Demanda_m3_hr'] > 30000).sum()
total_outliers = outliers_negativos + outliers_positivos

print(f"   • Demanda < 0:       {outliers_negativos:5} registros ({outliers_negativos/len(df)*100:.2f}%)")
print(f"   • Demanda > 30,000:  {outliers_positivos:5} registros ({outliers_positivos/len(df)*100:.2f}%)")
print(f"   • Total outliers:    {total_outliers:5} registros ({total_outliers/len(df)*100:.2f}%)")

# ESCENARIO 1: CON OUTLIERS
print(f"\n📌 ESCENARIO 1: CON OUTLIERS")
df_con_outliers = df[features_completas + ['Demanda_m3_hr']].dropna()
print(f"   Registros: {len(df_con_outliers)}")

# ESCENARIO 2: SIN OUTLIERS (filtrado)
print(f"\n📌 ESCENARIO 2: SIN OUTLIERS (filtrado 0-30,000 m³/hr)")
df_sin_outliers = df[
    (df['Demanda_m3_hr'] >= 0) & 
    (df['Demanda_m3_hr'] <= 30000)
][features_completas + ['Demanda_m3_hr']].dropna()
print(f"   Registros: {len(df_sin_outliers)}")
print(f"   Removidos: {len(df_con_outliers) - len(df_sin_outliers)}")

# ESCENARIO 3: CON IMPUTACIÓN (winsorize extremos)
print(f"\n📌 ESCENARIO 3: CON IMPUTACIÓN (winsorize a percentiles 1-99)")
df_imputado = df_con_outliers.copy()
p1 = df_imputado['Demanda_m3_hr'].quantile(0.01)
p99 = df_imputado['Demanda_m3_hr'].quantile(0.99)
df_imputado['Demanda_m3_hr'] = df_imputado['Demanda_m3_hr'].clip(lower=p1, upper=p99)
print(f"   Registros: {len(df_imputado)}")
print(f"   Rango imputado: [{p1:.1f}, {p99:.1f}] m³/hr")

# ==================== 7. ENTRENAR 3 MODELOS ====================
print("\n" + "="*80)
print("[7/8] ENTRENANDO 3 MODELOS COMPARATIVOS")
print("="*80)

resultados = {}

for escenario_nombre, df_escenario in [
    ("CON_OUTLIERS", df_con_outliers),
    ("SIN_OUTLIERS", df_sin_outliers),
    ("IMPUTADO", df_imputado)
]:
    print(f"\n{'='*80}")
    print(f"🔬 ENTRENANDO: {escenario_nombre}")
    print(f"{'='*80}")
    
    # Preparar X, y
    X = df_escenario[features_completas]
    y = df_escenario['Demanda_m3_hr']
    
    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, shuffle=False
    )
    
    print(f"\n📊 Split de datos:")
    print(f"   Train: {len(X_train):5} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Val:   {len(X_val):5} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(X_test):5} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Entrenar XGBoost
    print(f"\n🚀 Entrenando XGBoost...")
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
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Métricas
    metricas = {
        'train_r2': r2_score(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_mape': np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100,
        'val_r2': r2_score(y_val, y_val_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'val_mape': np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100,
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_mape': np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100,
        'n_features': len(features_completas),
        'n_registros': len(X)
    }
    
    resultados[escenario_nombre] = {
        'model': model,
        'metricas': metricas
    }
    
    print(f"\n✅ MÉTRICAS - {escenario_nombre}:")
    print(f"   Test R²:   {metricas['test_r2']:.4f}")
    print(f"   Test RMSE: {metricas['test_rmse']:,.2f} m³/hr")
    print(f"   Test MAE:  {metricas['test_mae']:,.2f} m³/hr")
    print(f"   Test MAPE: {metricas['test_mape']:.2f}%")

# ==================== 8. ANÁLISIS DE IMPORTANCIA Y COMPARACIÓN ====================
print("\n" + "="*80)
print("[8/8] ANÁLISIS DE IMPORTANCIA Y COMPARACIÓN")
print("="*80)

# Usar modelo SIN OUTLIERS para análisis de importancia
modelo_final = resultados['SIN_OUTLIERS']['model']

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features_completas,
    'importance': modelo_final.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📈 TOP 20 FEATURES MÁS IMPORTANTES:")
for idx, row in feature_importance.head(20).iterrows():
    categoria = '🌡️' if any(x in row['feature'] for x in ['temp', 'HR', 'precip', 'humedad', 'sensacion', 'lluvia']) else \
                '📊' if 'Demanda' in row['feature'] else '🕐'
    print(f"   {categoria} {row['feature']:30} : {row['importance']:.4f}")

# Importancia TOTAL por categoría
print(f"\n📊 IMPORTANCIA TOTAL POR CATEGORÍA:")
importancia_temporal = feature_importance[feature_importance['feature'].isin(features_temporales)]['importance'].sum()
importancia_demanda = feature_importance[feature_importance['feature'].isin(features_demanda)]['importance'].sum()
importancia_clima = feature_importance[feature_importance['feature'].isin(features_clima)]['importance'].sum()

print(f"   🕐 Temporal:  {importancia_temporal:.4f} ({importancia_temporal*100:.2f}%)")
print(f"   📊 Demanda:   {importancia_demanda:.4f} ({importancia_demanda*100:.2f}%)")
print(f"   🌡️ Climática: {importancia_clima:.4f} ({importancia_clima*100:.2f}%)")

# Comparación entre escenarios
print(f"\n📊 COMPARACIÓN ENTRE ESCENARIOS (Test):")
print(f"\n{'Escenario':<20} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'MAPE':>8} {'N':>7}")
print("-" * 80)
for nombre, data in resultados.items():
    m = data['metricas']
    print(f"{nombre:<20} {m['test_r2']:>8.4f} {m['test_rmse']:>10,.0f} {m['test_mae']:>10,.0f} {m['test_mape']:>7.2f}% {m['n_registros']:>7,}")

# ==================== 9. GUARDAR MODELO Y RESULTADOS ====================
print("\n" + "="*80)
print("GUARDANDO MODELO Y RESULTADOS")
print("="*80)

# Usar modelo SIN OUTLIERS como modelo final
output_dir = Path('models/demanda_clima_avanzado')
output_dir.mkdir(parents=True, exist_ok=True)

# Guardar modelo
model_path = output_dir / 'demanda_clima_avanzado_xgboost_model.pkl'
joblib.dump(modelo_final, model_path)
print(f"\n✅ Modelo guardado: {model_path}")

# Guardar features
features_path = output_dir / 'features.txt'
with open(features_path, 'w') as f:
    for feature in features_completas:
        f.write(f"{feature}\n")
print(f"✅ Features guardadas: {features_path}")

# Guardar métricas de todos los escenarios
metricas_comparacion = {}
for nombre, data in resultados.items():
    metricas_comparacion[nombre] = data['metricas']

metricas_path = output_dir / 'metricas_comparacion.json'
with open(metricas_path, 'w') as f:
    json.dump(metricas_comparacion, f, indent=4)
print(f"✅ Métricas guardadas: {metricas_path}")

# Guardar feature importance
importance_path = output_dir / 'feature_importance.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"✅ Feature importance guardada: {importance_path}")

# Guardar resumen por categoría
resumen_categorias = {
    'importancia_temporal': float(importancia_temporal),
    'importancia_demanda': float(importancia_demanda),
    'importancia_clima': float(importancia_clima),
    'n_features_temporal': len(features_temporales),
    'n_features_demanda': len(features_demanda),
    'n_features_clima': len(features_clima),
    'n_features_total': len(features_completas)
}

resumen_path = output_dir / 'resumen_categorias.json'
with open(resumen_path, 'w') as f:
    json.dump(resumen_categorias, f, indent=4)
print(f"✅ Resumen por categorías guardado: {resumen_path}")

# ==================== 10. CONCLUSIONES ====================
print("\n" + "="*80)
print("🎯 CONCLUSIONES Y RECOMENDACIONES")
print("="*80)

mejor_escenario = max(resultados.items(), key=lambda x: x[1]['metricas']['test_r2'])
print(f"\n✅ Mejor escenario: {mejor_escenario[0]}")
print(f"   R² Test: {mejor_escenario[1]['metricas']['test_r2']:.4f}")
print(f"   MAPE Test: {mejor_escenario[1]['metricas']['test_mape']:.2f}%")

print(f"\n💡 HALLAZGOS CLAVE:")
print(f"   1. Features climáticas representan {importancia_clima*100:.1f}% de la importancia total")
print(f"   2. Se crearon {len(features_clima)} features climáticas avanzadas")
print(f"   3. Outliers impactan: Δ R² = {resultados['CON_OUTLIERS']['metricas']['test_r2'] - resultados['SIN_OUTLIERS']['metricas']['test_r2']:.4f}")
print(f"   4. Modelo final usa {len(features_completas)} features totales")

print(f"\n🎯 RECOMENDACIONES:")
print(f"   ✓ Usar modelo SIN OUTLIERS para producción")
print(f"   ✓ Features climáticas aportan valor (especialmente extremos)")
print(f"   ✓ Mantener monitoreo de outliers en ETL")
print(f"   ✓ Considerar features de extremos para interfaz operacional")

print("\n" + "="*80)
print("✅ PROCESO COMPLETADO")
print("="*80)
print(f"\n📁 Archivos generados en: {output_dir}/")
print(f"   • demanda_clima_avanzado_xgboost_model.pkl")
print(f"   • features.txt ({len(features_completas)} features)")
print(f"   • metricas_comparacion.json")
print(f"   • feature_importance.csv")
print(f"   • resumen_categorias.json")

print("\n" + "="*80)
