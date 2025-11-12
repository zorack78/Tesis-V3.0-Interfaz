"""
Entrenamiento del modelo de predicción de DEMANDA con FEATURES CLIMÁTICAS AVANZADAS
Basado en experiencia operacional: importa más los EXTREMOS y CAMBIOS BRUSCOS que valores absolutos
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
print("ENTRENAMIENTO: MODELO DE DEMANDA CON CLIMA AVANZADO")
print("="*80)

# ==================== 1. CARGAR DATOS ====================
print("\n[1/8] Cargando datos...")

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
print(f"   📊 Variables clima: {list(df_clima.columns)}")

# ==================== 2. MERGE CON DATOS CLIMÁTICOS ====================
print("\n[2/8] Merging datos climáticos...")

# Join por timestamp
df = df.join(df_clima[['temp', 'HR', 'mmhr']], how='inner')

print(f"   ✅ Registros después del merge: {len(df)}")

# Renombrar para claridad
df['temperatura'] = df['temp']
df['humedad_relativa'] = df['HR']
df['precipitacion'] = df['mmhr']

# Verificar valores nulos
nulos = df[['temperatura', 'humedad_relativa', 'precipitacion']].isna().sum()
if nulos.sum() > 0:
    print(f"   ⚠️  Valores nulos detectados:")
    print(nulos[nulos > 0])
    df = df.dropna(subset=['temperatura', 'humedad_relativa', 'precipitacion'])
    print(f"   ✅ Registros después de limpiar: {len(df)}")
else:
    print(f"   ✅ Sin valores nulos en datos climáticos")

print(f"   🌡️  Temperatura: [{df['temperatura'].min():.1f}, {df['temperatura'].max():.1f}] °C")
print(f"   💧 Humedad: [{df['humedad_relativa'].min():.0f}, {df['humedad_relativa'].max():.0f}] %")
print(f"   🌧️  Precipitación: [{df['precipitacion'].min():.1f}, {df['precipitacion'].max():.1f}] mm/hr")

# ==================== 3. CREAR FEATURES CLIMÁTICAS AVANZADAS ====================
print("\n[3/8] Creando features climáticas avanzadas...")
print("   (Basado en experiencia operacional: extremos y cambios bruscos)")

# --- TEMPERATURA ---
print("\n   🌡️  Features de TEMPERATURA:")

# Umbrales según experiencia operacional
TEMP_CALOR_MODERADO = 23  # Empieza a aumentar demanda
TEMP_CALOR_ALTO = 27      # Aumenta significativamente
TEMP_CALOR_EXTREMO = 30   # Impacto máximo
TEMP_FRIO_MODERADO = 12   # Empieza a cambiar patrón
TEMP_FRIO_ALTO = 8        # Cambio significativo
TEMP_FRIO_EXTREMO = 5     # Impacto máximo

# 1. EXTREMOS DE TEMPERATURA (más importantes que valor absoluto)
df['temp_calor_moderado'] = (df['temperatura'] >= TEMP_CALOR_MODERADO).astype(int)
df['temp_calor_alto'] = (df['temperatura'] >= TEMP_CALOR_ALTO).astype(int)
df['temp_calor_extremo'] = (df['temperatura'] >= TEMP_CALOR_EXTREMO).astype(int)
df['temp_frio_moderado'] = (df['temperatura'] <= TEMP_FRIO_MODERADO).astype(int)
df['temp_frio_alto'] = (df['temperatura'] <= TEMP_FRIO_ALTO).astype(int)
df['temp_frio_extremo'] = (df['temperatura'] <= TEMP_FRIO_EXTREMO).astype(int)

print(f"      • Extremos térmicos (6 binarios)")
print(f"        - Calor extremo (>{TEMP_CALOR_EXTREMO}°C): {df['temp_calor_extremo'].sum()} hrs ({df['temp_calor_extremo'].mean()*100:.1f}%)")
print(f"        - Frío extremo (<{TEMP_FRIO_EXTREMO}°C): {df['temp_frio_extremo'].sum()} hrs ({df['temp_frio_extremo'].mean()*100:.1f}%)")

# 2. DESVIACIÓN DE TEMPERATURA RESPECTO A RANGO CONFORTABLE
TEMP_CONFORTABLE_MIN = 15
TEMP_CONFORTABLE_MAX = 22
df['temp_desviacion_confort'] = 0.0
df.loc[df['temperatura'] < TEMP_CONFORTABLE_MIN, 'temp_desviacion_confort'] = \
    TEMP_CONFORTABLE_MIN - df['temperatura']
df.loc[df['temperatura'] > TEMP_CONFORTABLE_MAX, 'temp_desviacion_confort'] = \
    df['temperatura'] - TEMP_CONFORTABLE_MAX

print(f"      • Desviación del rango confortable ({TEMP_CONFORTABLE_MIN}-{TEMP_CONFORTABLE_MAX}°C)")
print(f"        - Máxima desviación: {df['temp_desviacion_confort'].max():.1f}°C")

# 3. LAGs de temperatura (tendencias)
df['temp_lag_1h'] = df['temperatura'].shift(1)
df['temp_lag_6h'] = df['temperatura'].shift(6)
df['temp_lag_24h'] = df['temperatura'].shift(24)
df['temp_lag_168h'] = df['temperatura'].shift(168)  # Semana anterior

print(f"      • LAGs de temperatura (1h, 6h, 24h, 168h)")

# 4. CAMBIOS BRUSCOS de temperatura (crítico según experiencia)
df['temp_cambio_1h'] = df['temperatura'].diff()  # Cambio en última hora
df['temp_cambio_6h'] = df['temperatura'].diff(6)  # Cambio en 6 horas
df['temp_cambio_24h'] = df['temperatura'].diff(24)  # Cambio día a día

# Detectar cambios bruscos (más de X grados)
df['temp_cambio_brusco_1h'] = (df['temp_cambio_1h'].abs() > 3).astype(int)  # >3°C en 1h
df['temp_cambio_brusco_6h'] = (df['temp_cambio_6h'].abs() > 8).astype(int)  # >8°C en 6h
df['temp_cambio_brusco_24h'] = (df['temp_cambio_24h'].abs() > 12).astype(int)  # >12°C en 24h

cambios_bruscos_1h = df['temp_cambio_brusco_1h'].sum()
cambios_bruscos_6h = df['temp_cambio_brusco_6h'].sum()
print(f"      • Cambios bruscos detectados:")
print(f"        - En 1 hora (>3°C): {cambios_bruscos_1h} eventos")
print(f"        - En 6 horas (>8°C): {cambios_bruscos_6h} eventos")

# 5. OLAS DE CALOR/FRÍO (temperatura sostenida fuera de confort)
# Rolling para detectar períodos prolongados
df['temp_rolling_mean_6h'] = df['temperatura'].rolling(window=6, min_periods=1).mean()
df['temp_rolling_mean_24h'] = df['temperatura'].rolling(window=24, min_periods=1).mean()
df['temp_rolling_max_24h'] = df['temperatura'].rolling(window=24, min_periods=1).max()
df['temp_rolling_min_24h'] = df['temperatura'].rolling(window=24, min_periods=1).min()
df['temp_rolling_std_24h'] = df['temperatura'].rolling(window=24, min_periods=1).std()

# Ola de calor: temperatura media 24h > umbral
df['ola_calor'] = (df['temp_rolling_mean_24h'] > TEMP_CALOR_MODERADO).astype(int)
df['ola_frio'] = (df['temp_rolling_mean_24h'] < TEMP_FRIO_MODERADO).astype(int)

print(f"      • Olas térmicas (temperatura sostenida 24h):")
print(f"        - Ola de calor: {df['ola_calor'].sum()} horas ({df['ola_calor'].mean()*100:.1f}%)")
print(f"        - Ola de frío: {df['ola_frio'].sum()} horas ({df['ola_frio'].mean()*100:.1f}%)")

# 6. Amplitud térmica (diferencia max-min en 24h)
df['temp_amplitud_24h'] = df['temp_rolling_max_24h'] - df['temp_rolling_min_24h']
print(f"      • Amplitud térmica 24h: promedio {df['temp_amplitud_24h'].mean():.1f}°C")

# --- HUMEDAD RELATIVA ---
print("\n   💧 Features de HUMEDAD:")

# LAGs de humedad
df['hr_lag_24h'] = df['humedad_relativa'].shift(24)
df['hr_rolling_mean_24h'] = df['humedad_relativa'].rolling(window=24, min_periods=1).mean()

# Extremos de humedad (afecta sensación térmica)
df['hr_muy_baja'] = (df['humedad_relativa'] < 30).astype(int)  # Aire seco
df['hr_muy_alta'] = (df['humedad_relativa'] > 80).astype(int)  # Aire húmedo

print(f"      • HR muy baja (<30%): {df['hr_muy_baja'].sum()} hrs ({df['hr_muy_baja'].mean()*100:.1f}%)")
print(f"      • HR muy alta (>80%): {df['hr_muy_alta'].sum()} hrs ({df['hr_muy_alta'].mean()*100:.1f}%)")

# --- SENSACIÓN TÉRMICA ---
# Combinación temperatura + humedad (más realista que temp sola)
df['sensacion_termica'] = df['temperatura'] + (df['humedad_relativa'] - 50) * 0.1
print(f"      • Sensación térmica calculada (temp + HR ajustada)")

# --- PRECIPITACIÓN ---
print("\n   🌧️  Features de PRECIPITACIÓN:")

# Acumulados de precipitación
df['precip_acum_6h'] = df['precipitacion'].rolling(window=6, min_periods=1).sum()
df['precip_acum_24h'] = df['precipitacion'].rolling(window=24, min_periods=1).sum()

# Indicadores de lluvia
df['tiene_lluvia'] = (df['precipitacion'] > 0.1).astype(int)  # Lluvia significativa
df['lluvia_intensa'] = (df['precipitacion'] > 5.0).astype(int)  # Lluvia fuerte

horas_lluvia = df['tiene_lluvia'].sum()
print(f"      • Horas con lluvia: {horas_lluvia} ({horas_lluvia/len(df)*100:.1f}%)")
print(f"      • Lluvia intensa (>5mm/hr): {df['lluvia_intensa'].sum()} eventos")

# --- INTERACCIONES CLIMA x TIEMPO ---
print("\n   🔄 Features de INTERACCIÓN:")

# Temperatura extrema en horas punta (más crítico)
# Horas punta residencial: 7-9 AM, 18-22 PM
df['hora'] = df.index.hour
df['temp_extrema_hora_punta'] = (
    ((df['hora'].isin([7,8,9,18,19,20,21,22])) & 
     ((df['temperatura'] > TEMP_CALOR_ALTO) | (df['temperatura'] < TEMP_FRIO_ALTO)))
).astype(int)

# Fin de semana con buen clima (aumenta consumo recreativo)
df['finde_buen_clima'] = (
    (df['is_weekend'] == 1) & 
    (df['temperatura'] >= 18) & 
    (df['temperatura'] <= 28) &
    (df['precipitacion'] < 0.5)
).astype(int)

print(f"      • Temp. extrema en hora punta: {df['temp_extrema_hora_punta'].sum()} hrs")
print(f"      • Fin de semana con buen clima: {df['finde_buen_clima'].sum()} hrs")

print(f"\n   ✅ Total features climáticas creadas: ~40 nuevas features")

# ==================== 4. FILTRAR OUTLIERS ====================
print("\n[4/8] Filtrando outliers en Demanda...")

antes_filtro = len(df)
df = df[
    (df['Demanda_m3_hr'] >= 0) &
    (df['Demanda_m3_hr'] <= 30000)
].copy()
outliers_removidos = antes_filtro - len(df)

print(f"   ✅ Outliers removidos: {outliers_removidos} ({outliers_removidos/antes_filtro*100:.2f}%)")
print(f"   ✅ Datos limpios: {len(df)} registros")

# ==================== 5. PREPARAR FEATURES DE DEMANDA (ORIGINALES) ====================
print("\n[5/8] Preparando features de demanda (LAGs históricos)...")

# Crear LAGs de DEMANDA
df['Demanda_lag_1h'] = df['Demanda_m3_hr'].shift(1)
df['Demanda_lag_2h'] = df['Demanda_m3_hr'].shift(2)
df['Demanda_lag_24h'] = df['Demanda_m3_hr'].shift(24)
df['Demanda_lag_168h'] = df['Demanda_m3_hr'].shift(168)

# Rolling statistics de Demanda
df['Demanda_rolling_mean_6h'] = df['Demanda_m3_hr'].rolling(window=6, min_periods=1).mean()
df['Demanda_rolling_std_6h'] = df['Demanda_m3_hr'].rolling(window=6, min_periods=1).std()
df['Demanda_rolling_mean_24h'] = df['Demanda_m3_hr'].rolling(window=24, min_periods=1).mean()
df['Demanda_rolling_std_24h'] = df['Demanda_m3_hr'].rolling(window=24, min_periods=1).std()

# Diferencias
df['Demanda_diff_1h'] = df['Demanda_m3_hr'].diff()
df['Demanda_diff_24h'] = df['Demanda_m3_hr'].diff(24)

# Ratio
df['Demanda_ratio_vs_24h'] = df['Demanda_m3_hr'] / (df['Demanda_lag_24h'] + 1)

print("   ✅ Features LAG de demanda creadas")

# ==================== 6. DEFINIR FEATURES FINALES ====================
print("\n[6/8] Definiendo lista de features...")

# Features temporales básicas
features_temporales = [
    'hour', 'day_of_week', 'month', 'is_weekend',
    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
    'feriado', 'temporada_turistica_alta'
]

# Features de demanda histórica
features_demanda = [
    'Demanda_lag_1h', 'Demanda_lag_2h', 'Demanda_lag_24h', 'Demanda_lag_168h',
    'Demanda_rolling_mean_6h', 'Demanda_rolling_std_6h',
    'Demanda_rolling_mean_24h', 'Demanda_rolling_std_24h',
    'Demanda_diff_1h', 'Demanda_diff_24h',
    'Demanda_ratio_vs_24h'
]

# Features climáticas AVANZADAS (nuevas)
features_clima = [
    # Temperatura - extremos
    'temp_calor_moderado', 'temp_calor_alto', 'temp_calor_extremo',
    'temp_frio_moderado', 'temp_frio_alto', 'temp_frio_extremo',
    'temp_desviacion_confort',
    # Temperatura - LAGs
    'temp_lag_1h', 'temp_lag_6h', 'temp_lag_24h', 'temp_lag_168h',
    # Temperatura - cambios bruscos (CRÍTICO según experiencia)
    'temp_cambio_1h', 'temp_cambio_6h', 'temp_cambio_24h',
    'temp_cambio_brusco_1h', 'temp_cambio_brusco_6h', 'temp_cambio_brusco_24h',
    # Temperatura - olas térmicas
    'temp_rolling_mean_6h', 'temp_rolling_mean_24h',
    'temp_rolling_max_24h', 'temp_rolling_min_24h', 'temp_rolling_std_24h',
    'ola_calor', 'ola_frio',
    'temp_amplitud_24h',
    # Humedad
    'humedad_relativa', 'hr_lag_24h', 'hr_rolling_mean_24h',
    'hr_muy_baja', 'hr_muy_alta',
    # Sensación térmica
    'sensacion_termica',
    # Precipitación
    'precipitacion', 'precip_acum_6h', 'precip_acum_24h',
    'tiene_lluvia', 'lluvia_intensa',
    # Interacciones
    'temp_extrema_hora_punta', 'finde_buen_clima'
]

# Combinar todas
features_completas = features_temporales + features_demanda + features_clima

print(f"\n   📊 Resumen de features:")
print(f"      • Temporales: {len(features_temporales)}")
print(f"      • Demanda histórica: {len(features_demanda)}")
print(f"      • Clima avanzadas: {len(features_clima)}")
print(f"      • TOTAL: {len(features_completas)} features")

# ==================== 7. SPLIT DE DATOS ====================
print("\n[7/8] Dividiendo datos (70% train, 15% val, 15% test)...")

# Verificar que no hay NaN en features
df_clean = df[features_completas + ['Demanda_m3_hr']].dropna()
nans_eliminados = len(df) - len(df_clean)
print(f"   ✅ Registros sin NaN: {len(df_clean)}")
if nans_eliminados > 0:
    print(f"   ⚠️  {nans_eliminados} registros eliminados por NaN ({nans_eliminados/len(df)*100:.1f}%)")

X = df_clean[features_completas]
y = df_clean['Demanda_m3_hr']

# Split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, shuffle=False
)

# Split temp: 50% val, 50% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, shuffle=False
)

print(f"   ✅ Train: {len(X_train)} registros ({len(X_train)/len(X)*100:.1f}%)")
print(f"   ✅ Val:   {len(X_val)} registros ({len(X_val)/len(X)*100:.1f}%)")
print(f"   ✅ Test:  {len(X_test)} registros ({len(X_test)/len(X)*100:.1f}%)")

# ==================== 8. ENTRENAR MODELO XGBOOST ====================
print("\n[8/8] Entrenando modelo XGBoost con clima avanzado...")

model = xgb.XGBRegressor(
    n_estimators=300,  # Más árboles para capturar complejidad
    max_depth=10,       # Más profundidad para interacciones
    learning_rate=0.05, # Learning rate más bajo para mejor generalización
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("   ✅ Modelo entrenado")

# ==================== 9. EVALUAR MODELO ====================
print("\n" + "="*80)
print("EVALUANDO MODELO")
print("="*80)

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

# ==================== 10. ANALIZAR IMPORTANCIA DE FEATURES ====================
print("\n" + "="*80)
print("IMPORTANCIA DE FEATURES")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': features_completas,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📈 Top 20 features más importantes:")
for idx, row in feature_importance.head(20).iterrows():
    # Marcar features climáticas
    emoji = "🌡️" if row['feature'] in features_clima else "📊"
    print(f"   {emoji} {row['feature']:35} : {row['importance']:.4f}")

# Analizar importancia de features climáticas
clima_importance = feature_importance[feature_importance['feature'].isin(features_clima)]
clima_importance_total = clima_importance['importance'].sum()

print(f"\n🌡️  ANÁLISIS DE FEATURES CLIMÁTICAS:")
print(f"   • Total features clima: {len(features_clima)}")
print(f"   • Importancia acumulada: {clima_importance_total:.4f} ({clima_importance_total*100:.1f}%)")

# Top features climáticas
print(f"\n   📈 Top 10 features CLIMÁTICAS:")
for idx, row in clima_importance.head(10).iterrows():
    print(f"      {row['feature']:35} : {row['importance']:.4f}")

# ==================== 11. COMPARAR CON MODELOS ANTERIORES ====================
print("\n" + "="*80)
print("COMPARACIÓN CON MODELOS ANTERIORES")
print("="*80)

# Cargar métricas modelo sin temperatura
metricas_sin_temp_path = Path('models/demanda/metricas.json')
if metricas_sin_temp_path.exists():
    with open(metricas_sin_temp_path, 'r') as f:
        metricas_sin_temp = json.load(f)
    
    print("\n📊 MODELO ORIGINAL (sin clima):")
    print(f"   R² Test:  {metricas_sin_temp.get('test_r2', 0):.4f}")
    print(f"   MAPE Test: {metricas_sin_temp.get('test_mape', 0):.2f}%")
    print(f"   Features: {metricas_sin_temp.get('n_features', 0)}")

# Cargar métricas modelo con temperatura simple
metricas_temp_simple_path = Path('models/demanda_temperatura/metricas.json')
if metricas_temp_simple_path.exists():
    with open(metricas_temp_simple_path, 'r') as f:
        metricas_temp_simple = json.load(f)
    
    print("\n📊 MODELO CON TEMP SIMPLE:")
    print(f"   R² Test:  {metricas_temp_simple.get('test_r2', 0):.4f}")
    print(f"   MAPE Test: {metricas_temp_simple.get('test_mape', 0):.2f}%")
    print(f"   Features: {metricas_temp_simple.get('n_features', 0)}")
    print(f"   Temp importance: {metricas_temp_simple.get('temperatura_importance', 0):.4f}")

print("\n📊 MODELO NUEVO (clima avanzado):")
print(f"   R² Test:  {test_r2:.4f}")
print(f"   MAPE Test: {test_mape:.2f}%")
print(f"   Features: {len(features_completas)}")
print(f"   Clima importance: {clima_importance_total:.4f} ({clima_importance_total*100:.1f}%)")

if metricas_sin_temp_path.exists():
    mejora_r2 = test_r2 - metricas_sin_temp.get('test_r2', 0)
    mejora_mape = metricas_sin_temp.get('test_mape', 0) - test_mape
    
    print("\n💡 MEJORA vs MODELO ORIGINAL:")
    print(f"   R²:   {mejora_r2:+.4f} {'✅' if mejora_r2 > 0 else '❌'}")
    print(f"   MAPE: {mejora_mape:+.2f}% {'✅' if mejora_mape > 0 else '❌'}")

# ==================== 12. GUARDAR MODELO ====================
print("\n" + "="*80)
print("GUARDANDO MODELO Y MÉTRICAS")
print("="*80)

output_dir = Path('models/demanda_clima_avanzado')
output_dir.mkdir(parents=True, exist_ok=True)

# Guardar modelo
model_path = output_dir / 'demanda_clima_avanzado_xgboost_model.pkl'
joblib.dump(model, model_path)
print(f"\n✅ Modelo guardado: {model_path}")

# Guardar features
features_path = output_dir / 'features.txt'
with open(features_path, 'w') as f:
    for feature in features_completas:
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
    'n_features': len(features_completas),
    'n_features_clima': len(features_clima),
    'clima_importance_total': float(clima_importance_total),
    'features': features_completas,
    'features_clima': features_clima
}

metricas_path = output_dir / 'metricas.json'
with open(metricas_path, 'w') as f:
    json.dump(metricas, f, indent=4)
print(f"✅ Métricas guardadas: {metricas_path}")

# Guardar feature importance
importance_path = output_dir / 'feature_importance.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"✅ Feature importance guardada: {importance_path}")

# Guardar resumen de features climáticas
clima_importance_path = output_dir / 'clima_features_importance.csv'
clima_importance.to_csv(clima_importance_path, index=False)
print(f"✅ Importancia features clima guardada: {clima_importance_path}")

print("\n" + "="*80)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*80)
print(f"\n📁 Archivos generados en: {output_dir}")
print(f"   • demanda_clima_avanzado_xgboost_model.pkl")
print(f"   • features.txt ({len(features_completas)} features)")
print(f"   • metricas.json")
print(f"   • feature_importance.csv")
print(f"   • clima_features_importance.csv")

print(f"\n🌡️  Features climáticas incluidas: {len(features_clima)}")
print(f"   Importancia total: {clima_importance_total:.4f} ({clima_importance_total*100:.1f}%)")

if test_mape < 5.0:
    print(f"\n🎉 EXCELENTE: MAPE Test = {test_mape:.2f}% (< 5%)")
elif test_mape < 10.0:
    print(f"\n✅ BUENO: MAPE Test = {test_mape:.2f}% (< 10%)")
else:
    print(f"\n⚠️  ACEPTABLE: MAPE Test = {test_mape:.2f}%")

print("\n💡 RECOMENDACIÓN:")
print("   Este modelo considera:")
print("   • EXTREMOS térmicos (calor/frío intenso)")
print("   • CAMBIOS BRUSCOS de temperatura")
print("   • OLAS DE CALOR/FRÍO sostenidas")
print("   • Interacciones clima x hora del día")
print("   • Humedad y precipitación")
print("   Basado en experiencia operacional real.")

print("\n" + "="*80)
