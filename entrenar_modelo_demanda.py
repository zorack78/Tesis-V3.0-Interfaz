"""
Pipeline de entrenamiento para predecir DEMANDA_m3_hr

Similar al pipeline original pero:
- Variable objetivo: Demanda_m3_hr (no Volumen_Total_m3)
- Manejo de outliers
- Validación de patrones esperados (pico mediodía, valle madrugada)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("ENTRENAMIENTO: MODELO DE PREDICCIÓN DE DEMANDA")
print("="*80)

# 1. Cargar datos con Demanda
data_path = Path('data/processed/data_processed_demanda_valid.csv')
print(f"\n📂 Cargando: {data_path}")

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
print(f"✅ Datos cargados: {len(df)} registros")

# 2. Análisis de outliers
print("\n" + "="*80)
print("ANÁLISIS Y MANEJO DE OUTLIERS")
print("="*80)

# Calcular límites IQR
Q1 = df['Demanda_m3_hr'].quantile(0.25)
Q3 = df['Demanda_m3_hr'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 3 * IQR  # 3 IQR para ser conservadores
limite_superior = Q3 + 3 * IQR

print(f"\n📊 Rangos de Demanda:")
print(f"   Q1 (P25): {Q1:,.0f} m³/hr")
print(f"   Q3 (P75): {Q3:,.0f} m³/hr")
print(f"   IQR: {IQR:,.0f} m³/hr")
print(f"   Límite inferior: {limite_inferior:,.0f} m³/hr")
print(f"   Límite superior: {limite_superior:,.0f} m³/hr")

# Identificar outliers
outliers = df[(df['Demanda_m3_hr'] < limite_inferior) | 
              (df['Demanda_m3_hr'] > limite_superior)]
print(f"\n⚠️  Outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

# Opción: Remover outliers o mantenerlos
# Por ahora los mantenemos pero los registramos
print(f"   Decisión: MANTENER outliers para el entrenamiento")
print(f"   Razón: Representan eventos reales del sistema")

# 3. Verificar datos necesarios
print("\n" + "="*80)
print("VERIFICACIÓN DE DATOS")
print("="*80)

required_features = [
    'hour', 'day_of_week', 'month', 'is_weekend',
    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
    'feriado', 'temporada_turistica_alta'
]

missing_features = [f for f in required_features if f not in df.columns]
if missing_features:
    print(f"❌ Features faltantes: {missing_features}")
    exit(1)

print(f"✅ Todas las features requeridas están presentes")

# 4. Crear LAGs de DEMANDA (no de volumen)
print("\n📐 Creando LAGs de Demanda...")

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

print(f"✅ Features LAG creadas")

# 5. Remover filas con NaN en features críticas
print("\n🧹 Limpiando datos...")
df_clean = df.dropna(subset=[
    'Demanda_m3_hr',
    'Demanda_lag_1h', 'Demanda_lag_24h', 'Demanda_lag_168h',
    'Demanda_rolling_mean_24h', 'Demanda_rolling_std_24h'
]).reset_index(drop=True)

print(f"✅ Datos limpios: {len(df_clean)} registros (perdidos: {len(df) - len(df_clean)})")

# 6. Seleccionar features para el modelo
print("\n" + "="*80)
print("SELECCIÓN DE FEATURES")
print("="*80)

feature_cols = [
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

# Verificar que todas existen
missing = [f for f in feature_cols if f not in df_clean.columns]
if missing:
    print(f"❌ Features faltantes: {missing}")
    exit(1)

print(f"✅ Features seleccionadas: {len(feature_cols)}")
for i, f in enumerate(feature_cols, 1):
    print(f"   {i:>2}. {f}")

# 7. Preparar X, y
X = df_clean[feature_cols]
y = df_clean['Demanda_m3_hr']

print(f"\n📊 Dataset preparado:")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   y media: {y.mean():,.0f} m³/hr")
print(f"   y std: {y.std():,.0f} m³/hr")

# 8. Split temporal
print("\n" + "="*80)
print("DIVISIÓN DE DATOS (TEMPORAL)")
print("="*80)

# Usar 70% train, 15% val, 15% test (temporal)
n = len(df_clean)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]
X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print(f"✅ División completada:")
print(f"   Train: {len(X_train)} registros ({len(X_train)/n*100:.1f}%)")
print(f"   Val:   {len(X_val)} registros ({len(X_val)/n*100:.1f}%)")
print(f"   Test:  {len(X_test)} registros ({len(X_test)/n*100:.1f}%)")

# 9. Entrenar XGBoost
print("\n" + "="*80)
print("ENTRENAMIENTO DEL MODELO")
print("="*80)

print(f"\n⏳ Entrenando XGBoost...")

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)

print(f"✅ Modelo entrenado")

# 10. Evaluación
print("\n" + "="*80)
print("EVALUACIÓN DEL MODELO")
print("="*80)

# Predicciones
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Métricas por conjunto
for name, y_true, y_pred in [
    ('Train', y_train, y_train_pred),
    ('Validación', y_val, y_val_pred),
    ('Test', y_test, y_test_pred)
]:
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    
    print(f"\n📊 {name}:")
    print(f"   R²:   {r2:.4f}")
    print(f"   RMSE: {rmse:,.0f} m³/hr")
    print(f"   MAE:  {mae:,.0f} m³/hr")
    print(f"   MAPE: {mape:.2f}%")

# 11. Feature Importance
print("\n" + "="*80)
print("IMPORTANCIA DE FEATURES")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔝 Top 10 Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:.<40} {row['importance']:.4f}")

# 12. Validación de patrones
print("\n" + "="*80)
print("VALIDACIÓN DE PATRONES ESPERADOS")
print("="*80)

# Usar datos de test
df_test = df_clean.iloc[val_end:].copy()
df_test['Demanda_pred'] = y_test_pred

# Promedio por hora
patron_hora = df_test.groupby('hour').agg({
    'Demanda_m3_hr': 'mean',
    'Demanda_pred': 'mean'
}).round(0)

hora_pico_real = patron_hora['Demanda_m3_hr'].idxmax()
hora_pico_pred = patron_hora['Demanda_pred'].idxmax()
hora_valle_real = patron_hora['Demanda_m3_hr'].idxmin()
hora_valle_pred = patron_hora['Demanda_pred'].idxmin()

print(f"\n🔍 Patrones por hora del día:")
print(f"   HORARIO DE INFLEXIÓN MÁXIMO real:      {hora_pico_real}:00 → {patron_hora.loc[hora_pico_real, 'Demanda_m3_hr']:,.0f} m³/hr")
print(f"   HORARIO DE INFLEXIÓN MÁXIMO predicha:  {hora_pico_pred}:00 → {patron_hora.loc[hora_pico_pred, 'Demanda_pred']:,.0f} m³/hr")
print(f"   {'✅ CORRECTO' if 11 <= hora_pico_pred <= 14 else '❌ INCORRECTO'} (esperado: 11-14)")
print()
print(f"   HORARIO DE INFLEXIÓN MÍNIMO real:     {hora_valle_real}:00 → {patron_hora.loc[hora_valle_real, 'Demanda_m3_hr']:,.0f} m³/hr")
print(f"   HORARIO DE INFLEXIÓN MÍNIMO predicha: {hora_valle_pred}:00 → {patron_hora.loc[hora_valle_pred, 'Demanda_pred']:,.0f} m³/hr")
print(f"   {'✅ CORRECTO' if 3 <= hora_valle_pred <= 5 else '❌ INCORRECTO'} (esperado: 3-5)")

# 13. Guardar modelo
print("\n" + "="*80)
print("GUARDANDO MODELO")
print("="*80)

output_dir = Path('models/demanda/')
output_dir.mkdir(parents=True, exist_ok=True)

# Guardar modelo
model_path = output_dir / 'demanda_xgboost_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✅ Modelo guardado: {model_path}")

# Guardar features
features_path = output_dir / 'features.txt'
with open(features_path, 'w') as f:
    for feat in feature_cols:
        f.write(f"{feat}\n")
print(f"✅ Features guardadas: {features_path}")

# Guardar métricas
metricas = {
    'modelo': 'XGBoost',
    'variable_objetivo': 'Demanda_m3_hr',
    'n_features': len(feature_cols),
    'train': {
        'n': int(len(X_train)),
        'r2': float(r2_score(y_train, y_train_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        'mae': float(mean_absolute_error(y_train, y_train_pred)),
        'mape': float(np.mean(np.abs((y_train - y_train_pred) / (y_train + 1))) * 100)
    },
    'val': {
        'n': int(len(X_val)),
        'r2': float(r2_score(y_val, y_val_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
        'mae': float(mean_absolute_error(y_val, y_val_pred)),
        'mape': float(np.mean(np.abs((y_val - y_val_pred) / (y_val + 1))) * 100)
    },
    'test': {
        'n': int(len(X_test)),
        'r2': float(r2_score(y_test, y_test_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        'mae': float(mean_absolute_error(y_test, y_test_pred)),
        'mape': float(np.mean(np.abs((y_test - y_test_pred) / (y_test + 1))) * 100)
    },
    'patrones_validados': {
        'hora_pico_predicha': int(hora_pico_pred),
        'hora_valle_predicha': int(hora_valle_pred),
        'pico_correcto': bool(11 <= hora_pico_pred <= 14),
        'valle_correcto': bool(3 <= hora_valle_pred <= 5)
    },
    'top_features': feature_importance.head(10).to_dict('records')
}

metricas_path = output_dir / 'metricas.json'
with open(metricas_path, 'w', encoding='utf-8') as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)
print(f"✅ Métricas guardadas: {metricas_path}")

print("\n" + "="*80)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*80)
print(f"""
RESUMEN FINAL:
• Modelo: XGBoost
• Variable: Demanda_m3_hr
• Features: {len(feature_cols)}
• Test R²: {r2_score(y_test, y_test_pred):.4f}
• Test MAPE: {np.mean(np.abs((y_test - y_test_pred) / (y_test + 1))) * 100:.2f}%
• Horario de inflexión máximo: {hora_pico_pred}:00 {'✅' if 11 <= hora_pico_pred <= 14 else '❌'}
• Horario de inflexión mínimo: {hora_valle_pred}:00 {'✅' if 3 <= hora_valle_pred <= 5 else '❌'}

Archivos generados:
• {model_path}
• {features_path}
• {metricas_path}
""")
