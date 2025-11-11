"""
Script de prueba para verificar la función de evaluación testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*80)
print("PRUEBA: Función de Evaluación Testing")
print("="*80)

# Cargar modelo
model_path = Path('models/demanda/demanda_xgboost_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f"\n✅ Modelo cargado")

# Cargar features
features_path = Path('models/demanda/features.txt')
with open(features_path, 'r') as f:
    features = [line.strip() for line in f.readlines()]
print(f"✅ Features cargadas: {len(features)}")

# Cargar datos completos
data_path = Path('data/processed/data_processed_demanda_valid.csv')
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()
print(f"✅ Datos cargados: {len(df)} registros")

# Split temporal (15% final = test)
n = len(df)
val_end = int(n * 0.85)
df_test = df.iloc[val_end:].reset_index(drop=True)
print(f"\n📊 Datos de testing: {len(df_test)} registros ({len(df_test)/n*100:.1f}%)")

# Verificar features
print(f"\n🔍 Verificando features...")
missing = [f for f in features if f not in df_test.columns]
if missing:
    print(f"❌ Features faltantes: {missing}")
else:
    print(f"✅ Todas las features presentes")

# Preparar datos
X_test = df_test[features]
y_test = df_test['Demanda_m3_hr']

print(f"\n📈 Estadísticas de testing:")
print(f"   Demanda media: {y_test.mean():,.0f} m³/hr")
print(f"   Demanda std: {y_test.std():,.0f} m³/hr")
print(f"   Demanda min: {y_test.min():,.0f} m³/hr")
print(f"   Demanda max: {y_test.max():,.0f} m³/hr")

# Predecir
print(f"\n⏳ Realizando predicciones...")
y_pred = model.predict(X_test)

# Calcular métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\n" + "="*80)
print("MÉTRICAS DE CALIDAD")
print("="*80)
print(f"\n📊 Resultados:")
print(f"   R² (Coef. Determinación): {r2:.4f} {'✅ EXCELENTE' if r2 > 0.95 else '⚠️ BUENO' if r2 > 0.90 else '❌ MEJORABLE'}")
print(f"   RMSE (Error Cuadrático):  {rmse:,.0f} m³/hr")
print(f"   MAE (Error Absoluto):     {mae:,.0f} m³/hr")
print(f"   MAPE (Error Porcentual):  {mape:.2f}% {'✅ EXCELENTE' if mape < 5 else '⚠️ BUENO' if mape < 10 else '❌ MEJORABLE'}")

print(f"\n💡 Interpretación:")
print(f"   - El modelo explica el {r2*100:.2f}% de la variabilidad")
print(f"   - Error típico: {rmse:,.0f} m³/hr ({(rmse/y_test.mean())*100:.2f}% de la media)")
print(f"   - Desviación promedio: {mae:,.0f} m³/hr ({(mae/y_test.mean())*100:.2f}% de la media)")

print(f"\n" + "="*80)
print("✅ PRUEBA COMPLETADA EXITOSAMENTE")
print("="*80)
