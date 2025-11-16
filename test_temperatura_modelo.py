"""
Script para probar cómo el modelo V3.0 responde a cambios de temperatura
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Cargar modelo
model_path = Path('models/forecasting/modelo_forecasting_xgboost.pkl')
with open(model_path, 'rb') as f:
    modelo = pickle.load(f)

# Cargar features y umbrales desde sistema_info
with open('data/processed/sistema_info_v3.json', 'r') as f:
    sistema_info = json.load(f)
    features = sistema_info['modelo']['features']
    umbrales = sistema_info['umbrales']

print(f"✅ Modelo cargado: {len(features)} features")

# Cargar dataset completo
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"✅ Dataset: {len(df)} registros")

# Tomar una muestra base (hora 14:00, día laboral)
muestra_base = df[
    (df['timestamp'].dt.hour == 14) &
    (df['timestamp'].dt.weekday < 5)
].iloc[0].copy()

print(f"\n📊 Muestra base (hora 14:00):")
print(f"   Temperatura original: {muestra_base.get('clima_temp_c__mean_win_12h', 'N/A'):.1f}°C")

# Predecir con diferentes temperaturas
temperaturas = [15, 18, 20, 25, 28, 33]
resultados = []

for temp in temperaturas:
    # Modificar features de temperatura
    muestra = muestra_base.copy()
    muestra['clima_temp_c__max_win_12h'] = temp + 2
    muestra['clima_temp_c__mean_win_12h'] = temp
    muestra['clima_temp_c__std_win_12h'] = 1.5
    muestra['clima_temp_c__lag_6h'] = temp - 0.5
    muestra['clima_temp_delta_1h'] = 0.2
    muestra['clima_temp_delta_3h'] = 0.3
    muestra['clima_temp_delta_6h'] = 0.5
    muestra['clima_temp_delta_12h'] = 0.8
    
    # Predecir
    X = pd.DataFrame([muestra])[features]
    q_net = modelo.predict(X)[0]
    
    # Calcular demanda
    demanda = abs(q_net) if q_net < 0 else 0
    
    resultados.append({
        'temperatura': temp,
        'q_net': q_net,
        'demanda': demanda
    })
    
    print(f"\n🌡️ Temp {temp}°C:")
    print(f"   Q_net predicho: {q_net:,.0f} m³/hr")
    print(f"   Demanda: {demanda:,.0f} m³/hr")

# Análisis
print("\n" + "="*60)
print("📈 ANÁLISIS DE SENSIBILIDAD A TEMPERATURA")
print("="*60)

df_res = pd.DataFrame(resultados)
print(df_res.to_string(index=False))

# Correlación
corr = df_res['temperatura'].corr(df_res['demanda'])
print(f"\n📊 Correlación Temperatura-Demanda: {corr:.3f}")

if corr > 0.1:
    print("✅ CORRECTO: Mayor temperatura → Mayor demanda")
elif corr < -0.1:
    print("❌ INVERTIDO: Mayor temperatura → Menor demanda")
else:
    print("⚠️ NEUTRAL: Temperatura no influye significativamente")
