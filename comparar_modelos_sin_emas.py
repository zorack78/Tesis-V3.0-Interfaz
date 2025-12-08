"""
Comparar predicciones: CON EMAs vs SIN EMAs
============================================
Genera gráfico comparativo para ver impacto de eliminar EMAs de corto plazo.
"""

import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from pathlib import Path

print("="*80)
print("COMPARACIÓN: MODELOS CON vs SIN EMAs")
print("="*80)

# Cargar dataset
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').copy()

# Agregar features categóricas e interacciones
def agregar_features(df):
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
    
    df['regimen_equilibrio'] = (df['Q_net_m3h'].abs() < 500).astype(int)
    df['es_hora_bisagra'] = df['hora'].isin([6,7,8,9,10,12,18,19,20]).astype(int)
    
    # Interacciones
    df['temp_x_hora'] = df['clima_temp_c'] * df['hora']
    df['delta_temp_6h_x_hora'] = df['clima_temp_delta_6h'] * df['hora']
    
    df['dia_semana'] = df['timestamp'].dt.dayofweek
    df['es_fin_de_semana'] = (df['dia_semana'] >= 5).astype(int)
    df['temp_x_finde'] = df['clima_temp_c'] * df['es_fin_de_semana']
    
    return df

df = agregar_features(df)

# Split
n = len(df)
val_end = int(n * 0.85)
df_test = df.iloc[val_end:].copy()

# Cargar features
with open('models/forecasting/features.txt') as f:
    features_con_emas = [l.strip() for l in f.readlines()]

with open('models/forecasting_sin_emas/features.txt') as f:
    features_sin_emas = [l.strip() for l in f.readlines()]

# Cargar modelos
print("\n📦 Cargando modelos...")
modelo_con_emas = joblib.load('models/forecasting/modelo_forecasting_xgboost.pkl')
modelo_sin_emas = joblib.load('models/forecasting_sin_emas/modelo_xgboost_sin_emas.pkl')

# Predecir
X_con = df_test[features_con_emas]
X_sin = df_test[features_sin_emas]

y_pred_con = modelo_con_emas.predict(X_con)
y_pred_sin = modelo_sin_emas.predict(X_sin)
y_real = df_test['Q_net_m3h'].values

# Calcular métricas
from sklearn.metrics import mean_absolute_error, r2_score

mae_con = mean_absolute_error(y_real, y_pred_con)
r2_con = r2_score(y_real, y_pred_con)

mae_sin = mean_absolute_error(y_real, y_pred_sin)
r2_sin = r2_score(y_real, y_pred_sin)

print(f"\n{'='*80}")
print("MÉTRICAS EN TEST SET")
print(f"{'='*80}")
print(f"\nCON EMAs (6h, 12h, 24h):")
print(f"  R² = {r2_con:.4f}")
print(f"  MAE = {mae_con:,.0f} m³/hr")

print(f"\nSIN EMAs (solo lag_168h):")
print(f"  R² = {r2_sin:.4f}")
print(f"  MAE = {mae_sin:,.0f} m³/hr")

print(f"\n{'='*80}")
print(f"DIFERENCIA:")
print(f"  ΔR² = {r2_sin - r2_con:+.4f}")
print(f"  ΔMAE = {mae_sin - mae_con:+,.0f} m³/hr")
print(f"{'='*80}")

# Graficar últimos 7 días
ultimos_7d = df_test.tail(7*24)
y_real_7d = ultimos_7d['Q_net_m3h'].values
y_con_7d = y_pred_con[-7*24:]
y_sin_7d = y_pred_sin[-7*24:]
timestamps_7d = ultimos_7d['timestamp']

fig = go.Figure()

# Real
fig.add_trace(go.Scatter(
    x=timestamps_7d,
    y=y_real_7d,
    name='Real',
    line=dict(color='black', width=2),
    mode='lines'
))

# CON EMAs
fig.add_trace(go.Scatter(
    x=timestamps_7d,
    y=y_con_7d,
    name=f'CON EMAs (MAE={mae_con:.0f})',
    line=dict(color='red', width=1.5, dash='dash'),
    mode='lines'
))

# SIN EMAs
fig.add_trace(go.Scatter(
    x=timestamps_7d,
    y=y_sin_7d,
    name=f'SIN EMAs (MAE={mae_sin:.0f})',
    line=dict(color='blue', width=1.5, dash='dot'),
    mode='lines'
))

fig.update_layout(
    title=f"Comparación: CON vs SIN EMAs de Corto Plazo (últimos 7 días)<br>ΔMAE = {mae_sin - mae_con:+.0f} m³/hr",
    xaxis_title="Fecha",
    yaxis_title="Q_net (m³/hr)",
    hovermode='x unified',
    width=1400,
    height=600
)

output_path = 'outputs/figures/comparacion_con_sin_emas.html'
Path('outputs/figures').mkdir(parents=True, exist_ok=True)
fig.write_html(output_path)

print(f"\n✅ Gráfico guardado en: {output_path}")

# Análisis del peak del 25 de septiembre
peak_date = pd.to_datetime('2025-09-25 08:00:00')
if peak_date in ultimos_7d['timestamp'].values:
    idx_peak = ultimos_7d[ultimos_7d['timestamp'] == peak_date].index[0]
    idx_local = len(ultimos_7d) - len(y_real_7d) + list(ultimos_7d.index).index(idx_peak)
    
    real_peak = y_real_7d[idx_local]
    pred_con_peak = y_con_7d[idx_local]
    pred_sin_peak = y_sin_7d[idx_local]
    
    print(f"\n{'='*80}")
    print(f"ANÁLISIS DEL PEAK ANÓMALO (Sep 25, 08:00)")
    print(f"{'='*80}")
    print(f"Q_net Real:        {real_peak:>10,.0f} m³/hr")
    print(f"Pred CON EMAs:     {pred_con_peak:>10,.0f} m³/hr  (error: {abs(real_peak - pred_con_peak):>7,.0f})")
    print(f"Pred SIN EMAs:     {pred_sin_peak:>10,.0f} m³/hr  (error: {abs(real_peak - pred_sin_peak):>7,.0f})")
    print(f"\n¿Modelo SIN EMAs es más conservador en peak? {abs(real_peak - pred_sin_peak) < abs(real_peak - pred_con_peak)}")
    print(f"{'='*80}")

print("\n✅ Análisis completo")
