"""
Análisis detallado de por qué las predicciones son bajas el 3-4 de julio 2025
"""

import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filtrar periodo TEST
n = len(df)
val_end = int(n * 0.85)
df_test = df.iloc[val_end:].copy()

# Filtrar 3 y 4 de julio 2025
df_julio = df_test[(df_test['timestamp'] >= '2025-07-03') & 
                    (df_test['timestamp'] <= '2025-07-04 23:59:59')].copy()

print('='*80)
print('ANÁLISIS: 3-4 JULIO 2025 - ¿Por qué predicciones bajas?')
print('='*80)

print(f'\n📅 Periodo analizado: {df_julio["timestamp"].min()} a {df_julio["timestamp"].max()}')
print(f'📊 Total registros: {len(df_julio)}')

# Estadísticas de Q_net real
print('\n' + '='*80)
print('1️⃣ ANÁLISIS DE Q_NET REAL')
print('='*80)
print(f'Q_net medio: {df_julio["Q_net_m3h"].mean():,.1f} m³/h')
print(f'Q_net std: {df_julio["Q_net_m3h"].std():,.1f} m³/h')
print(f'Q_net min: {df_julio["Q_net_m3h"].min():,.1f} m³/h')
print(f'Q_net max: {df_julio["Q_net_m3h"].max():,.1f} m³/h')

# Comparar con promedio del test set completo
q_net_promedio_test = df_test['Q_net_m3h'].mean()
print(f'\n📊 Comparación:')
print(f'   Q_net promedio TEST completo: {q_net_promedio_test:,.1f} m³/h')
print(f'   Q_net promedio 3-4 julio:     {df_julio["Q_net_m3h"].mean():,.1f} m³/h')
diferencia = df_julio["Q_net_m3h"].mean() - q_net_promedio_test
pct = ((df_julio["Q_net_m3h"].mean() / q_net_promedio_test) - 1) * 100
print(f'   Diferencia: {diferencia:,.1f} m³/h ({pct:.1f}%)')

# Verificar features climáticas
print('\n' + '='*80)
print('2️⃣ CONDICIONES CLIMÁTICAS (3-4 JULIO)')
print('='*80)

if 'clima_temp_c' in df_julio.columns:
    temp = df_julio['clima_temp_c']
    print(f'🌡️  Temperatura:')
    print(f'   Promedio: {temp.mean():.1f}°C')
    print(f'   Mín-Máx: {temp.min():.1f}°C - {temp.max():.1f}°C')
    
    temp_test = df_test['clima_temp_c'].mean()
    print(f'   Promedio TEST: {temp_test:.1f}°C')
    print(f'   Diferencia: {temp.mean() - temp_test:+.1f}°C')
    
if 'clima_HR_pct' in df_julio.columns:
    hr = df_julio['clima_HR_pct']
    print(f'\n💧 Humedad:')
    print(f'   Promedio: {hr.mean():.1f}%')
    print(f'   Mín-Máx: {hr.min():.1f}% - {hr.max():.1f}%')

if 'clima_lluvia_mmhr' in df_julio.columns:
    lluvia = df_julio['clima_lluvia_mmhr']
    total_lluvia = lluvia.sum()
    print(f'\n🌧️  Lluvia:')
    print(f'   Total acumulada: {total_lluvia:.1f} mm')
    print(f'   Máxima horaria: {lluvia.max():.1f} mm/h')
    if total_lluvia > 0:
        print(f'   ⚠️  HAY PRECIPITACIONES en este periodo')

# Verificar día de la semana
print('\n' + '='*80)
print('3️⃣ CARACTERÍSTICAS TEMPORALES')
print('='*80)

df_julio['dia_semana'] = df_julio['timestamp'].dt.day_name()
df_julio['es_finde'] = df_julio['timestamp'].dt.dayofweek >= 5

print(f'📆 Días de la semana:')
for dia in df_julio['dia_semana'].unique():
    print(f'   • {dia}')

if df_julio['es_finde'].any():
    print(f'\n🏖️  INCLUYE FIN DE SEMANA')
    print(f'   Horas de fin de semana: {df_julio["es_finde"].sum()} de {len(df_julio)}')

# Verificar si es feriado
if 'es_feriado' in df_julio.columns:
    if df_julio['es_feriado'].any():
        print(f'\n🎉 ES FERIADO')
        print(f'   Horas marcadas como feriado: {df_julio["es_feriado"].sum()}')

# Análisis de LAG features
print('\n' + '='*80)
print('4️⃣ ANÁLISIS DE LAG FEATURES (Historial)')
print('='*80)

lag_cols = [col for col in df_julio.columns if '_lag_' in col]
if lag_cols:
    print(f'Features LAG disponibles: {len(lag_cols)}')
    
    if 'clima_temp_c__lag_168h' in df_julio.columns:
        temp_lag = df_julio['clima_temp_c__lag_168h'].mean()
        temp_actual = df_julio['clima_temp_c'].mean()
        print(f'\n🌡️  Temperatura hace 7 días (LAG 168h):')
        print(f'   LAG 168h: {temp_lag:.1f}°C')
        print(f'   Actual: {temp_actual:.1f}°C')
        print(f'   Cambio: {temp_actual - temp_lag:+.1f}°C')

# Verificar valores extremos
print('\n' + '='*80)
print('5️⃣ VALORES EXTREMOS DE Q_NET')
print('='*80)

top_descargas = df_julio.nsmallest(5, 'Q_net_m3h')
print('\n📉 Top 5 horas con MAYOR descarga (Q_net más negativo):')
for idx, row in top_descargas.iterrows():
    print(f'   {row["timestamp"]}: Q_net = {row["Q_net_m3h"]:,.0f} m³/h')

top_recargas = df_julio.nlargest(5, 'Q_net_m3h')
print('\n📈 Top 5 horas con MENOR descarga (Q_net menos negativo):')
for idx, row in top_recargas.iterrows():
    print(f'   {row["timestamp"]}: Q_net = {row["Q_net_m3h"]:,.0f} m³/h')

# CONCLUSIÓN
print('\n' + '='*80)
print('CONCLUSIÓN')
print('='*80)

causas = []

if df_julio['Q_net_m3h'].mean() < (q_net_promedio_test - 2000):
    causas.append('Q_net significativamente más negativo que promedio (MAYOR descarga)')

if 'clima_temp_c' in df_julio.columns:
    if df_julio['clima_temp_c'].mean() > (df_test['clima_temp_c'].mean() + 2):
        causas.append('Temperaturas más altas de lo normal (verano)')

if df_julio['es_finde'].any():
    causas.append('Incluye fin de semana (patrones de consumo diferentes)')

if 'clima_lluvia_mmhr' in df_julio.columns and df_julio['clima_lluvia_mmhr'].sum() > 5:
    causas.append('Precipitaciones detectadas')

print('\n🔍 Posibles causas de la sub-predicción:')
if len(causas) > 0:
    for i, causa in enumerate(causas, 1):
        print(f'   {i}. {causa}')
else:
    print('   Sin causas obvias detectadas')

print('\n💡 Explicación:')
print('   Los modelos ML aprenden patrones promedio del entrenamiento.')
print('   Si este periodo tiene características únicas o extremas que')
print('   no estaban bien representadas en el train set (ej: temperaturas')
print('   muy altas, fin de semana de verano), los modelos tenderán a')
print('   predecir valores más cercanos al promedio aprendido.')
print('\n   Esto resulta en SUB-PREDICCIÓN cuando la demanda real es')
print('   anormalmente alta.')

print('\n' + '='*80)
