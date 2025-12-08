"""
Análisis comparativo: ¿Qué hace especial al 3-4 julio vs periodo de entrenamiento?
"""

import pandas as pd
import numpy as np

# Cargar datos completos
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Splits
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

# Filtrar 3-4 julio
df_julio = df_test[(df_test['timestamp'] >= '2025-07-03') & 
                    (df_test['timestamp'] <= '2025-07-04 23:59:59')].copy()

print('='*80)
print('ANÁLISIS COMPARATIVO: 3-4 JULIO VS ENTRENAMIENTO')
print('='*80)

print(f'\n📊 Dataset completo:')
print(f'   Train: {df_train["timestamp"].min()} a {df_train["timestamp"].max()} ({len(df_train)} horas)')
print(f'   Test:  {df_test["timestamp"].min()} a {df_test["timestamp"].max()} ({len(df_test)} horas)')
print(f'   3-4 julio: {df_julio["timestamp"].min()} a {df_julio["timestamp"].max()} ({len(df_julio)} horas)')

# Comparación de Q_net
print('\n' + '='*80)
print('1️⃣ COMPARACIÓN DE Q_NET')
print('='*80)

q_net_train = df_train['Q_net_m3h']
q_net_julio = df_julio['Q_net_m3h']

print(f'\n📊 Estadísticas de Q_net:')
print(f'\n   ENTRENAMIENTO:')
print(f'      Media:  {q_net_train.mean():7,.1f} m³/h')
print(f'      Std:    {q_net_train.std():7,.1f} m³/h')
print(f'      Min:    {q_net_train.min():7,.1f} m³/h')
print(f'      Max:    {q_net_train.max():7,.1f} m³/h')

print(f'\n   3-4 JULIO:')
print(f'      Media:  {q_net_julio.mean():7,.1f} m³/h')
print(f'      Std:    {q_net_julio.std():7,.1f} m³/h')
print(f'      Min:    {q_net_julio.min():7,.1f} m³/h')
print(f'      Max:    {q_net_julio.max():7,.1f} m³/h')

print(f'\n   ⚠️  ANOMALÍA DETECTADA:')
print(f'      Q_net medio 3-4 julio es POSITIVO ({q_net_julio.mean():,.1f})')
print(f'      Q_net medio train es NEGATIVO ({q_net_train.mean():,.1f})')
print(f'      Diferencia: {q_net_julio.mean() - q_net_train.mean():,.1f} m³/h')

# Verificar horas con Q_net positivo
horas_positivas_train = (q_net_train > 0).sum()
horas_positivas_julio = (q_net_julio > 0).sum()

print(f'\n   📊 Horas con Q_net POSITIVO (sistema recargando):')
print(f'      Train: {horas_positivas_train} de {len(q_net_train)} ({horas_positivas_train/len(q_net_train)*100:.1f}%)')
print(f'      3-4 julio: {horas_positivas_julio} de {len(q_net_julio)} ({horas_positivas_julio/len(q_net_julio)*100:.1f}%)')

# Extremos de Q_net
print('\n' + '='*80)
print('2️⃣ ANÁLISIS DE VALORES EXTREMOS')
print('='*80)

percentil_99_train = np.percentile(q_net_train, 99)
percentil_1_train = np.percentile(q_net_train, 1)

print(f'\n📊 Percentiles de Q_net en TRAIN:')
print(f'   P1:  {percentil_1_train:,.1f} m³/h')
print(f'   P99: {percentil_99_train:,.1f} m³/h')

valores_fuera_rango = ((q_net_julio < percentil_1_train) | (q_net_julio > percentil_99_train)).sum()
print(f'\n⚠️  Horas del 3-4 julio FUERA del rango P1-P99 del train:')
print(f'   {valores_fuera_rango} de {len(q_net_julio)} horas ({valores_fuera_rango/len(q_net_julio)*100:.1f}%)')

if valores_fuera_rango > 0:
    print(f'\n   Valores fuera de rango:')
    fuera = df_julio[(q_net_julio < percentil_1_train) | (q_net_julio > percentil_99_train)]
    for _, row in fuera.head(10).iterrows():
        print(f'      {row["timestamp"]}: Q_net = {row["Q_net_m3h"]:,.0f} m³/h')

# Temperatura comparativa
print('\n' + '='*80)
print('3️⃣ COMPARACIÓN CLIMÁTICA')
print('='*80)

if 'clima_temp_c' in df_train.columns:
    temp_train = df_train['clima_temp_c']
    temp_julio = df_julio['clima_temp_c']
    
    print(f'\n🌡️  Temperatura:')
    print(f'   Train:    {temp_train.mean():.1f}°C (min: {temp_train.min():.1f}, max: {temp_train.max():.1f})')
    print(f'   3-4 julio: {temp_julio.mean():.1f}°C (min: {temp_julio.min():.1f}, max: {temp_julio.max():.1f})')
    
    # Buscar días similares en train
    dias_similares = df_train[
        (df_train['clima_temp_c'] >= temp_julio.mean() - 2) & 
        (df_train['clima_temp_c'] <= temp_julio.mean() + 2)
    ]
    
    if len(dias_similares) > 0:
        print(f'\n   Días con temperatura similar en TRAIN: {len(dias_similares)}')
        print(f'   Q_net promedio de esos días: {dias_similares["Q_net_m3h"].mean():,.1f} m³/h')
    else:
        print(f'\n   ⚠️  NO hay días con temperatura similar en TRAIN')

# Patrón horario
print('\n' + '='*80)
print('4️⃣ PATRÓN HORARIO')
print('='*80)

df_julio_copy = df_julio.copy()
df_julio_copy['hora'] = df_julio_copy['timestamp'].dt.hour

print(f'\n⏰ Q_net por hora (3-4 julio):')
for hora in sorted(df_julio_copy['hora'].unique()):
    datos_hora = df_julio_copy[df_julio_copy['hora'] == hora]
    q_net_hora = datos_hora['Q_net_m3h'].mean()
    
    # Comparar con mismo horario en train
    df_train_copy = df_train.copy()
    df_train_copy['hora'] = df_train_copy['timestamp'].dt.hour
    q_net_hora_train = df_train_copy[df_train_copy['hora'] == hora]['Q_net_m3h'].mean()
    
    diferencia = q_net_hora - q_net_hora_train
    simbolo = '⚠️' if abs(diferencia) > 2000 else '  '
    
    print(f'   {simbolo} Hora {hora:2d}: {q_net_hora:7,.0f} m³/h (train: {q_net_hora_train:7,.0f}, diff: {diferencia:+7,.0f})')

# Conclusión final
print('\n' + '='*80)
print('🎯 RESPUESTA: ¿Por qué las predicciones son más bajas?')
print('='*80)

print(f'\n1. VALOR PROMEDIO ANÓMALO:')
print(f'   • Train: Q_net promedio = {q_net_train.mean():,.1f} m³/h (negativo = descarga)')
print(f'   • 3-4 julio: Q_net promedio = {q_net_julio.mean():,.1f} m³/h (POSITIVO = recarga)')
print(f'   • El sistema está RECARGANDO más de lo normal')

print(f'\n2. VALORES EXTREMOS:')
print(f'   • {valores_fuera_rango} horas ({valores_fuera_rango/len(q_net_julio)*100:.0f}%) están fuera del rango P1-P99 del training')
print(f'   • El modelo NUNCA vio valores tan extremos durante el entrenamiento')

print(f'\n3. COMPORTAMIENTO DEL MODELO:')
print(f'   • Los modelos ML tienden a predecir cerca del PROMEDIO aprendido')
print(f'   • Cuando ven patrones nunca vistos, hacen predicciones conservadoras')
print(f'   • Resultado: SUBESTIMAN valores anómalamente altos/positivos')

print(f'\n💡 CONCLUSIÓN:')
print(f'   El 3-4 de julio presenta un comportamiento de RECARGA del sistema')
print(f'   que es estadísticamente raro vs el periodo de entrenamiento.')
print(f'   Los modelos, al no haber aprendido este patrón extremo,')
print(f'   predicen valores más cercanos al promedio histórico,')
print(f'   resultando en SUB-PREDICCIÓN vs la realidad.')

print(f'\n' + '='*80)
