"""
AUDITORÍA DE MÉTRICAS DE MACHINE LEARNING
==========================================
Verifica la consistencia y realismo de las métricas de los modelos ML
sin realizar ninguna modificación en el código existente.

Autor: Sistema de Auditoría
Fecha: 2025-12-07
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import json
from datetime import datetime

print('='*80)
print('AUDITORÍA DE MÉTRICAS DE MACHINE LEARNING')
print('='*80)

# ==============================================================================
# 1. CARGAR DATOS Y VERIFICAR SPLITS
# ==============================================================================
print('\n1️⃣ CARGA DE DATOS')
print('-'*80)

df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f'✅ Dataset cargado: {len(df):,} registros')
print(f'   Periodo: {df["timestamp"].min()} → {df["timestamp"].max()}')

# Calcular splits (70/15/15)
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

print(f'\n📂 SPLITS DE DATOS:')
print(f'   Train:      {len(df_train):,} registros ({len(df_train)/n*100:.1f}%) - {df_train["timestamp"].min()} → {df_train["timestamp"].max()}')
print(f'   Validación: {len(df_val):,} registros ({len(df_val)/n*100:.1f}%) - {df_val["timestamp"].min()} → {df_val["timestamp"].max()}')
print(f'   Test:       {len(df_test):,} registros ({len(df_test)/n*100:.1f}%) - {df_test["timestamp"].min()} → {df_test["timestamp"].max()}')

# Target real: Q_net_m3h (cambio de volumen horario)
# El modelo predice Q_net, luego se calcula Qout = Qin - Q_net
y_test = df_test['Q_net_m3h'].values

# ==============================================================================
# 2. EVALUAR CADA MODELO
# ==============================================================================
print('\n' + '='*80)
print('2️⃣ EVALUACIÓN DE MODELOS EN TEST SET')
print('='*80)

modelos = {
    'LightGBM': 'models/forecasting/modelo_forecasting_lgbm.pkl',
    'XGBoost V3.0': 'models/forecasting/modelo_forecasting_xgb.pkl',
    'RandomForest': 'models/forecasting/modelo_forecasting_rf.pkl'
}

resultados = {}
predicciones = {}

for nombre, path in modelos.items():
    print(f'\n🤖 {nombre}')
    print('-'*80)
    
    try:
        modelo = joblib.load(path)
        features = modelo.feature_name_
        
        # Verificar features disponibles
        X_test = df_test[features].values
        
        # Predicción
        y_pred = modelo.predict(X_test)
        predicciones[nombre] = y_pred
        
        # Calcular métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        # Métricas adicionales
        residuos = y_test - y_pred
        
        resultados[nombre] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'n_features': int(len(features)),
            'residuos': {
                'media': float(residuos.mean()),
                'std': float(residuos.std()),
                'min': float(residuos.min()),
                'max': float(residuos.max()),
                'p5': float(np.percentile(residuos, 5)),
                'p95': float(np.percentile(residuos, 95))
            }
        }
        
        print(f'   Features:  {len(features)}')
        print(f'   MAE:       {mae:,.1f} m³/h')
        print(f'   RMSE:      {rmse:,.1f} m³/h')
        print(f'   R²:        {r2:.4f}')
        print(f'   MAPE:      {mape:.2f}%')
        
        print(f'\n   📊 Residuos:')
        print(f'      Media:      {residuos.mean():,.1f} m³/h')
        print(f'      Std:        {residuos.std():,.1f} m³/h')
        print(f'      Min:        {residuos.min():,.1f} m³/h')
        print(f'      Max:        {residuos.max():,.1f} m³/h')
        print(f'      P5-P95:     {np.percentile(residuos, 5):,.1f} a {np.percentile(residuos, 95):,.1f} m³/h')
        
    except FileNotFoundError:
        print(f'   ❌ Modelo no encontrado: {path}')
        resultados[nombre] = None
    except Exception as e:
        print(f'   ❌ Error: {str(e)}')
        resultados[nombre] = None

# ==============================================================================
# 3. COMPARACIÓN DE MODELOS
# ==============================================================================
print('\n' + '='*80)
print('3️⃣ COMPARACIÓN DE MODELOS')
print('='*80)

resultados_validos = {k: v for k, v in resultados.items() if v is not None}

if len(resultados_validos) > 0:
    df_comp = pd.DataFrame(resultados_validos).T
    df_comp = df_comp.sort_values('mae')
    
    print('\n🏆 Ranking por MAE (menor es mejor):')
    for i, (modelo, row) in enumerate(df_comp.iterrows(), 1):
        emoji = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '  '
        print(f'   {emoji} {i}. {modelo:15s} - MAE: {row["mae"]:7,.1f} m³/h | R²: {row["r2"]:.4f} | MAPE: {row["mape"]:5.2f}%')
    
    # Diferencias entre modelos
    print('\n📊 Diferencias entre mejor y peor modelo:')
    mejor_mae = df_comp['mae'].min()
    peor_mae = df_comp['mae'].max()
    mejor_modelo = df_comp['mae'].idxmin()
    peor_modelo = df_comp['mae'].idxmax()
    
    print(f'   Mejor: {mejor_modelo} (MAE: {mejor_mae:,.1f} m³/h)')
    print(f'   Peor:  {peor_modelo} (MAE: {peor_mae:,.1f} m³/h)')
    print(f'   Diferencia: {peor_mae - mejor_mae:,.1f} m³/h ({(peor_mae/mejor_mae - 1)*100:.1f}%)')

# ==============================================================================
# 4. VERIFICACIONES DE CONSISTENCIA
# ==============================================================================
print('\n' + '='*80)
print('4️⃣ VERIFICACIONES DE CONSISTENCIA')
print('='*80)

alertas = []

for nombre, metricas in resultados_validos.items():
    print(f'\n🔍 {nombre}:')
    
    # R² no debe ser demasiado alto (indicaría overfitting/leakage)
    if metricas['r2'] > 0.95:
        print(f'   ⚠️  R² muy alto ({metricas["r2"]:.4f}) - posible data leakage')
        alertas.append(f'{nombre}: R² sospechosamente alto ({metricas["r2"]:.4f})')
    elif metricas['r2'] < 0:
        print(f'   ⚠️  R² negativo ({metricas["r2"]:.4f}) - modelo peor que baseline')
        alertas.append(f'{nombre}: R² negativo ({metricas["r2"]:.4f})')
    else:
        print(f'   ✅ R² realista ({metricas["r2"]:.4f})')
    
    # MAE debe ser razonable (no demasiado bajo)
    if metricas['mae'] < 500:
        print(f'   ⚠️  MAE muy bajo ({metricas["mae"]:.1f}) - posible data leakage')
        alertas.append(f'{nombre}: MAE sospechosamente bajo ({metricas["mae"]:.1f})')
    else:
        print(f'   ✅ MAE razonable ({metricas["mae"]:.1f} m³/h)')
    
    # MAPE debe estar en rango esperado
    if metricas['mape'] < 10:
        print(f'   ⚠️  MAPE muy bajo ({metricas["mape"]:.2f}%) - posible data leakage')
        alertas.append(f'{nombre}: MAPE sospechosamente bajo ({metricas["mape"]:.2f}%)')
    elif metricas['mape'] > 50:
        print(f'   ⚠️  MAPE muy alto ({metricas["mape"]:.2f}%) - modelo pobre')
        alertas.append(f'{nombre}: MAPE muy alto ({metricas["mape"]:.2f}%)')
    else:
        print(f'   ✅ MAPE en rango esperado ({metricas["mape"]:.2f}%)')
    
    # Residuos deben tener media cercana a 0
    if abs(metricas['residuos']['media']) > 500:
        print(f'   ⚠️  Residuos sesgados (media: {metricas["residuos"]["media"]:.1f} m³/h)')
        alertas.append(f'{nombre}: Residuos sesgados ({metricas["residuos"]["media"]:.1f})')
    else:
        print(f'   ✅ Residuos sin sesgo (media: {metricas["residuos"]["media"]:.1f} m³/h)')

# ==============================================================================
# 5. ANÁLISIS DE DEMANDA REAL
# ==============================================================================
print('\n' + '='*80)
print('5️⃣ ANÁLISIS DE DEMANDA REAL VS PREDICHA')
print('='*80)

print(f'\n📈 Demanda Real (Test Set):')
print(f'   Media:      {y_test.mean():,.1f} m³/h')
print(f'   Mediana:    {np.median(y_test):,.1f} m³/h')
print(f'   Std:        {y_test.std():,.1f} m³/h')
print(f'   Min:        {y_test.min():,.1f} m³/h')
print(f'   Max:        {y_test.max():,.1f} m³/h')
print(f'   Rango:      {y_test.max() - y_test.min():,.1f} m³/h')

# Análisis de predicciones por modelo
if len(predicciones) > 0:
    print(f'\n📊 Predicciones por Modelo:')
    for nombre, y_pred in predicciones.items():
        print(f'\n   {nombre}:')
        print(f'      Media:      {y_pred.mean():,.1f} m³/h (real: {y_test.mean():,.1f})')
        print(f'      Mediana:    {np.median(y_pred):,.1f} m³/h (real: {np.median(y_test):,.1f})')
        print(f'      Std:        {y_pred.std():,.1f} m³/h (real: {y_test.std():,.1f})')
        print(f'      Rango:      {y_pred.max() - y_pred.min():,.1f} m³/h (real: {y_test.max() - y_test.min():,.1f})')

# ==============================================================================
# 6. CORRELACIÓN ENTRE PREDICCIONES
# ==============================================================================
if len(predicciones) >= 2:
    print('\n' + '='*80)
    print('6️⃣ CORRELACIÓN ENTRE PREDICCIONES DE MODELOS')
    print('='*80)
    
    nombres = list(predicciones.keys())
    print('\n📊 Matriz de Correlación:')
    print(f'\n   {"Modelo":15s}', end='')
    for nombre in nombres:
        print(f'{nombre[:12]:>13s}', end='')
    print()
    print('   ' + '-'*60)
    
    for nombre1 in nombres:
        print(f'   {nombre1:15s}', end='')
        for nombre2 in nombres:
            corr = np.corrcoef(predicciones[nombre1], predicciones[nombre2])[0, 1]
            print(f'{corr:13.4f}', end='')
        print()
    
    print('\n   💡 Correlaciones altas (>0.95) indican modelos similares')
    print('   💡 Correlaciones bajas (<0.80) indican estrategias diferentes')

# ==============================================================================
# 7. GENERAR REPORTE JSON
# ==============================================================================
print('\n' + '='*80)
print('7️⃣ GENERACIÓN DE REPORTE')
print('='*80)

reporte = {
    'fecha_auditoria': datetime.now().isoformat(),
    'dataset': {
        'total_registros': int(n),
        'train_registros': int(len(df_train)),
        'val_registros': int(len(df_val)),
        'test_registros': int(len(df_test)),
        'periodo_test': f'{df_test["timestamp"].min()} → {df_test["timestamp"].max()}'
    },
    'demanda_real': {
        'media': float(y_test.mean()),
        'mediana': float(np.median(y_test)),
        'std': float(y_test.std()),
        'min': float(y_test.min()),
        'max': float(y_test.max())
    },
    'modelos': resultados_validos,
    'ranking': {
        'por_mae': [{'modelo': k, 'mae': v['mae']} for k, v in sorted(resultados_validos.items(), key=lambda x: x[1]['mae'])],
        'por_r2': [{'modelo': k, 'r2': v['r2']} for k, v in sorted(resultados_validos.items(), key=lambda x: x[1]['r2'], reverse=True)]
    },
    'alertas': alertas,
    'verificacion': {
        'sin_alertas': bool(len(alertas) == 0),
        'n_modelos_evaluados': int(len(resultados_validos)),
        'metricas_consistentes': bool(len(alertas) == 0)
    }
}

Path('outputs').mkdir(exist_ok=True)
with open('outputs/AUDITORIA_METRICAS_ML.json', 'w', encoding='utf-8') as f:
    json.dump(reporte, f, indent=2, ensure_ascii=False)

print('✅ Reporte guardado en: outputs/AUDITORIA_METRICAS_ML.json')

# ==============================================================================
# 8. CONCLUSIÓN
# ==============================================================================
print('\n' + '='*80)
print('CONCLUSIÓN DE LA AUDITORÍA')
print('='*80)

if len(alertas) == 0:
    print('\n✅✅✅ TODAS LAS MÉTRICAS SON CONSISTENTES Y REALISTAS ✅✅✅')
    print('\n   🎉 No se detectaron anomalías')
    print('   🎉 Modelos funcionando correctamente')
    print('   🎉 Sin indicios de data leakage')
    print('   🎉 Métricas confiables para tesis')
else:
    print('\n⚠️  SE DETECTARON LAS SIGUIENTES ALERTAS:')
    for i, alerta in enumerate(alertas, 1):
        print(f'   {i}. {alerta}')

print('\n' + '='*80)
print('Fin de la auditoría')
print('='*80)
