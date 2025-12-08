"""
SCRIPT OFICIAL PARA CALCULAR MÉTRICAS
======================================
Script oficial para calcular métricas del modelo de predicción de demanda
sobre el período de test. Este script garantiza reproducibilidad y consistencia.

USO:
    python calcular_metricas_oficiales.py

CARACTERÍSTICAS:
- Usa modelo pickle guardado (no entrena)
- Target estándar: Qout (demanda)
- Período de test fijo: 70/15/15 split
- Sin data leakage: Qin_perfil de train solamente
- Resultados guardados con timestamp y versión de modelo

Autor: Sistema de Análisis
Fecha: 2025-12-08
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error,
    mean_absolute_percentage_error
)
import json

print('='*80)
print('CÁLCULO OFICIAL DE MÉTRICAS - MODELO PREDICCIÓN DEMANDA')
print('='*80)

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

MODELOS = {
    'LightGBM': 'models/forecasting/modelo_forecasting_lgbm.pkl',
    'XGBoost': 'models/forecasting/modelo_forecasting_xgboost.pkl',
    'RandomForest': 'models/forecasting/modelo_forecasting_randomforest.pkl',
}

DATASET_PATH = 'data/processed/dataset_features_completo.csv'
OUTPUT_PATH = 'outputs/metricas_oficiales_todos_modelos.json'
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15

print(f'\n⚙️  CONFIGURACIÓN:')
print(f'   • Modelos: LightGBM, XGBoost, RandomForest')
print(f'   • Dataset: {DATASET_PATH}')
print(f'   • Split: {SPLIT_TRAIN}/{SPLIT_VAL}/{SPLIT_TEST}')
print(f'   • Target: Qout (demanda)')

# ==============================================================================
# 1. CARGAR DATOS
# ==============================================================================

print(f'\n1️⃣ Cargando datos...')
print('-'*80)

df = pd.read_csv(DATASET_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])

n = len(df)
train_end = int(n * SPLIT_TRAIN)
val_end = int(n * (SPLIT_TRAIN + SPLIT_VAL))

df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

print(f'✅ Datos cargados: {n:,} registros totales')
print(f'   • Train: {len(df_train):,} registros ({SPLIT_TRAIN:.0%})')
print(f'   • Validation: {len(df_val):,} registros ({SPLIT_VAL:.0%})')
print(f'   • Test: {len(df_test):,} registros ({SPLIT_TEST:.0%})')
print(f'\n📅 Período de TEST:')
print(f'   Inicio: {df_test["timestamp"].min()}')
print(f'   Fin:    {df_test["timestamp"].max()}')
print(f'   Días:   {(df_test["timestamp"].max() - df_test["timestamp"].min()).days}')

# ==============================================================================
# 2. CARGAR MODELOS
# ==============================================================================

print(f'\n2️⃣ Cargando modelos...')
print('-'*80)

modelos_cargados = {}
modelos_info = {}
features_list = None

for nombre_modelo, path_modelo in MODELOS.items():
    print(f'\n📦 {nombre_modelo}:')
    
    if not Path(path_modelo).exists():
        print(f'   ⚠️  No encontrado: {path_modelo}')
        print(f'   ⏭️  Saltando...')
        continue
    
    try:
        modelo = joblib.load(path_modelo)
        modelos_cargados[nombre_modelo] = modelo
        
        # Información del modelo
        info = {
            'path': path_modelo,
            'tipo': type(modelo).__name__,
            'fecha_modificacion': datetime.fromtimestamp(
                Path(path_modelo).stat().st_mtime
            ).isoformat(),
            'size_mb': Path(path_modelo).stat().st_size / (1024 * 1024),
            'existe': True,
        }
        
        if hasattr(modelo, 'feature_name_'):
            info['num_features'] = len(modelo.feature_name_)
            if features_list is None:
                features_list = modelo.feature_name_
        elif hasattr(modelo, 'n_features_in_'):
            info['num_features'] = modelo.n_features_in_
        
        if hasattr(modelo, 'get_params'):
            params = modelo.get_params()
            info['hiperparametros'] = {
                k: v for k, v in params.items() 
                if k in ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves',
                        'min_samples_split', 'min_samples_leaf']
                and v is not None
            }
        
        modelos_info[nombre_modelo] = info
        
        print(f'   ✅ Cargado: {info["tipo"]}')
        print(f'   • Features: {info.get("num_features", "N/A")}')
        print(f'   • Tamaño: {info["size_mb"]:.2f} MB')
        print(f'   • Modificado: {info["fecha_modificacion"]}')
        if 'hiperparametros' in info:
            print(f'   • Hiperparámetros principales:')
            for k, v in list(info['hiperparametros'].items())[:4]:
                print(f'      - {k}: {v}')
        
    except Exception as e:
        print(f'   ❌ Error al cargar: {str(e)}')
        modelos_info[nombre_modelo] = {
            'path': path_modelo,
            'existe': False,
            'error': str(e)
        }

if not modelos_cargados:
    print(f'\n❌ ERROR: No se pudo cargar ningún modelo')
    print(f'   Verifica que existan los archivos .pkl en models/forecasting/')
    exit(1)

print(f'\n✅ Modelos cargados: {len(modelos_cargados)} de {len(MODELOS)}')

# ==============================================================================
# 3. PREPARAR FEATURES
# ==============================================================================

print(f'\n3️⃣ Preparando features...')
print('-'*80)

# Identificar columnas de features (excluir target y timestamp)
exclude_cols = ['timestamp', 'Q_net_m3h', 'sist_Qin_m3h', 'sist_Vtotal_m3']
if features_list is not None:
    feature_cols = features_list
else:
    feature_cols = [col for col in df.columns if col not in exclude_cols]

# Verificar que todas las features existen
missing_features = [f for f in feature_cols if f not in df_test.columns]
if missing_features:
    print(f'⚠️  Advertencia: {len(missing_features)} features no encontradas')
    feature_cols = [f for f in feature_cols if f in df_test.columns]

X_test = df_test[feature_cols].fillna(0)

print(f'✅ Features preparadas: {X_test.shape[1]} columnas')
print(f'   • Registros de test: {X_test.shape[0]:,}')

# ==============================================================================
# 4. CALCULAR Qin_perfil (SIN DATA LEAKAGE)
# ==============================================================================

print(f'\n4️⃣ Calculando Qin_perfil (solo con datos de TRAIN)...')
print('-'*80)

# IMPORTANTE: Usar solo datos de TRAIN para evitar data leakage
if 'sist_Qin_m3h' not in df_train.columns:
    print('❌ ERROR: Columna sist_Qin_m3h no encontrada en datos de train')
    exit(1)

# Crear columna 'hora' si no existe
if 'hora' not in df_train.columns:
    df_train['hora'] = pd.to_datetime(df_train['timestamp']).dt.hour

if 'hora' not in df_test.columns:
    df_test['hora'] = pd.to_datetime(df_test['timestamp']).dt.hour

# Calcular perfil horario (mediana por hora) solo con TRAIN
Qin_perfil = df_train.groupby('hora')['sist_Qin_m3h'].median()

print(f'✅ Qin_perfil calculado (24 valores horarios)')
print(f'   Rango: {Qin_perfil.min():,.0f} - {Qin_perfil.max():,.0f} m³/h')
print(f'   Media: {Qin_perfil.mean():,.0f} m³/h')

# Mapear a test
df_test['Qin_perfil'] = df_test['hora'].map(Qin_perfil)

# ==============================================================================
# 5. GENERAR PREDICCIONES (TODOS LOS MODELOS)
# ==============================================================================

print(f'\n5️⃣ Generando predicciones para todos los modelos...')
print('-'*80)

predicciones_qnet = {}
predicciones_qout = {}

for nombre_modelo, modelo in modelos_cargados.items():
    print(f'\n📊 {nombre_modelo}:')
    
    try:
        # Predecir Q_net
        y_pred_Qnet = modelo.predict(X_test)
        predicciones_qnet[nombre_modelo] = y_pred_Qnet
        
        print(f'   ✅ Predicciones Q_net: {len(y_pred_Qnet):,} valores')
        print(f'      Rango: {y_pred_Qnet.min():,.0f} - {y_pred_Qnet.max():,.0f} m³/h')
        print(f'      Media: {y_pred_Qnet.mean():,.0f} m³/h')
        
        # Calcular Qout_pred (demanda predicha)
        y_pred_Qout = df_test['Qin_perfil'].values - y_pred_Qnet
        predicciones_qout[nombre_modelo] = y_pred_Qout
        
        print(f'   ✅ Predicciones Qout (demanda):')
        print(f'      Rango: {y_pred_Qout.min():,.0f} - {y_pred_Qout.max():,.0f} m³/h')
        print(f'      Media: {y_pred_Qout.mean():,.0f} m³/h')
        
    except Exception as e:
        print(f'   ❌ Error en predicción: {str(e)}')

print(f'\n✅ Predicciones generadas para {len(predicciones_qnet)} modelos')

# ==============================================================================
# 6. CALCULAR QOUT REAL
# ==============================================================================

print(f'\n6️⃣ Calculando Qout real...')
print('-'*80)

# Calcular Q_net real y Qout real
if 'Q_net_m3h' not in df_test.columns:
    print('❌ ERROR: Columna Q_net_m3h no encontrada')
    exit(1)

y_true_Qnet = df_test['Q_net_m3h'].values
y_true_Qout = df_test['Qin_perfil'].values - y_true_Qnet

print(f'✅ Qout real calculado')
print(f'   Rango: {y_true_Qout.min():,.0f} - {y_true_Qout.max():,.0f} m³/h')
print(f'   Media: {y_true_Qout.mean():,.0f} m³/h')

# ==============================================================================
# 7. CALCULAR MÉTRICAS (TODOS LOS MODELOS)
# ==============================================================================

print(f'\n7️⃣ Calculando métricas para todos los modelos...')
print('-'*80)

metricas_todos = {}

for nombre_modelo in predicciones_qnet.keys():
    print(f'\n📊 {nombre_modelo}:')
    print('   ' + '-'*70)
    
    y_pred_Qnet = predicciones_qnet[nombre_modelo]
    y_pred_Qout = predicciones_qout[nombre_modelo]
    
    # Métricas para Q_net
    metricas_qnet = {
        'r2': r2_score(y_true_Qnet, y_pred_Qnet),
        'mae': mean_absolute_error(y_true_Qnet, y_pred_Qnet),
        'rmse': np.sqrt(mean_squared_error(y_true_Qnet, y_pred_Qnet)),
        'mape': mean_absolute_percentage_error(y_true_Qnet, y_pred_Qnet) * 100,
    }
    
    # Métricas para Qout (PRINCIPAL)
    metricas_qout = {
        'r2': r2_score(y_true_Qout, y_pred_Qout),
        'mae': mean_absolute_error(y_true_Qout, y_pred_Qout),
        'rmse': np.sqrt(mean_squared_error(y_true_Qout, y_pred_Qout)),
        'mape': mean_absolute_percentage_error(y_true_Qout, y_pred_Qout) * 100,
    }
    
    metricas_todos[nombre_modelo] = {
        'Q_net': metricas_qnet,
        'Qout': metricas_qout,
    }
    
    print(f'   Q_net (predicción directa):')
    print(f'      • R²:   {metricas_qnet["r2"]:.4f}')
    print(f'      • MAE:  {metricas_qnet["mae"]:,.2f} m³/h')
    print(f'      • RMSE: {metricas_qnet["rmse"]:,.2f} m³/h')
    print(f'      • MAPE: {metricas_qnet["mape"]:.2f}%')
    
    print(f'   Qout (DEMANDA - TARGET PRINCIPAL):')
    print(f'      • R²:   {metricas_qout["r2"]:.4f}')
    print(f'      • MAE:  {metricas_qout["mae"]:,.2f} m³/h')
    print(f'      • RMSE: {metricas_qout["rmse"]:,.2f} m³/h')
    print(f'      • MAPE: {metricas_qout["mape"]:.2f}%')

# Comparación de modelos
print(f'\n' + '='*80)
print('📊 COMPARACIÓN DE MODELOS (Qout - Demanda)')
print('='*80)
print(f'\n{"Modelo":<15} {"R²":>10} {"MAE (m³/h)":>15} {"RMSE (m³/h)":>15} {"MAPE (%)":>12}')
print('-'*80)

for nombre_modelo, metricas in metricas_todos.items():
    m = metricas['Qout']
    print(f'{nombre_modelo:<15} {m["r2"]:>10.4f} {m["mae"]:>15,.0f} {m["rmse"]:>15,.0f} {m["mape"]:>12.2f}')

# Identificar mejor modelo
mejor_modelo_r2 = max(metricas_todos.items(), key=lambda x: x[1]['Qout']['r2'])
mejor_modelo_mae = min(metricas_todos.items(), key=lambda x: x[1]['Qout']['mae'])

print(f'\n🏆 MEJOR MODELO:')
print(f'   • Por R²:  {mejor_modelo_r2[0]} (R² = {mejor_modelo_r2[1]["Qout"]["r2"]:.4f})')
print(f'   • Por MAE: {mejor_modelo_mae[0]} (MAE = {mejor_modelo_mae[1]["Qout"]["mae"]:,.0f} m³/h)')

# ==============================================================================
# 8. GUARDAR RESULTADOS
# ==============================================================================

print(f'\n8️⃣ Guardando resultados...')
print('-'*80)

resultados = {
    'metadata': {
        'fecha_calculo': datetime.now().isoformat(),
        'script': 'calcular_metricas_oficiales.py',
        'version': '2.0',
        'descripcion': 'Métricas oficiales de todos los modelos',
        'modelos_evaluados': list(modelos_cargados.keys()),
    },
    'modelos': modelos_info,
    'dataset': {
        'path': DATASET_PATH,
        'total_registros': n,
        'split': {
            'train': {'registros': len(df_train), 'porcentaje': SPLIT_TRAIN},
            'validation': {'registros': len(df_val), 'porcentaje': SPLIT_VAL},
            'test': {'registros': len(df_test), 'porcentaje': SPLIT_TEST},
        },
        'periodo_test': {
            'inicio': df_test['timestamp'].min().isoformat(),
            'fin': df_test['timestamp'].max().isoformat(),
            'dias': (df_test['timestamp'].max() - df_test['timestamp'].min()).days,
        },
    },
    'features': {
        'num_features': len(feature_cols),
        'sin_data_leakage': True,
        'Qin_perfil_desde': 'train_only',
    },
    'metricas': metricas_todos,
    'ranking': {
        'mejor_por_r2': {
            'modelo': mejor_modelo_r2[0],
            'r2': mejor_modelo_r2[1]['Qout']['r2'],
        },
        'mejor_por_mae': {
            'modelo': mejor_modelo_mae[0],
            'mae': mejor_modelo_mae[1]['Qout']['mae'],
        },
    },
    'estadisticas': {
        'Qout_real': {
            'min': float(y_true_Qout.min()),
            'max': float(y_true_Qout.max()),
            'mean': float(y_true_Qout.mean()),
            'std': float(y_true_Qout.std()),
        },
    },
}

# Crear directorio de outputs si no existe
Path('outputs').mkdir(exist_ok=True)

# Guardar JSON
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print(f'✅ Resultados guardados en: {OUTPUT_PATH}')

# También guardar versión con timestamp
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
output_timestamped = f'outputs/metricas_oficiales_{timestamp_str}.json'
with open(output_timestamped, 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print(f'✅ Copia con timestamp: {output_timestamped}')

# ==============================================================================
# 9. RESUMEN FINAL
# ==============================================================================

print('\n' + '='*80)
print('✅ CÁLCULO DE MÉTRICAS COMPLETADO')
print('='*80)

print(f'\n🏆 MEJOR MODELO (por R²): {mejor_modelo_r2[0]}')
print(f'   • R²:   {mejor_modelo_r2[1]["Qout"]["r2"]:.4f}')
print(f'   • MAE:  {mejor_modelo_r2[1]["Qout"]["mae"]:,.0f} m³/h')
print(f'   • RMSE: {mejor_modelo_r2[1]["Qout"]["rmse"]:,.0f} m³/h')

print(f'\n📊 TODOS LOS MODELOS (Qout - Demanda):')
for nombre_modelo, metricas in metricas_todos.items():
    m = metricas['Qout']
    print(f'   {nombre_modelo}:')
    print(f'      R² = {m["r2"]:.4f}, MAE = {m["mae"]:,.0f} m³/h, RMSE = {m["rmse"]:,.0f} m³/h')

print(f'\n📄 Resultados guardados en:')
print(f'   • {OUTPUT_PATH}')
print(f'   • {output_timestamped}')

print(f'\n💡 Para reportar en tesis (tabla comparativa):')
print(f'\n{"Modelo":<15} {"R²":>10} {"MAE":>15} {"RMSE":>15}')
print('-'*55)
for nombre_modelo, metricas in metricas_todos.items():
    m = metricas['Qout']
    print(f'{nombre_modelo:<15} {m["r2"]:>10.4f} {m["mae"]:>15,.0f} {m["rmse"]:>15,.0f}')

print(f'\n   Mejor modelo: {mejor_modelo_r2[0]} con R² = {mejor_modelo_r2[1]["Qout"]["r2"]:.4f}')

print('\n' + '='*80)
