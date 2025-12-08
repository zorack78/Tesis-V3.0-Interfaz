"""
GRÁFICOS COMPARATIVOS DE MODELOS ML
====================================
Genera gráficos de la primera y última semana del periodo de TEST
comparando las predicciones de los 3 modelos ML contra la demanda real.

Autor: Sistema de Análisis
Fecha: 2025-12-07
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates
from datetime import datetime

print('='*80)
print('GENERACIÓN DE GRÁFICOS COMPARATIVOS - PRIMERA Y ÚLTIMA SEMANA TEST')
print('='*80)

# ==============================================================================
# 1. CARGAR DATOS Y MODELOS
# ==============================================================================
print('\n1️⃣ Cargando datos y modelos...')
print('-'*80)

# Cargar dataset
df = pd.read_csv('data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Splits 70/15/15
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
df_test = df.iloc[val_end:].copy()

print(f'✅ Test set: {len(df_test)} registros')
print(f'   Periodo: {df_test["timestamp"].min()} → {df_test["timestamp"].max()}')

# Calcular perfil Qin usando SOLO datos de TRAIN (sin data leakage)
print('\n📊 Calculando perfil Qin (solo con datos de TRAIN - sin leakage)...')
df_train_for_qin = df.iloc[:train_end].copy()

# Usar BD_Qin si existe, sino usar sist_Qin_m3h
qin_path = Path('data/raw/BD_Qin_m3_UTC.csv')
if qin_path.exists():
    df_qin_raw = pd.read_csv(qin_path)
    df_qin_raw['timestamp'] = pd.to_datetime(df_qin_raw['timestamp'])
    
    # Filtrar SOLO hasta el final del train set
    fecha_limite_train = df_train_for_qin['timestamp'].max()
    df_qin_train = df_qin_raw[df_qin_raw['timestamp'] <= fecha_limite_train].copy()
    df_qin_train['hora'] = df_qin_train['timestamp'].dt.hour
    
    # Calcular perfil por hora
    qin_perfil = df_qin_train.groupby('hora')['Qin'].median().to_dict()
    print(f'   ✅ Perfil Qin: {len(df_qin_train):,} registros de TRAIN (BD_Qin)')
else:
    # Fallback
    df_train_for_qin['hora'] = df_train_for_qin['timestamp'].dt.hour
    qin_perfil = df_train_for_qin.groupby('hora')['sist_Qin_m3h'].median().to_dict()
    print(f'   ✅ Perfil Qin: {len(df_train_for_qin):,} registros de TRAIN (fallback)')

print(f'   Rango Qin: {min(qin_perfil.values()):,.0f} - {max(qin_perfil.values()):,.0f} m³/h')

# Aplicar perfil Qin al test set
df_test['hora'] = df_test['timestamp'].dt.hour
qin_test = df_test['hora'].map(qin_perfil).values

# Calcular Qout (demanda real) = Qin - Q_net
# Los modelos predicen Q_net, entonces Qout_pred = Qin - Q_net_pred
y_test_qnet = df_test['Q_net_m3h'].values
y_test = qin_test - y_test_qnet  # Qout real
timestamps = df_test['timestamp'].values

print(f'   ✅ Qout calculado para test set (Qout = Qin_perfil - Q_net)')
print(f'   Rango Qout: {y_test.min():,.0f} - {y_test.max():,.0f} m³/h')

# Configuración de modelos y colores
colores = {
    'LightGBM': '#2E86AB',      # Azul
    'XGBoost': '#A23B72',       # Morado  
    'RandomForest': '#F18F01'   # Naranja
}

predicciones = {}

# Cargar features del modelo principal
try:
    with open('models/forecasting/features.txt') as f:
        features = [line.strip() for line in f]
    print(f'✅ {len(features)} features cargadas')
except:
    print('❌ Error cargando features')
    exit(1)

# Preparar datos de entrenamiento para modelos que necesiten entrenarse
n_total = len(df)
train_end = int(n_total * 0.70)
df_train = df.iloc[:train_end].copy()

X_train = df_train[features].fillna(0).values
y_train = df_train['Q_net_m3h'].values
X_test_full = df_test[features].fillna(0).values

# ==============================================================================
# CARGAR/ENTRENAR MODELOS
# ==============================================================================

# 1. LightGBM (cargar pre-entrenado)
print('\n1️⃣ Cargando LightGBM...')
try:
    from lightgbm import LGBMRegressor
    modelo_lgb = joblib.load('models/forecasting/modelo_forecasting_lgbm.pkl')
    y_pred_lgb = modelo_lgb.predict(X_test_full)
    predicciones['LightGBM'] = y_pred_lgb
    print('   ✅ LightGBM cargado y predicción completada')
except Exception as e:
    print(f'   ⚠️  LightGBM no disponible: {str(e)}')

# 2. XGBoost (cargar pre-entrenado o buscar alternativa)
print('\n2️⃣ Buscando XGBoost...')
try:
    from xgboost import XGBRegressor
    # Intentar cargar modelo pre-entrenado
    try:
        modelo_xgb = joblib.load('models/forecasting/modelo_forecasting_xgboost.pkl')
        print('   ✅ XGBoost cargado desde archivo')
    except:
        # Si no existe, verificar si LightGBM es el modelo actual
        if 'LightGBM' in predicciones:
            print('   ℹ️  Modelo actual es LightGBM, no XGBoost separado')
        else:
            print('   ℹ️  Entrenando XGBoost...')
            modelo_xgb = XGBRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            modelo_xgb.fit(X_train, y_train)
            print('   ✅ XGBoost entrenado')
    
    y_pred_xgb = modelo_xgb.predict(X_test_full)
    predicciones['XGBoost'] = y_pred_xgb
    print('   ✅ XGBoost predicción completada')
except Exception as e:
    print(f'   ⚠️  XGBoost no disponible: {str(e)}')

# 3. RandomForest (entrenar)
print('\n3️⃣ Entrenando RandomForest...')
try:
    from sklearn.ensemble import RandomForestRegressor
    modelo_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    modelo_rf.fit(X_train, y_train)
    y_pred_rf = modelo_rf.predict(X_test_full)
    predicciones['RandomForest'] = y_pred_rf
    print('   ✅ RandomForest entrenado y predicción completada')
except Exception as e:
    print(f'   ❌ RandomForest falló: {str(e)}')

if len(predicciones) == 0:
    print('\n❌ No se pudo cargar ningún modelo. Terminando.')
    exit(1)

print(f'\n✅ {len(predicciones)} modelos cargados y listos para graficar')

# Convertir predicciones de Q_net a Qout (Qout = Qin - Q_net)
print('\n📊 Convirtiendo predicciones a Qout...')
predicciones_qout = {}
for nombre, y_pred_qnet in predicciones.items():
    y_pred_qout = qin_test - y_pred_qnet
    predicciones_qout[nombre] = y_pred_qout
    print(f'   ✅ {nombre}: Rango Qout pred = {y_pred_qout.min():,.0f} - {y_pred_qout.max():,.0f} m³/h')

# Usar predicciones_qout en lugar de predicciones
predicciones = predicciones_qout

# ==============================================================================
# 2. EXTRAER PRIMERA Y ÚLTIMA SEMANA
# ==============================================================================
print('\n2️⃣ Extrayendo primera y última semana...')
print('-'*80)

# Primera semana (168 horas)
primera_semana_idx = slice(0, min(168, len(df_test)))
ultima_semana_idx = slice(max(0, len(df_test) - 168), len(df_test))

# Datos primera semana
ts_primera = timestamps[primera_semana_idx]
y_real_primera = y_test[primera_semana_idx]
pred_primera = {nombre: pred[primera_semana_idx] for nombre, pred in predicciones.items()}

# Datos última semana
ts_ultima = timestamps[ultima_semana_idx]
y_real_ultima = y_test[ultima_semana_idx]
pred_ultima = {nombre: pred[ultima_semana_idx] for nombre, pred in predicciones.items()}

print(f'✅ Primera semana: {len(ts_primera)} registros ({ts_primera[0]} a {ts_primera[-1]})')
print(f'✅ Última semana:  {len(ts_ultima)} registros ({ts_ultima[0]} a {ts_ultima[-1]})')

# ==============================================================================
# 3. CALCULAR MÉTRICAS POR SEMANA
# ==============================================================================
def calcular_metricas(y_real, y_pred):
    """Calcula MAE, RMSE y R² entre valores reales y predichos"""
    mae = np.mean(np.abs(y_real - y_pred))
    rmse = np.sqrt(np.mean((y_real - y_pred)**2))
    
    ss_res = np.sum((y_real - y_pred)**2)
    ss_tot = np.sum((y_real - np.mean(y_real))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return mae, rmse, r2

print('\n3️⃣ Métricas por semana y modelo:')
print('-'*80)

metricas_primera = {}
metricas_ultima = {}

for nombre in predicciones.keys():
    # Primera semana
    mae_1, rmse_1, r2_1 = calcular_metricas(y_real_primera, pred_primera[nombre])
    metricas_primera[nombre] = {'mae': mae_1, 'rmse': rmse_1, 'r2': r2_1}
    
    # Última semana
    mae_u, rmse_u, r2_u = calcular_metricas(y_real_ultima, pred_ultima[nombre])
    metricas_ultima[nombre] = {'mae': mae_u, 'rmse': rmse_u, 'r2': r2_u}
    
    print(f'\n{nombre}:')
    print(f'  Primera semana - MAE: {mae_1:,.1f} | RMSE: {rmse_1:,.1f} | R²: {r2_1:.4f}')
    print(f'  Última semana  - MAE: {mae_u:,.1f} | RMSE: {rmse_u:,.1f} | R²: {r2_u:.4f}')

# ==============================================================================
# 4. GENERAR GRÁFICO - PRIMERA SEMANA
# ==============================================================================
print('\n4️⃣ Generando gráfico - Primera semana...')
print('-'*80)

fig1, ax1 = plt.subplots(figsize=(16, 8))

# Línea real (más gruesa y destacada)
ax1.plot(ts_primera, y_real_primera, 
         color='black', linewidth=2.5, label='Real', 
         marker='o', markersize=3, alpha=0.8, zorder=10)

# Predicciones de cada modelo
for nombre, y_pred in pred_primera.items():
    label = f'{nombre}'
    ax1.plot(ts_primera, y_pred, 
             color=colores.get(nombre, '#999999'), 
             linewidth=1.8, label=label,
             marker='s', markersize=2.5, alpha=0.7)

# Configuración del gráfico
ax1.set_xlabel('Fecha y Hora', fontsize=12, fontweight='bold')
ax1.set_ylabel('Qout - Demanda (m³/h)', fontsize=12, fontweight='bold')
# Formato simplificado de fecha (solo día/mes hora)
fecha_inicio = pd.Timestamp(ts_primera[0]).strftime('%d/%m %H:00')
fecha_fin = pd.Timestamp(ts_primera[-1]).strftime('%d/%m %H:00')
ax1.set_title(f'Comparativa Modelos ML - PRIMERA SEMANA TEST\n{fecha_inicio} a {fecha_fin}',
              fontsize=14, fontweight='bold', pad=20)

# Formato de fechas en eje X
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
ax1.xaxis.set_major_locator(mdates.DayLocator())
ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=6))

# Grid y leyenda
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.legend(loc='best', fontsize=10, framealpha=0.9)

# Anotación con estadísticas
stats_text = f"Periodo: {len(ts_primera)} horas\n"
stats_text += f"Qout Real: {y_real_primera.mean():,.0f} ± {y_real_primera.std():,.0f} m³/h\n"
stats_text += f"Rango: [{y_real_primera.min():,.0f}, {y_real_primera.max():,.0f}]"
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.xticks(rotation=0)
plt.tight_layout()

# Guardar
Path('outputs/figures').mkdir(parents=True, exist_ok=True)
fig1.savefig('outputs/figures/comparativa_primera_semana_test.png', dpi=300, bbox_inches='tight')
print('✅ Guardado: outputs/figures/comparativa_primera_semana_test.png')

# ==============================================================================
# 5. GENERAR GRÁFICO - ÚLTIMA SEMANA
# ==============================================================================
print('\n5️⃣ Generando gráfico - Última semana...')
print('-'*80)

fig2, ax2 = plt.subplots(figsize=(12, 8))

# Línea real (más gruesa y destacada)
ax2.plot(ts_ultima, y_real_ultima, 
         color='black', linewidth=2.5, label='Real', 
         marker='o', markersize=3, alpha=0.8, zorder=10)

# Predicciones de cada modelo
for nombre, y_pred in pred_ultima.items():
    label = f'{nombre}'
    ax2.plot(ts_ultima, y_pred, 
             color=colores.get(nombre, '#999999'), 
             linewidth=1.8, label=label,
             marker='s', markersize=2.5, alpha=0.7)

# Configuración del gráfico
ax2.set_xlabel('Fecha y Hora', fontsize=12, fontweight='bold')
ax2.set_ylabel('Qout - Demanda (m³/h)', fontsize=12, fontweight='bold')
# Formato simplificado de fecha (solo día/mes hora)
fecha_inicio_u = pd.Timestamp(ts_ultima[0]).strftime('%d/%m %H:00')
fecha_fin_u = pd.Timestamp(ts_ultima[-1]).strftime('%d/%m %H:00')
ax2.set_title(f'Comparativa Modelos ML - ÚLTIMA SEMANA TEST\n{fecha_inicio_u} a {fecha_fin_u}',
              fontsize=14, fontweight='bold', pad=20)

# Formato de fechas en eje X
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
ax2.xaxis.set_major_locator(mdates.DayLocator())
ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=6))

# Grid y leyenda
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.legend(loc='best', fontsize=10, framealpha=0.9)

# Anotación con estadísticas
stats_text = f"Periodo: {len(ts_ultima)} horas\n"
stats_text += f"Qout Real: {y_real_ultima.mean():,.0f} ± {y_real_ultima.std():,.0f} m³/h\n"
stats_text += f"Rango: [{y_real_ultima.min():,.0f}, {y_real_ultima.max():,.0f}]"
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.xticks(rotation=0)
plt.tight_layout()

# Guardar
fig2.savefig('outputs/figures/comparativa_ultima_semana_test.png', dpi=300, bbox_inches='tight')
print('✅ Guardado: outputs/figures/comparativa_ultima_semana_test.png')

# ==============================================================================
# 6. GRÁFICO COMBINADO (OPCIONAL) - AMBAS SEMANAS EN SUBPLOTS
# ==============================================================================
print('\n6️⃣ Generando gráfico combinado...')
print('-'*80)

fig3, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 10))

# ===== SUBPLOT SUPERIOR: PRIMERA SEMANA =====
ax_top.plot(ts_primera, y_real_primera, 
            color='black', linewidth=2.5, label='Real', 
            marker='o', markersize=3, alpha=0.8, zorder=10)

for nombre, y_pred in pred_primera.items():
    label = f'{nombre}'
    ax_top.plot(ts_primera, y_pred, 
                color=colores.get(nombre, '#999999'), 
                linewidth=1.8, label=label,
                marker='s', markersize=2.5, alpha=0.7)

ax_top.set_ylabel('Qout - Demanda (m³/h)', fontsize=11, fontweight='bold')
fecha_i = pd.Timestamp(ts_primera[0]).strftime('%d/%m %H:00')
fecha_f = pd.Timestamp(ts_primera[-1]).strftime('%d/%m %H:00')
ax_top.set_title(f'PRIMERA SEMANA TEST: {fecha_i} a {fecha_f}',
                 fontsize=12, fontweight='bold', pad=15)
ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
ax_top.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax_top.legend(loc='best', fontsize=9, framealpha=0.9)

# ===== SUBPLOT INFERIOR: ÚLTIMA SEMANA =====
ax_bottom.plot(ts_ultima, y_real_ultima, 
               color='black', linewidth=2.5, label='Real', 
               marker='o', markersize=3, alpha=0.8, zorder=10)

for nombre, y_pred in pred_ultima.items():
    label = f'{nombre}'
    ax_bottom.plot(ts_ultima, y_pred, 
                   color=colores.get(nombre, '#999999'), 
                   linewidth=1.8, label=label,
                   marker='s', markersize=2.5, alpha=0.7)

ax_bottom.set_xlabel('Fecha y Hora', fontsize=11, fontweight='bold')
ax_bottom.set_ylabel('Qout - Demanda (m³/h)', fontsize=11, fontweight='bold')
fecha_i_u = pd.Timestamp(ts_ultima[0]).strftime('%d/%m %H:00')
fecha_f_u = pd.Timestamp(ts_ultima[-1]).strftime('%d/%m %H:00')
ax_bottom.set_title(f'ÚLTIMA SEMANA TEST: {fecha_i_u} a {fecha_f_u}',
                    fontsize=12, fontweight='bold', pad=15)
ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
ax_bottom.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax_bottom.legend(loc='best', fontsize=9, framealpha=0.9)

plt.suptitle('Comparativa Modelos ML - Primera y Última Semana del Periodo TEST',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

# Guardar
fig3.savefig('outputs/figures/comparativa_ambas_semanas_test.png', dpi=300, bbox_inches='tight')
print('✅ Guardado: outputs/figures/comparativa_ambas_semanas_test.png')

# ==============================================================================
# 7. RESUMEN FINAL
# ==============================================================================
print('\n' + '='*80)
print('RESUMEN DE GRÁFICOS GENERADOS')
print('='*80)

print('\n📊 Archivos generados:')
print('   1. outputs/figures/comparativa_primera_semana_test.png')
print('   2. outputs/figures/comparativa_ultima_semana_test.png')
print('   3. outputs/figures/comparativa_ambas_semanas_test.png')

print('\n📈 Modelos evaluados:', len(predicciones))
for nombre in predicciones.keys():
    print(f'   ✅ {nombre}')

print('\n⏰ Periodos analizados:')
print(f'   • Primera semana: {ts_primera[0]} a {ts_primera[-1]} ({len(ts_primera)} horas)')
print(f'   • Última semana:  {ts_ultima[0]} a {ts_ultima[-1]} ({len(ts_ultima)} horas)')

print('\n' + '='*80)
print('✅ PROCESO COMPLETADO EXITOSAMENTE')
print('='*80)

plt.close('all')
