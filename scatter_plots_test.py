"""
Gráficos de Scatter Plot: Predicción vs Demanda Real
Para primera y última semana del TEST set - 3 modelos ML
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print('='*80)
print('SCATTER PLOTS - PREDICCIÓN VS DEMANDA REAL (Qout)')
print('='*80)

# ==============================================================================
# 1. CARGAR DATOS Y PREPARAR TEST SET
# ==============================================================================
print('\n1️⃣ Cargando datos...')
print('-'*80)

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

qin_path = Path('data/raw/BD_Qin_m3_UTC.csv')
if qin_path.exists():
    df_qin_raw = pd.read_csv(qin_path)
    df_qin_raw['timestamp'] = pd.to_datetime(df_qin_raw['timestamp'])
    df_qin_raw['hora'] = df_qin_raw['timestamp'].dt.hour
    
    fecha_limite_train = df_train_for_qin['timestamp'].max()
    df_qin_train = df_qin_raw[df_qin_raw['timestamp'] <= fecha_limite_train].copy()
    
    qin_perfil = df_qin_train.groupby('hora')['Qin'].median().to_dict()
    print(f'   ✅ Perfil Qin: {len(df_qin_train):,} registros de TRAIN (BD_Qin)')
else:
    qin_perfil = df_train_for_qin.groupby('hora')['sist_Qin_m3h'].median().to_dict()
    print(f'   ✅ Perfil Qin: {len(df_train_for_qin):,} registros de TRAIN (fallback)')

print(f'   Rango Qin: {min(qin_perfil.values()):,.0f} - {max(qin_perfil.values()):,.0f} m³/h')

# Aplicar perfil Qin al test set
qin_test = df_test['hora'].map(qin_perfil).values

# Cargar features del modelo principal
try:
    with open('models/forecasting/features.txt') as f:
        features = [line.strip() for line in f]
    print(f'✅ {len(features)} features cargadas')
except:
    print('❌ Error cargando features')
    exit(1)

# Preparar datos de entrenamiento
n_total = len(df)
train_end_full = int(n_total * 0.70)
df_train = df.iloc[:train_end_full].copy()

X_train = df_train[features].fillna(0).values
y_train = df_train['Q_net_m3h'].values
X_test = df_test[features].fillna(0).values
y_test_qnet = df_test['Q_net_m3h'].values

# Convertir a Qout = Qin_perfil - Q_net
y_test = qin_test - y_test_qnet

print(f'   ✅ Qout calculado para test set (Qout = Qin_perfil - Q_net)')
print(f'   Rango Qout: {y_test.min():,.0f} - {y_test.max():,.0f} m³/h')

# ==============================================================================
# 2. CARGAR/ENTRENAR MODELOS
# ==============================================================================
print('\n2️⃣ Cargando/entrenando modelos...')
print('-'*80)

predicciones = {}
colores = {
    'LightGBM': '#2E86AB',     # Azul
    'XGBoost': '#A23B72',      # Morado
    'RandomForest': '#F18F01'  # Naranja
}

# --- LightGBM ---
print('1️⃣ Cargando LightGBM...')
try:
    lgbm_data = joblib.load('models/forecasting/modelo_forecasting_lgbm.pkl')
    # Extraer modelo si está en formato diccionario
    modelo_lgbm = lgbm_data['modelo'] if isinstance(lgbm_data, dict) else lgbm_data
    y_pred_lgbm = modelo_lgbm.predict(X_test)
    predicciones['LightGBM'] = y_pred_lgbm
    print('   ✅ LightGBM cargado y predicción completada')
except Exception as e:
    print(f'   ❌ Error cargando LightGBM: {e}')

# --- XGBoost ---
print('\n2️⃣ Buscando XGBoost...')
try:
    try:
        xgb_data = joblib.load('models/forecasting/modelo_forecasting_xgboost.pkl')
    except:
        xgb_data = joblib.load('models/modelo_forecasting_xgboost.pkl')
    
    # Extraer modelo si está en formato diccionario
    modelo_xgb = xgb_data['modelo'] if isinstance(xgb_data, dict) else xgb_data
    y_pred_xgb = modelo_xgb.predict(X_test)
    predicciones['XGBoost'] = y_pred_xgb
    print('   ✅ XGBoost cargado desde archivo')
    print('   ✅ XGBoost predicción completada')
except Exception as e:
    print(f'   ❌ Error: XGBoost - {e}')

# --- RandomForest ---
print('\n3️⃣ Entrenando RandomForest...')
try:
    modelo_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    modelo_rf.fit(X_train, y_train)
    y_pred_rf = modelo_rf.predict(X_test)
    predicciones['RandomForest'] = y_pred_rf
    print('   ✅ RandomForest entrenado y predicción completada')
except Exception as e:
    print(f'   ❌ Error: {e}')

print(f'\n✅ {len(predicciones)} modelos cargados')

# Convertir predicciones de Q_net a Qout
predicciones_qout = {}
for nombre, y_pred_qnet in predicciones.items():
    y_pred_qout = qin_test - y_pred_qnet
    predicciones_qout[nombre] = y_pred_qout

# ==============================================================================
# 3. EXTRAER PRIMERA Y ÚLTIMA SEMANA
# ==============================================================================
print('\n3️⃣ Extrayendo primera y última semana...')
print('-'*80)

ts_test = df_test['timestamp'].values
primera_semana_idx = np.arange(0, min(168, len(df_test)))
ultima_semana_idx = np.arange(max(0, len(df_test)-168), len(df_test))

y_real_primera = y_test[primera_semana_idx]
y_real_ultima = y_test[ultima_semana_idx]

pred_primera = {nombre: pred[primera_semana_idx] for nombre, pred in predicciones_qout.items()}
pred_ultima = {nombre: pred[ultima_semana_idx] for nombre, pred in predicciones_qout.items()}

print(f'✅ Primera semana: {len(primera_semana_idx)} registros')
print(f'✅ Última semana: {len(ultima_semana_idx)} registros')

# ==============================================================================
# 4. FUNCIÓN PARA CALCULAR MÉTRICAS
# ==============================================================================
def calcular_metricas(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    return mae, rmse, r2

# ==============================================================================
# 5. GENERAR SCATTER PLOTS - PRIMERA SEMANA
# ==============================================================================
print('\n4️⃣ Generando scatter plots - Primera semana...')
print('-'*80)

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle('Scatter Plot: Predicción vs Demanda Real - PRIMERA SEMANA TEST',
              fontsize=14, fontweight='bold', y=1.02)

for idx, (nombre, y_pred) in enumerate(pred_primera.items()):
    ax = axes1[idx]
    
    # Calcular métricas
    mae, rmse, r2 = calcular_metricas(y_real_primera, y_pred)
    
    # Scatter plot
    ax.scatter(y_real_primera, y_pred, 
               color=colores[nombre], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Línea diagonal perfecta (y=x)
    min_val = min(y_real_primera.min(), y_pred.min())
    max_val = max(y_real_primera.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Predicción Perfecta', alpha=0.7)
    
    # Configuración
    ax.set_xlabel('Demanda Real (Qout) [m³/h]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Predicción (Qout) [m³/h]', fontsize=10, fontweight='bold')
    ax.set_title(f'{nombre}', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=9)
    
    # Anotación con métricas
    stats_text = f'MAE: {mae:,.0f} m³/h\nRMSE: {rmse:,.0f} m³/h\nR²: {r2:.4f}'
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
Path('outputs/figures').mkdir(parents=True, exist_ok=True)
fig1.savefig('outputs/figures/scatter_primera_semana_test.png', dpi=300, bbox_inches='tight')
print('✅ Guardado: outputs/figures/scatter_primera_semana_test.png')

# ==============================================================================
# 6. GENERAR SCATTER PLOTS - ÚLTIMA SEMANA
# ==============================================================================
print('\n5️⃣ Generando scatter plots - Última semana...')
print('-'*80)

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle('Scatter Plot: Predicción vs Demanda Real - ÚLTIMA SEMANA TEST',
              fontsize=14, fontweight='bold', y=1.02)

for idx, (nombre, y_pred) in enumerate(pred_ultima.items()):
    ax = axes2[idx]
    
    # Calcular métricas
    mae, rmse, r2 = calcular_metricas(y_real_ultima, y_pred)
    
    # Scatter plot
    ax.scatter(y_real_ultima, y_pred, 
               color=colores[nombre], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Línea diagonal perfecta (y=x)
    min_val = min(y_real_ultima.min(), y_pred.min())
    max_val = max(y_real_ultima.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Predicción Perfecta', alpha=0.7)
    
    # Configuración
    ax.set_xlabel('Demanda Real (Qout) [m³/h]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Predicción (Qout) [m³/h]', fontsize=10, fontweight='bold')
    ax.set_title(f'{nombre}', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=9)
    
    # Anotación con métricas
    stats_text = f'MAE: {mae:,.0f} m³/h\nRMSE: {rmse:,.0f} m³/h\nR²: {r2:.4f}'
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
fig2.savefig('outputs/figures/scatter_ultima_semana_test.png', dpi=300, bbox_inches='tight')
print('✅ Guardado: outputs/figures/scatter_ultima_semana_test.png')

# ==============================================================================
# 7. GENERAR SCATTER PLOTS COMBINADOS (2 FILAS x 3 COLUMNAS)
# ==============================================================================
print('\n6️⃣ Generando scatter plots combinados...')
print('-'*80)

fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))
fig3.suptitle('Scatter Plots Comparativos: Primera y Última Semana TEST',
              fontsize=14, fontweight='bold', y=0.995)

# Fila superior: Primera semana
for idx, (nombre, y_pred) in enumerate(pred_primera.items()):
    ax = axes3[0, idx]
    
    mae, rmse, r2 = calcular_metricas(y_real_primera, y_pred)
    
    ax.scatter(y_real_primera, y_pred, 
               color=colores[nombre], alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
    
    min_val = min(y_real_primera.min(), y_pred.min())
    max_val = max(y_real_primera.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfecta', alpha=0.7)
    
    ax.set_xlabel('Real (Qout) [m³/h]', fontsize=9, fontweight='bold')
    ax.set_ylabel('Predicción [m³/h]', fontsize=9, fontweight='bold')
    ax.set_title(f'{nombre} - PRIMERA SEMANA', fontsize=11, fontweight='bold', pad=8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=8)
    
    stats_text = f'MAE: {mae:,.0f}\nRMSE: {rmse:,.0f}\nR²: {r2:.3f}'
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Fila inferior: Última semana
for idx, (nombre, y_pred) in enumerate(pred_ultima.items()):
    ax = axes3[1, idx]
    
    mae, rmse, r2 = calcular_metricas(y_real_ultima, y_pred)
    
    ax.scatter(y_real_ultima, y_pred, 
               color=colores[nombre], alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
    
    min_val = min(y_real_ultima.min(), y_pred.min())
    max_val = max(y_real_ultima.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfecta', alpha=0.7)
    
    ax.set_xlabel('Real (Qout) [m³/h]', fontsize=9, fontweight='bold')
    ax.set_ylabel('Predicción [m³/h]', fontsize=9, fontweight='bold')
    ax.set_title(f'{nombre} - ÚLTIMA SEMANA', fontsize=11, fontweight='bold', pad=8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=8)
    
    stats_text = f'MAE: {mae:,.0f}\nRMSE: {rmse:,.0f}\nR²: {r2:.3f}'
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
fig3.savefig('outputs/figures/scatter_combinado_test.png', dpi=300, bbox_inches='tight')
print('✅ Guardado: outputs/figures/scatter_combinado_test.png')

# ==============================================================================
# 8. GENERAR SCATTER PLOT - TODO EL PERIODO TEST
# ==============================================================================
print('\n7️⃣ Generando scatter plots - TODO EL PERIODO TEST...')
print('-'*80)

fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))
fig4.suptitle('Scatter Plot: Predicción vs Demanda Real - TODO EL PERIODO TEST',
              fontsize=14, fontweight='bold', y=1.02)

for idx, (nombre, y_pred) in enumerate(predicciones_qout.items()):
    ax = axes4[idx]
    
    # Calcular métricas para todo el test
    mae, rmse, r2 = calcular_metricas(y_test, y_pred)
    
    # Scatter plot con muestreo para visualización (muy denso con 2255 puntos)
    # Graficamos todos los puntos pero con menor tamaño
    ax.scatter(y_test, y_pred, 
               color=colores[nombre], alpha=0.4, s=20, edgecolors='none')
    
    # Línea diagonal perfecta (y=x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Predicción Perfecta', alpha=0.7)
    
    # Configuración
    ax.set_xlabel('Demanda Real (Qout) [m³/h]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Predicción (Qout) [m³/h]', fontsize=10, fontweight='bold')
    ax.set_title(f'{nombre}', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=9)
    
    # Anotación con métricas
    stats_text = f'n = {len(y_test):,} registros\n'
    stats_text += f'MAE: {mae:,.0f} m³/h\nRMSE: {rmse:,.0f} m³/h\nR²: {r2:.4f}'
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
fig4.savefig('outputs/figures/scatter_completo_test.png', dpi=300, bbox_inches='tight')
print('✅ Guardado: outputs/figures/scatter_completo_test.png')

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================
print('\n' + '='*80)
print('RESUMEN DE SCATTER PLOTS GENERADOS')
print('='*80)
print('\n📊 Archivos generados:')
print('   1. outputs/figures/scatter_primera_semana_test.png')
print('   2. outputs/figures/scatter_ultima_semana_test.png')
print('   3. outputs/figures/scatter_combinado_test.png')
print('   4. outputs/figures/scatter_completo_test.png (TODO EL TEST)')
print('\n📈 Modelos evaluados:', len(predicciones_qout))
for nombre in predicciones_qout.keys():
    print(f'   ✅ {nombre}')
print('\n⏰ Periodos analizados:')
print(f'   • Primera semana: {len(primera_semana_idx)} horas')
print(f'   • Última semana: {len(ultima_semana_idx)} horas')
print(f'   • Test completo: {len(y_test)} horas')
print('\n' + '='*80)
print('✅ PROCESO COMPLETADO EXITOSAMENTE')
print('='*80)
