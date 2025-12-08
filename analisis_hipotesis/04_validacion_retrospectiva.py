# -*- coding: utf-8 -*-
"""
Validacion Retrospectiva - Datos de Test (15% final)
Simula predicciones con la interfaz sobre datos nunca vistos por el modelo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

print("=" * 80)
print("VALIDACION RETROSPECTIVA - DATOS DE TEST")
print("=" * 80)

# Configuracion
plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path('outputs/validacion_retrospectiva')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar datos de test
print("\n[1] Cargando datos de test...")
df_test = pd.read_csv('../data/processed/data_test.csv')
df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
print(f"   OK - {len(df_test):,} registros de test (15%)")
print(f"   Periodo: {df_test['timestamp'].min()} -> {df_test['timestamp'].max()}")

# 2. Cargar modelo y features
print("\n[2] Cargando modelo...")
modelo = joblib.load('../models/forecasting/modelo_forecasting_xgboost.pkl')
with open('../models/forecasting/features.txt') as f:
    features = [line.strip() for line in f]
print(f"   OK - Modelo cargado ({len(features)} features)")

# 3. Cargar dataset completo (para contexto historico, como hace la interfaz)
print("\n[3] Cargando dataset completo para contexto...")
df_completo = pd.read_csv('../data/processed/dataset_features_completo.csv')
df_completo['timestamp'] = pd.to_datetime(df_completo['timestamp'])
print(f"   OK - {len(df_completo):,} registros historicos cargados")

# 4. Cargar datos de train para contexto (excluir test)
df_train = pd.read_csv('../data/processed/data_train.csv')
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
df_val = pd.read_csv('data/processed/data_validation.csv')
df_val['timestamp'] = pd.to_datetime(df_val['timestamp'])

# Concatenar train + val para usar como contexto historico
df_historico = pd.concat([df_train, df_val], ignore_index=True)
print(f"   OK - {len(df_historico):,} registros historicos (train+val)")

# 5. Generar predicciones usando features existentes
print("\n[4] Generando predicciones sobre datos de test...")
print("   (Usando solo features disponibles en el dataset)")

# Verificar que features estan disponibles
features_disponibles = [f for f in features if f in df_test.columns]
print(f"   Features disponibles: {len(features_disponibles)}/{len(features)}")

if len(features_disponibles) >= 30:  # Al menos 30 features
    X_test = df_test[features_disponibles].fillna(0)
    y_test = df_test['Q_net_m3h']
    
    # Completar features faltantes con ceros (como hace la interfaz)
    for feature in features:
        if feature not in features_disponibles:
            X_test[feature] = 0
    
    # Reordenar columnas segun el orden del modelo
    X_test = X_test[features]
    
    # Predecir
    y_pred = modelo.predict(X_test)
    
    # Crear DataFrame de resultados
    df_resultados = pd.DataFrame({
        'timestamp': df_test['timestamp'],
        'Q_net_real': y_test,
        'Q_net_pred': y_pred,
        'error': y_test - y_pred,
        'error_abs': np.abs(y_test - y_pred),
        'error_pct': np.abs((y_test - y_pred) / y_test * 100)
    })
    
    print(f"   OK - {len(df_resultados):,} predicciones generadas")
else:
    print(f"   ERROR - Solo {len(features_disponibles)} features disponibles")
    print("   Se requieren al menos 30 features")
    exit(1)

# 5. Calcular metricas
print("\n[5] Calculando metricas de validacion...")

rmse = np.sqrt(np.mean(df_resultados['error']**2))
mae = np.mean(df_resultados['error_abs'])
mape = np.mean(df_resultados['error_pct'])
r2 = 1 - (np.sum(df_resultados['error']**2) / 
          np.sum((df_resultados['Q_net_real'] - df_resultados['Q_net_real'].mean())**2))

print("\nMETRICAS DE VALIDACION RETROSPECTIVA (TEST SET):")
print(f"   RMSE: {rmse:.2f} m3/hr")
print(f"   MAE: {mae:.2f} m3/hr")
print(f"   MAPE: {mape:.2f}%")
print(f"   R2: {r2:.4f}")

# Guardar resultados
csv_output = OUTPUT_DIR / 'resultados_validacion.csv'
df_resultados.to_csv(csv_output, index=False)
print(f"\n   Guardado: {csv_output}")

# 6. Grafica 1: Serie temporal (primeros 30 dias)
print("\n[6] Generando grafica 1: Serie temporal...")

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Panel 1: Primeros 30 dias
n_30d = min(30 * 24, len(df_resultados))
df_30d = df_resultados.iloc[:n_30d]

axes[0].plot(df_30d['timestamp'], df_30d['Q_net_real'], 
            label='Demanda Real', alpha=0.8, linewidth=1.5, color='blue')
axes[0].plot(df_30d['timestamp'], df_30d['Q_net_pred'], 
            label='Prediccion Modelo', alpha=0.8, linewidth=1.5, 
            linestyle='--', color='orange')
axes[0].fill_between(df_30d['timestamp'], df_30d['Q_net_real'], 
                     df_30d['Q_net_pred'], alpha=0.2, color='red')
axes[0].set_ylabel('Q_net (m3/hr)', fontsize=11)
axes[0].set_title('Validacion Retrospectiva - Primeros 30 Dias (Test Set)', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10, loc='upper right')
axes[0].grid(True, alpha=0.3)

# Panel 2: Error por hora del dia
df_resultados['hora'] = df_resultados['timestamp'].dt.hour
error_por_hora = df_resultados.groupby('hora')['error_abs'].agg(['mean', 'std'])

axes[1].bar(error_por_hora.index, error_por_hora['mean'], 
           alpha=0.7, color='coral', yerr=error_por_hora['std'], 
           capsize=5, error_kw={'linewidth': 1.5})
axes[1].set_xlabel('Hora del Dia', fontsize=11)
axes[1].set_ylabel('Error Absoluto Medio (m3/hr)', fontsize=11)
axes[1].set_title('Error de Prediccion por Hora del Dia', 
                  fontsize=13, fontweight='bold')
axes[1].set_xticks(range(24))
axes[1].grid(True, alpha=0.3, axis='y')

# Panel 3: Distribucion de errores
axes[2].hist(df_resultados['error'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[2].axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
axes[2].axvline(df_resultados['error'].mean(), color='green', 
               linestyle='--', linewidth=2, 
               label=f"Media = {df_resultados['error'].mean():.1f}")
axes[2].set_xlabel('Error (m3/hr)', fontsize=11)
axes[2].set_ylabel('Frecuencia', fontsize=11)
axes[2].set_title('Distribucion de Errores de Prediccion', 
                  fontsize=13, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
output_file = OUTPUT_DIR / '01_serie_temporal_validacion.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   OK - Guardado: {output_file}")
plt.close()

# 7. Grafica 2: Scatter prediccion vs real
print("\n[7] Generando grafica 2: Scatter prediccion vs real...")

fig, ax = plt.subplots(figsize=(10, 10))

scatter = ax.scatter(df_resultados['Q_net_real'], df_resultados['Q_net_pred'], 
                    alpha=0.5, s=30, c=df_resultados['error_abs'], 
                    cmap='RdYlGn_r', edgecolors='gray', linewidths=0.5)

# Linea perfecta
min_val = min(df_resultados['Q_net_real'].min(), df_resultados['Q_net_pred'].min())
max_val = max(df_resultados['Q_net_real'].max(), df_resultados['Q_net_pred'].max())
ax.plot([min_val, max_val], [min_val, max_val], 
        'r--', linewidth=2, label='Prediccion Perfecta', zorder=5)

# Banda de confianza +/- 500 m3/hr
ax.plot([min_val, max_val], [min_val - 500, max_val - 500],
        'gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.plot([min_val, max_val], [min_val + 500, max_val + 500],
        'gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.fill_between([min_val, max_val],
                [min_val - 500, max_val - 500],
                [min_val + 500, max_val + 500],
                alpha=0.1, color='gray', label='Banda +/- 500 m3/hr')

ax.set_xlabel('Q_net Real (m3/hr)', fontsize=12, fontweight='bold')
ax.set_ylabel('Q_net Predicho (m3/hr)', fontsize=12, fontweight='bold')
title_text = f'Prediccion vs Real - Test Set (Datos Nunca Vistos)\n'
title_text += f'R2={r2:.4f}, RMSE={rmse:.2f} m3/hr, MAE={mae:.2f} m3/hr'
ax.set_title(title_text, fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Error Absoluto (m3/hr)', fontsize=11)

plt.tight_layout()
output_file = OUTPUT_DIR / '02_scatter_prediccion_real.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   OK - Guardado: {output_file}")
plt.close()

# 8. Grafica 3: Analisis por segmentos
print("\n[8] Generando grafica 3: Analisis por segmentos...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Error por rango de temperatura (si hay columna temp)
temp_col = None
for col in ['temp', 'clima_temp_c', 'temperatura']:
    if col in df_test.columns:
        temp_col = col
        break

if temp_col:
    df_resultados['temp'] = df_test[temp_col].values
    df_resultados['rango_temp'] = pd.cut(df_resultados['temp'], 
                                         bins=[-100, 15, 20, 100], 
                                         labels=['Frio (<15C)', 'Normal (15-20C)', 'Calor (>20C)'])
    
    df_resultados.boxplot(column='error_abs', by='rango_temp', ax=axes[0, 0])
    axes[0, 0].set_title('Error por Rango de Temperatura', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Rango de Temperatura', fontsize=10)
    axes[0, 0].set_ylabel('Error Absoluto (m3/hr)', fontsize=10)
    axes[0, 0].get_figure().suptitle('')
    plt.sca(axes[0, 0])
    plt.xticks(rotation=15)
else:
    axes[0, 0].text(0.5, 0.5, 'Datos de temperatura\nno disponibles', 
                   ha='center', va='center', fontsize=12)
    axes[0, 0].set_title('Error por Rango de Temperatura', fontsize=12, fontweight='bold')

# Panel 2: Error por dia de la semana
df_resultados['dia_semana'] = df_resultados['timestamp'].dt.dayofweek
dias_nombres = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']
error_dia = df_resultados.groupby('dia_semana')['error_abs'].agg(['mean', 'std'])

axes[0, 1].bar(range(7), error_dia['mean'], alpha=0.7, color='skyblue',
              yerr=error_dia['std'], capsize=5)
axes[0, 1].set_xticks(range(7))
axes[0, 1].set_xticklabels(dias_nombres, rotation=0)
axes[0, 1].set_xlabel('Dia de la Semana', fontsize=10)
axes[0, 1].set_ylabel('Error Absoluto Medio (m3/hr)', fontsize=10)
axes[0, 1].set_title('Error por Dia de la Semana', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Panel 3: MAPE por hora
mape_hora = df_resultados.groupby('hora')['error_pct'].mean()
axes[1, 0].plot(mape_hora.index, mape_hora.values, marker='o', 
               linewidth=2, markersize=6, color='darkgreen')
axes[1, 0].axhline(mape, color='red', linestyle='--', 
                  linewidth=1.5, label=f'MAPE promedio: {mape:.1f}%')
axes[1, 0].set_xlabel('Hora del Dia', fontsize=10)
axes[1, 0].set_ylabel('MAPE (%)', fontsize=10)
axes[1, 0].set_title('MAPE por Hora del Dia', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(range(0, 24, 2))
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Percentiles de error
percentiles = [10, 25, 50, 75, 90, 95, 99]
valores_p = [np.percentile(df_resultados['error_abs'], p) for p in percentiles]

axes[1, 1].bar(range(len(percentiles)), valores_p, alpha=0.7, color='coral')
axes[1, 1].set_xticks(range(len(percentiles)))
axes[1, 1].set_xticklabels([f'P{p}' for p in percentiles])
axes[1, 1].set_xlabel('Percentil', fontsize=10)
axes[1, 1].set_ylabel('Error Absoluto (m3/hr)', fontsize=10)
axes[1, 1].set_title('Percentiles de Error Absoluto', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

for i, v in enumerate(valores_p):
    axes[1, 1].text(i, v + 20, f'{v:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
output_file = OUTPUT_DIR / '03_analisis_segmentos.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   OK - Guardado: {output_file}")
plt.close()

# 9. Guardar metricas
print("\n[9] Guardando metricas...")

metricas = pd.DataFrame({
    'Metrica': ['RMSE', 'MAE', 'MAPE', 'R2', 'Registros', 'Periodo'],
    'Valor': [f'{rmse:.2f}', f'{mae:.2f}', f'{mape:.2f}', f'{r2:.4f}',
              f'{len(df_resultados):,}', 
              f'{df_resultados["timestamp"].min()} - {df_resultados["timestamp"].max()}'],
    'Unidad': ['m3/hr', 'm3/hr', '%', '', '', '']
})

csv_output = OUTPUT_DIR / 'metricas_validacion.csv'
metricas.to_csv(csv_output, index=False)
print(f"   Guardado: {csv_output}")

print("\n" + "=" * 80)
print("VALIDACION RETROSPECTIVA COMPLETADA")
print("=" * 80)
print(f"\nResultados guardados en: {OUTPUT_DIR.absolute()}")
print("\nRESUMEN:")
print(f"  - Dataset de test: {len(df_resultados):,} predicciones (15% de datos)")
print(f"  - R2: {r2:.4f} (modelo explica {r2*100:.2f}% de la varianza)")
print(f"  - RMSE: {rmse:.2f} m3/hr")
print(f"  - MAE: {mae:.2f} m3/hr (error promedio)")
print(f"  - MAPE: {mape:.2f}% (error porcentual)")
