# -*- coding: utf-8 -*-
"""
Analisis 11: Comparacion Train vs Validation vs Test
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json

print("=" * 80)
print("ANALISIS 11: COMPARACION TRAIN / VALIDATION / TEST")
print("=" * 80)

plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path('outputs/inferencial')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar modelo
print("\n[1] Cargando modelo...")
modelo = joblib.load('../models/forecasting/modelo_forecasting_xgboost.pkl')
with open('../models/forecasting/features.txt') as f:
    features = [line.strip() for line in f]
print(f"   OK - Modelo con {len(features)} features")

# 2. Funcion para calcular metricas
def calcular_metricas(y_real, y_pred, nombre_set):
    """Calcula metricas de evaluacion"""
    mse = np.mean((y_real - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_real - y_pred))
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    r2 = 1 - (np.sum((y_real - y_pred) ** 2) / np.sum((y_real - np.mean(y_real)) ** 2))
    
    print(f"\n   {nombre_set}:")
    print(f"     RMSE: {rmse:.2f} m³")
    print(f"     MAE: {mae:.2f} m³")
    print(f"     MAPE: {mape:.2f}%")
    print(f"     R²: {r2:.4f}")
    
    return {
        'Dataset': nombre_set,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Registros': len(y_real)
    }

# 3. Procesar cada dataset
print("\n[2] Procesando datasets...")
metricas_all = []

datasets = {
    'Train': '../data/processed/data_train.csv',
    'Validation': '../data/processed/data_validation.csv',
    'Test': '../data/processed/data_test.csv'
}

for nombre, ruta in datasets.items():
    print(f"\n   [{nombre}] Cargando y prediciendo...")
    df = pd.read_csv(ruta)
    
    # Preparar features
    features_disponibles = [f for f in features if f in df.columns]
    X = df[features_disponibles].copy()
    
    for feature in features:
        if feature not in features_disponibles:
            X[feature] = 0
    
    X = X[features]
    y_real = df['Q_net_m3h'].values
    y_pred = modelo.predict(X)
    
    # Calcular metricas
    metricas = calcular_metricas(y_real, y_pred, nombre)
    metricas_all.append(metricas)

# 4. Crear DataFrame con metricas
df_metricas = pd.DataFrame(metricas_all)

# 5. Crear visualizaciones
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Comparación de Desempeño del Modelo: Train vs Validation vs Test',
             fontsize=16, fontweight='bold', y=1.00)

# ============================================================================
# Subplot 1: Tabla de metricas
# ============================================================================
print("\n[3] Generando tabla comparativa...")
ax1 = axes[0]
ax1.axis('tight')
ax1.axis('off')

# Preparar datos para tabla
tabla_data = []
for _, row in df_metricas.iterrows():
    tabla_data.append([
        row['Dataset'],
        f"{row['RMSE']:.2f}",
        f"{row['MAE']:.2f}",
        f"{row['MAPE']:.2f}%",
        f"{row['R2']:.4f}",
        f"{row['Registros']:,}"
    ])

tabla = ax1.table(cellText=tabla_data,
                  colLabels=['Dataset', 'RMSE (m3/hr)', 'MAE (m3/hr)', 'MAPE (%)', 'R²', 'Registros'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])

tabla.auto_set_font_size(False)
tabla.set_fontsize(11)
tabla.scale(1, 2.5)

# Colorear header
for i in range(6):
    tabla[(0, i)].set_facecolor('#4472C4')
    tabla[(0, i)].set_text_props(weight='bold', color='white')

# Colorear filas
colores = ['#E7E6E6', '#F2F2F2', '#E7E6E6']
for i in range(1, 4):
    for j in range(6):
        tabla[(i, j)].set_facecolor(colores[i-1])
        
    # Resaltar mejor metrica en cada columna (excepto registros)
    if i == 1:  # Train
        tabla[(i, 0)].set_facecolor('#C6E0B4')  # Verde claro
    elif i == 3:  # Test
        tabla[(i, 0)].set_facecolor('#FFE699')  # Amarillo claro

ax1.set_title('Tabla Comparativa de Métricas', fontsize=13, fontweight='bold', pad=20)

# ============================================================================
# Subplot 2: Grafico de barras agrupadas
# ============================================================================
print("\n[4] Generando grafico de barras...")
ax2 = axes[1]

# Metricas a graficar (normalizadas)
x = np.arange(len(df_metricas))
width = 0.2

# Normalizar metricas para visualizacion
rmse_norm = df_metricas['RMSE'] / df_metricas['RMSE'].max()
mae_norm = df_metricas['MAE'] / df_metricas['MAE'].max()
mape_norm = df_metricas['MAPE'] / df_metricas['MAPE'].max()
r2_values = df_metricas['R2']

bars1 = ax2.bar(x - width*1.5, rmse_norm, width, label='RMSE (norm)', 
                color='steelblue', alpha=0.8, edgecolor='black')
bars2 = ax2.bar(x - width*0.5, mae_norm, width, label='MAE (norm)', 
                color='lightblue', alpha=0.8, edgecolor='black')
bars3 = ax2.bar(x + width*0.5, mape_norm, width, label='MAPE (norm)', 
                color='orange', alpha=0.8, edgecolor='black')
bars4 = ax2.bar(x + width*1.5, r2_values, width, label='R² (sin norm)', 
                color='green', alpha=0.8, edgecolor='black')

ax2.set_xlabel('Dataset', fontsize=11, fontweight='bold')
ax2.set_ylabel('Valor Normalizado / R²', fontsize=11, fontweight='bold')
ax2.set_title('Comparación Visual de Métricas por Dataset', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(df_metricas['Dataset'])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, 1.1])

# Agregar valores en las barras
def agregar_valores(bars, valores, formato='.3f'):
    for bar, valor in zip(bars, valores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{valor:{formato}}', ha='center', va='bottom', fontsize=8)

agregar_valores(bars4, r2_values, '.4f')

# Agregar interpretacion
textstr = 'INTERPRETACIÓN:\n'
textstr += '• Errores bajos (RMSE, MAE, MAPE)\n'
textstr += '  indican mejor predicción\n'
textstr += '• R² cercano a 1 indica\n'
textstr += '  excelente ajuste\n'
textstr += '• Test muestra desempeño\n'
textstr += '  en datos nunca vistos'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_comparacion_datasets.png', dpi=300, bbox_inches='tight')
print(f"\n   OK - Guardado: {OUTPUT_DIR / '03_comparacion_datasets.png'}")
plt.close()

# 6. Analisis de diferencias
print("\n[5] Analizando diferencias entre datasets...")

# Diferencia relativa respecto a Train
diff_val_train = ((df_metricas.loc[1, 'RMSE'] - df_metricas.loc[0, 'RMSE']) / 
                  df_metricas.loc[0, 'RMSE'] * 100)
diff_test_train = ((df_metricas.loc[2, 'RMSE'] - df_metricas.loc[0, 'RMSE']) / 
                   df_metricas.loc[0, 'RMSE'] * 100)

print(f"\n   Diferencia RMSE Validation vs Train: {diff_val_train:+.2f}%")
print(f"   Diferencia RMSE Test vs Train: {diff_test_train:+.2f}%")

# Evaluar overfitting
if abs(diff_test_train) < 15:
    print("\n   [OK] El modelo generaliza bien (diferencia < 15%)")
    evaluacion = "Modelo generaliza correctamente"
elif abs(diff_test_train) < 30:
    print("\n   [PRECAUCION] Posible sobreajuste moderado (15% < diferencia < 30%)")
    evaluacion = "Posible sobreajuste moderado"
else:
    print("\n   [ALERTA] Sobreajuste significativo (diferencia > 30%)")
    evaluacion = "Sobreajuste significativo"

# Agregar evaluacion a DataFrame
df_metricas['Diff_vs_Train_%'] = [0, diff_val_train, diff_test_train]
df_metricas['Evaluacion'] = ['Baseline', 'Validación', evaluacion]

# 7. Guardar resultados
print("\n[6] Guardando resultados...")
df_metricas.to_csv(OUTPUT_DIR / '03_comparacion_metricas.csv', index=False, float_format='%.4f')

# Guardar resumen en JSON
resumen = {
    'train': {k: v for k, v in metricas_all[0].items()},
    'validation': {k: v for k, v in metricas_all[1].items()},
    'test': {k: v for k, v in metricas_all[2].items()},
    'diferencias': {
        'rmse_val_vs_train_pct': float(diff_val_train),
        'rmse_test_vs_train_pct': float(diff_test_train)
    },
    'evaluacion': evaluacion
}

with open(OUTPUT_DIR / '03_comparacion_resumen.json', 'w') as f:
    json.dump(resumen, f, indent=2)

print(f"   OK - Datos guardados")

print("\n" + "=" * 80)
print("ANALISIS COMPLETADO")
print("=" * 80)
print(f"\nEVALUACION FINAL: {evaluacion}")
print(f"R² en Test: {df_metricas.loc[2, 'R2']:.4f} (excelente si > 0.95)")
print("=" * 80)
