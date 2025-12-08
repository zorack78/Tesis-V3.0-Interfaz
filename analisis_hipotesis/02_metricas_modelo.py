# -*- coding: utf-8 -*-
"""
Analisis 2: Metricas del Modelo
- Visualizacion de metricas del modelo entrenado
- Interpretacion de resultados
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 80)
print("ANALISIS 2: METRICAS DEL MODELO")
print("=" * 80)

# Configuracion
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar metricas
print("\n[1] Cargando metricas del modelo...")
with open('../models/forecasting/metricas.json') as f:
    metricas = json.load(f)

print("\nMETRICAS DEL MODELO XGBOOST V3.0:")
print(f"   RMSE: {metricas['rmse']:.2f} m3/hr")
print(f"   MAE: {metricas['mae']:.2f} m3/hr")
print(f"   MAPE: {metricas['mape']:.2f}%")
print(f"   R2: {metricas['r2']:.4f}")
print(f"   Registros de prueba: {metricas['n_test']:,}")

# 2. Crear tabla de metricas
df_metricas = pd.DataFrame({
    'Metrica': ['RMSE', 'MAE', 'MAPE', 'R2', 'N_test'],
    'Valor': [
        metricas['rmse'], 
        metricas['mae'], 
        metricas['mape'], 
        metricas['r2'],
        metricas['n_test']
    ],
    'Unidad': ['m3/hr', 'm3/hr', '%', 'adimensional', 'registros'],
    'Interpretacion': [
        'Error cuadratico medio',
        'Error absoluto medio',
        'Error porcentual absoluto medio',
        'Coeficiente de determinacion',
        'Tamano del conjunto de prueba'
    ]
})

# 3. Guardar tabla
output_csv = OUTPUT_DIR / '02_metricas_modelo.csv'
df_metricas.to_csv(output_csv, index=False)
print(f"\nOK - Tabla guardada: {output_csv}")

# 4. Crear visualizacion
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Barra de errores
errores = [metricas['rmse'], metricas['mae']]
nombres_errores = ['RMSE', 'MAE']
axes[0, 0].bar(nombres_errores, errores, color=['coral', 'lightblue'])
axes[0, 0].set_ylabel('Error (m3/hr)', fontsize=12)
axes[0, 0].set_title('Metricas de Error', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(errores):
    axes[0, 0].text(i, v + 10, f'{v:.1f}', ha='center', fontweight='bold')

# Panel 2: MAPE
axes[0, 1].bar(['MAPE'], [metricas['mape']], color='orange', width=0.5)
axes[0, 1].set_ylabel('Error Porcentual (%)', fontsize=12)
axes[0, 1].set_title('Error Porcentual Absoluto Medio', 
                     fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].text(0, metricas['mape'] + 1, f"{metricas['mape']:.2f}%", 
               ha='center', fontweight='bold')

# Panel 3: R2
axes[1, 0].bar(['R2'], [metricas['r2']], color='green', width=0.5)
axes[1, 0].set_ylabel('R2 Score', fontsize=12)
axes[1, 0].set_title('Coeficiente de Determinacion', 
                     fontsize=13, fontweight='bold')
axes[1, 0].set_ylim([0, 1.1])
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].text(0, metricas['r2'] + 0.02, f"{metricas['r2']:.4f}", 
               ha='center', fontweight='bold')

# Panel 4: Tabla resumen
axes[1, 1].axis('off')
tabla_texto = f"""
RESUMEN DE METRICAS
{'=' * 40}

RMSE:  {metricas['rmse']:.2f} m3/hr
MAE:   {metricas['mae']:.2f} m3/hr
MAPE:  {metricas['mape']:.2f}%
R2:    {metricas['r2']:.4f}

Registros de prueba: {metricas['n_test']:,}

{'=' * 40}
Interpretacion:
- R2 cercano a 1 indica excelente ajuste
- RMSE/MAE bajos indican buen precision
- MAPE del {metricas['mape']:.1f}% es aceptable
"""
axes[1, 1].text(0.1, 0.5, tabla_texto, fontsize=11, family='monospace',
               verticalalignment='center')

plt.tight_layout()
output_file = OUTPUT_DIR / '02_metricas_visualizacion.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"OK - Visualizacion guardada: {output_file}")
plt.close()

print("\n" + "=" * 80)
print("ANALISIS COMPLETADO")
print("=" * 80)
