# -*- coding: utf-8 -*-
"""
Analisis 13: Features por Tipo (Interpretabilidad)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

print("=" * 80)
print("ANALISIS 13: FEATURES POR TIPO (INTERPRETABILIDAD)")
print("=" * 80)

plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path('outputs/interpretabilidad')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar modelo
print("\n[1] Cargando modelo...")
modelo = joblib.load('../models/forecasting/modelo_forecasting_xgboost.pkl')
with open('../models/forecasting/features.txt') as f:
    features = [line.strip() for line in f]
print(f"   OK - {len(features)} features cargadas")

# 2. Extraer importancia
print("\n[2] Extrayendo importancia de features...")
importancias = modelo.feature_importances_
df_importancia = pd.DataFrame({
    'feature': features,
    'importancia': importancias
}).sort_values('importancia', ascending=False)

print(f"   Total importancia: {df_importancia['importancia'].sum():.4f}")

# 3. Clasificar features por tipo
print("\n[3] Clasificando features por tipo...")

def clasificar_feature(feature_name):
    """Clasifica una feature segun su tipo"""
    feature_lower = feature_name.lower()
    
    # Temporales
    if any(palabra in feature_lower for palabra in ['hora', 'dia', 'mes', 'semana', 'periodo', 
                                                      'cal_', 'cos', 'sin', 'year']):
        return 'Temporal'
    
    # Autoregresivos
    elif any(palabra in feature_lower for palabra in ['lag', 'diff', 'ema', 'rolling', 
                                                        'win', 'ma_']):
        return 'Autoregresivo'
    
    # Climaticos
    elif any(palabra in feature_lower for palabra in ['temp', 'hr', 'mm', 'clima', 
                                                        'precipita', 'humedad']):
        return 'Climático'
    
    # Calendario / Eventos
    elif any(palabra in feature_lower for palabra in ['feriado', 'evento', 'festivo', 
                                                        'fin_de_semana', 'finde']):
        return 'Calendario'
    
    # Sistema / Operativo
    elif any(palabra in feature_lower for palabra in ['volumen', 'qin', 'q_net', 'estanque',
                                                        'capacidad', 'nivel']):
        return 'Sistema/Operativo'
    
    else:
        return 'Otro'

df_importancia['tipo'] = df_importancia['feature'].apply(clasificar_feature)

# 4. Agregar por tipo
print("\n[4] Agregando importancia por tipo...")
importancia_tipo = df_importancia.groupby('tipo')['importancia'].agg(['sum', 'count']).reset_index()
importancia_tipo = importancia_tipo.sort_values('sum', ascending=False)
importancia_tipo['porcentaje'] = (importancia_tipo['sum'] / importancia_tipo['sum'].sum()) * 100

print("\nIMPORTANCIA POR TIPO DE FEATURE:")
for _, row in importancia_tipo.iterrows():
    print(f"   {row['tipo']}: {row['porcentaje']:.2f}% ({row['count']} features)")

# 5. Crear visualizaciones
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
fig.suptitle('Análisis de Features por Tipo - Interpretabilidad del Modelo',
             fontsize=16, fontweight='bold', y=0.995)

# ============================================================================
# Subplot 1: Grafico de barras apiladas (top 20 features)
# ============================================================================
print("\n[5] Generando grafico de barras apiladas...")
ax1 = fig.add_subplot(gs[0, :])

top_n = 20
df_top = df_importancia.head(top_n)

# Colores por tipo
colores_tipo = {
    'Temporal': '#FF6B6B',
    'Autoregresivo': '#4ECDC4',
    'Climático': '#95E1D3',
    'Calendario': '#FFE66D',
    'Sistema/Operativo': '#C7CEEA',
    'Otro': '#CCCCCC'
}

# Preparar datos para barras apiladas
y_pos = np.arange(len(df_top))
left = np.zeros(len(df_top))

tipos_unicos = df_top['tipo'].unique()
for tipo in tipos_unicos:
    mask = df_top['tipo'] == tipo
    valores = df_top['importancia'].values * mask
    ax1.barh(y_pos, valores, left=left, color=colores_tipo.get(tipo, '#CCCCCC'),
             label=tipo, alpha=0.8, edgecolor='black')
    left += valores

ax1.set_yticks(y_pos)
ax1.set_yticklabels(df_top['feature'], fontsize=9)
ax1.set_xlabel('Importancia', fontsize=11, fontweight='bold')
ax1.set_title(f'Top {top_n} Features más Importantes (por tipo)',
              fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9, ncol=2)
ax1.grid(True, alpha=0.3, axis='x')

# Invertir eje Y para que el mas importante este arriba
ax1.invert_yaxis()

# ============================================================================
# Subplot 2: Grafico de torta - Distribucion por tipo
# ============================================================================
print("\n[6] Generando grafico de torta...")
ax2 = fig.add_subplot(gs[1, 0])

colores_ordenados = [colores_tipo.get(tipo, '#CCCCCC') for tipo in importancia_tipo['tipo']]

wedges, texts, autotexts = ax2.pie(importancia_tipo['porcentaje'],
                                     labels=importancia_tipo['tipo'],
                                     autopct='%1.1f%%',
                                     colors=colores_ordenados,
                                     startangle=90,
                                     explode=[0.05 if i == 0 else 0 for i in range(len(importancia_tipo))],
                                     textprops={'fontsize': 10})

# Mejorar formato de porcentajes
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

ax2.set_title('Distribución de Importancia por Tipo de Feature',
              fontsize=12, fontweight='bold')

# ============================================================================
# Subplot 3: Grafico de barras - Importancia total por tipo
# ============================================================================
print("\n[7] Generando grafico de barras por tipo...")
ax3 = fig.add_subplot(gs[1, 1])

colores_bars = [colores_tipo.get(tipo, '#CCCCCC') for tipo in importancia_tipo['tipo']]

bars = ax3.bar(range(len(importancia_tipo)), importancia_tipo['porcentaje'],
               color=colores_bars, alpha=0.8, edgecolor='black')

ax3.set_xticks(range(len(importancia_tipo)))
ax3.set_xticklabels(importancia_tipo['tipo'], rotation=45, ha='right', fontsize=10)
ax3.set_ylabel('Importancia Total (%)', fontsize=11, fontweight='bold')
ax3.set_title('Importancia Acumulada por Tipo de Feature',
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Agregar valores en las barras
for bar, pct, count in zip(bars, importancia_tipo['porcentaje'], importancia_tipo['count']):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{pct:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9)

# Agregar interpretacion
textstr = 'INTERPRETACIÓN:\n\n'
tipo_principal = importancia_tipo.iloc[0]
textstr += f'• {tipo_principal["tipo"]} domina\n'
textstr += f'  con {tipo_principal["porcentaje"]:.1f}%\n\n'
textstr += f'• {tipo_principal["count"]} features de\n'
textstr += f'  este tipo en el modelo'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax3.text(0.98, 0.98, textstr, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_features_por_tipo.png', dpi=300, bbox_inches='tight')
print(f"\n   OK - Guardado: {OUTPUT_DIR / '01_features_por_tipo.png'}")
plt.close()

# 6. Analisis detallado por tipo
print("\n[8] Generando analisis detallado por tipo...")

# Crear segunda figura con detalles
fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
fig2.suptitle('Detalle de Features por Tipo',
              fontsize=16, fontweight='bold', y=0.995)
axes = axes.flatten()

for idx, tipo in enumerate(importancia_tipo['tipo']):
    if idx >= 6:
        break
        
    ax = axes[idx]
    df_tipo = df_importancia[df_importancia['tipo'] == tipo].head(10)
    
    if len(df_tipo) > 0:
        bars = ax.barh(range(len(df_tipo)), df_tipo['importancia'],
                        color=colores_tipo.get(tipo, '#CCCCCC'),
                        alpha=0.8, edgecolor='black')
        
        ax.set_yticks(range(len(df_tipo)))
        ax.set_yticklabels(df_tipo['feature'], fontsize=8)
        ax.set_xlabel('Importancia', fontsize=9, fontweight='bold')
        ax.set_title(f'{tipo}\n(Top 10)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Stats en texto
        total_tipo = importancia_tipo[importancia_tipo['tipo'] == tipo]['porcentaje'].values[0]
        count_tipo = importancia_tipo[importancia_tipo['tipo'] == tipo]['count'].values[0]
        textstr = f'{total_tipo:.1f}%\n({count_tipo} features)'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
    else:
        ax.text(0.5, 0.5, 'Sin features', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{tipo}', fontsize=10, fontweight='bold')

# Ocultar axes vacios
for idx in range(len(importancia_tipo), 6):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_detalle_por_tipo.png', dpi=300, bbox_inches='tight')
print(f"   OK - Guardado: {OUTPUT_DIR / '02_detalle_por_tipo.png'}")
plt.close()

# 7. Guardar datos
print("\n[9] Guardando datos...")

# Tabla de importancia por tipo
importancia_tipo.to_csv(OUTPUT_DIR / '01_importancia_por_tipo.csv', 
                        index=False, float_format='%.4f')

# Tabla completa con clasificacion
df_importancia.to_csv(OUTPUT_DIR / '01_features_clasificadas.csv',
                      index=False, float_format='%.4f')

# Resumen estadistico
resumen = {
    'total_features': len(features),
    'tipos_identificados': len(importancia_tipo),
    'distribucion_por_tipo': {}
}

for _, row in importancia_tipo.iterrows():
    resumen['distribucion_por_tipo'][row['tipo']] = {
        'importancia_pct': float(row['porcentaje']),
        'num_features': int(row['count']),
        'importancia_promedio': float(row['sum'] / row['count'])
    }

import json
with open(OUTPUT_DIR / '01_resumen_tipos.json', 'w') as f:
    json.dump(resumen, f, indent=2)

print(f"   OK - Datos guardados")

print("\n" + "=" * 80)
print("RESUMEN DE INTERPRETABILIDAD")
print("=" * 80)
print(f"Total de features: {len(features)}")
print(f"\nTipo dominante: {importancia_tipo.iloc[0]['tipo']}")
print(f"  Importancia: {importancia_tipo.iloc[0]['porcentaje']:.2f}%")
print(f"  Numero de features: {int(importancia_tipo.iloc[0]['count'])}")
print("\nTop 3 tipos:")
for i in range(min(3, len(importancia_tipo))):
    row = importancia_tipo.iloc[i]
    print(f"  {i+1}. {row['tipo']}: {row['porcentaje']:.2f}% ({int(row['count'])} features)")
print("=" * 80)
print("ANALISIS COMPLETADO")
print("=" * 80)
