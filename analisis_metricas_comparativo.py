"""
Análisis comparativo de cálculo de métricas en el proyecto
- NO modifica ningún archivo
- Solo lee y compara
- Genera reporte de inconsistencias
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import re

print("="*80)
print("ANÁLISIS 1: COMPARACIÓN DE CÁLCULO DE MÉTRICAS")
print("="*80)

# ==============================================================================
# 1. IDENTIFICAR ARCHIVOS QUE CALCULAN MÉTRICAS
# ==============================================================================

archivos_con_metricas = {
    'Interfaz Gradio': 'interfaz_planificacion_qin_v1.py',
    'Modelo Gradio': 'modelo_gradio.py',
    'Gráficos Comparativa': 'graficos_comparativa_test.py',
    'Scatter Plots': 'scatter_plots_test.py',
    'Check R2': 'check_r2.py',
}

resultados = {}

# ==============================================================================
# 2. ANALIZAR CADA ARCHIVO
# ==============================================================================

for nombre, archivo in archivos_con_metricas.items():
    filepath = Path(archivo)
    
    if not filepath.exists():
        print(f"\n⚠️  {nombre}: Archivo no encontrado - {archivo}")
        continue
    
    print(f"\n{'='*80}")
    print(f"📄 Analizando: {nombre}")
    print(f"   Archivo: {archivo}")
    print('='*80)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        contenido = f.read()
    
    analisis = {
        'archivo': archivo,
        'target': [],
        'split_method': None,
        'periodo_test': None,
        'calculo_qout': None,
        'modelo_usado': [],
        'metricas_calculadas': [],
        'data_leakage_check': None,
    }
    
    # Buscar targets usados
    if 'Q_net_m3h' in contenido:
        analisis['target'].append('Q_net_m3h')
    if 'Qout' in contenido or 'qout' in contenido or 'y_test_qout' in contenido:
        analisis['target'].append('Qout (calculado)')
    if 'sist_Vtotal_m3' in contenido:
        analisis['target'].append('sist_Vtotal_m3')
    if 'Volumen_Total_m3' in contenido:
        analisis['target'].append('Volumen_Total_m3')
    
    # Buscar método de split
    if 'train_end = int(n * 0.70)' in contenido or 'train_end = int(n * 0.7)' in contenido:
        if '0.85' in contenido:
            analisis['split_method'] = '70/15/15'
        else:
            analisis['split_method'] = '70/30'
    elif 'train_end = int(n * 0.85)' in contenido:
        analisis['split_method'] = '85/15'
    
    # Buscar fechas específicas
    if '2025-03-22' in contenido or '2025-03-23' in contenido:
        analisis['periodo_test'] = 'Desde 2025-03-22/23'
    if '2025-06-26' in contenido:
        analisis['periodo_test'] = 'Desde 2025-06-26'
    if '2025-09-30' in contenido:
        if analisis['periodo_test']:
            analisis['periodo_test'] += ' hasta 2025-09-30'
        else:
            analisis['periodo_test'] = 'Hasta 2025-09-30'
    
    # Buscar cálculo de Qout
    patrones_qout = [
        r'qin.*?-.*?q_?net',
        r'perfil.*?qin',
        r'groupby\([\'"]hora[\'"]\).*?median',
        r'sist_Qin_m3h.*?-.*?Q_net',
        r'df_train.*?groupby',
        r'qin_test\s*=',
    ]
    
    metodos_qout = []
    for patron in patrones_qout:
        matches = re.findall(patron, contenido, re.IGNORECASE)
        if matches:
            metodos_qout.append(patron)
    
    if metodos_qout:
        analisis['calculo_qout'] = f'Sí ({len(metodos_qout)} patrones encontrados)'
    
    # Buscar prevención de data leakage
    if 'train_end' in contenido and 'iloc[:train_end]' in contenido:
        analisis['data_leakage_check'] = 'Sí (usa solo train para perfil)'
    elif 'data leakage' in contenido.lower():
        analisis['data_leakage_check'] = 'Mencionado en comentarios'
    
    # Buscar modelos usados
    if 'lgbm' in contenido.lower() or 'lightgbm' in contenido.lower():
        analisis['modelo_usado'].append('LightGBM')
    if 'xgboost' in contenido.lower() or 'xgb' in contenido.lower():
        analisis['modelo_usado'].append('XGBoost')
    if 'randomforest' in contenido.lower() or 'RandomForest' in contenido:
        analisis['modelo_usado'].append('RandomForest')
    
    # Buscar métricas calculadas
    metricas_buscadas = {
        'MAE': r'mean_absolute_error|MAE',
        'RMSE': r'mean_squared_error.*?sqrt|RMSE|np\.sqrt.*?mse',
        'R2': r'r2_score|R2|R²',
        'MAPE': r'MAPE|mean_absolute_percentage',
    }
    
    for metrica, patron in metricas_buscadas.items():
        if re.search(patron, contenido, re.IGNORECASE):
            analisis['metricas_calculadas'].append(metrica)
    
    resultados[nombre] = analisis
    
    # Imprimir resultados
    print(f"\n🎯 Target(s) usado(s): {', '.join(analisis['target']) if analisis['target'] else '❓ No detectado'}")
    print(f"📊 Método de split: {analisis['split_method'] or '❓ No detectado'}")
    print(f"📅 Periodo test: {analisis['periodo_test'] or '❓ No especificado'}")
    print(f"🧮 Cálculo Qout: {analisis['calculo_qout'] or '❌ No encontrado'}")
    print(f"🔒 Data Leakage Check: {analisis['data_leakage_check'] or '❓ No detectado'}")
    print(f"🤖 Modelo(s): {', '.join(analisis['modelo_usado']) if analisis['modelo_usado'] else '❓ No detectado'}")
    print(f"📈 Métricas: {', '.join(analisis['metricas_calculadas']) if analisis['metricas_calculadas'] else 'Ninguna'}")

# ==============================================================================
# 3. COMPARACIÓN Y DETECCIÓN DE INCONSISTENCIAS
# ==============================================================================

print("\n" + "="*80)
print("🔍 DETECCIÓN DE INCONSISTENCIAS")
print("="*80)

# Comparar targets
todos_targets = []
for r in resultados.values():
    todos_targets.extend(r['target'])
targets_unicos = set(todos_targets)

print(f"\n1️⃣ TARGETS DIFERENTES DETECTADOS: {len(targets_unicos)}")
for target in sorted(targets_unicos):
    archivos_con_target = [n for n, r in resultados.items() if target in r['target']]
    print(f"\n   • {target}:")
    for arch in archivos_con_target:
        print(f"      - {arch}")

if len(targets_unicos) > 1:
    print("\n   ⚠️  INCONSISTENCIA: Se usan diferentes targets en diferentes archivos")
    print("   💡 Esto explica por qué las métricas son diferentes")
else:
    print("\n   ✅ Consistente: Todos usan el mismo target")

# Comparar métodos de split
splits_unicos = set(r['split_method'] for r in resultados.values() if r['split_method'])
print(f"\n2️⃣ MÉTODOS DE SPLIT DIFERENTES DETECTADOS: {len(splits_unicos)}")
for split in sorted(splits_unicos):
    archivos_con_split = [n for n, r in resultados.items() if r['split_method'] == split]
    print(f"\n   • {split}:")
    for arch in archivos_con_split:
        print(f"      - {arch}")

if len(splits_unicos) > 1:
    print("\n   ⚠️  INCONSISTENCIA: Se usan diferentes métodos de split")
    print("   💡 Esto causa que evalúen períodos de test DIFERENTES")
else:
    print("\n   ✅ Consistente: Todos usan el mismo método")

# Comparar periodos
periodos_unicos = set(r['periodo_test'] for r in resultados.values() if r['periodo_test'])
print(f"\n3️⃣ PERIODOS DE TEST DIFERENTES DETECTADOS: {len(periodos_unicos)}")
for periodo in sorted(periodos_unicos):
    archivos_con_periodo = [n for n, r in resultados.items() if r['periodo_test'] == periodo]
    print(f"\n   • {periodo}:")
    for arch in archivos_con_periodo:
        print(f"      - {arch}")

# Comparar modelos
todos_modelos = []
for r in resultados.values():
    todos_modelos.extend(r['modelo_usado'])
modelos_unicos = set(todos_modelos)

print(f"\n4️⃣ MODELOS DIFERENTES DETECTADOS: {len(modelos_unicos)}")
for modelo in sorted(modelos_unicos):
    archivos_con_modelo = [n for n, r in resultados.items() if modelo in r['modelo_usado']]
    print(f"\n   • {modelo}:")
    for arch in archivos_con_modelo:
        print(f"      - {arch}")

# Data leakage
archivos_con_leakage_check = [(n, r['data_leakage_check']) for n, r in resultados.items() if r['data_leakage_check']]
print(f"\n5️⃣ PREVENCIÓN DE DATA LEAKAGE:")
if archivos_con_leakage_check:
    for nombre, check in archivos_con_leakage_check:
        print(f"   ✅ {nombre}: {check}")
else:
    print("   ⚠️  No se detectó prevención explícita en ningún archivo")

# ==============================================================================
# 4. GENERAR REPORTE JSON
# ==============================================================================

Path('outputs').mkdir(exist_ok=True)

reporte = {
    'fecha_analisis': pd.Timestamp.now().isoformat(),
    'archivos_analizados': len(resultados),
    'archivos_encontrados': len([r for r in resultados.values() if r]),
    'archivos_no_encontrados': len(archivos_con_metricas) - len(resultados),
    'resultados': {k: {**v, 'target': list(v['target']), 'modelo_usado': list(v['modelo_usado'])} 
                   for k, v in resultados.items()},
    'inconsistencias': {
        'targets_diferentes': len(targets_unicos),
        'targets_lista': sorted(list(targets_unicos)),
        'splits_diferentes': len(splits_unicos),
        'splits_lista': sorted(list(splits_unicos)),
        'modelos_diferentes': len(modelos_unicos),
        'modelos_lista': sorted(list(modelos_unicos)),
        'periodos_diferentes': len(periodos_unicos),
        'periodos_lista': sorted(list(periodos_unicos)),
    },
    'recomendaciones': []
}

# Generar recomendaciones
if len(targets_unicos) > 1:
    reporte['recomendaciones'].append({
        'severidad': 'ALTA',
        'problema': 'Targets diferentes',
        'descripcion': 'Los archivos usan diferentes targets (Q_net vs Qout vs Volumen_Total)',
        'solucion': 'Decidir UN target oficial para la tesis y actualizar todos los scripts'
    })

if len(splits_unicos) > 1:
    reporte['recomendaciones'].append({
        'severidad': 'ALTA',
        'problema': 'Splits diferentes',
        'descripcion': 'Los archivos usan diferentes métodos de split, evaluando períodos DIFERENTES',
        'solucion': 'Estandarizar a 70/15/15 en todos los scripts'
    })

if 'Volumen_Total_m3' in targets_unicos:
    reporte['recomendaciones'].append({
        'severidad': 'CRÍTICA',
        'problema': 'Target inexistente',
        'descripcion': 'modelo_gradio.py busca "Volumen_Total_m3" que NO existe en data_test.csv',
        'solucion': 'Corregir a "sist_Vtotal_m3" o cambiar a "Q_net_m3h"'
    })

with open('outputs/ANALISIS_1_METRICAS.json', 'w', encoding='utf-8') as f:
    json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)

# ==============================================================================
# 5. RESUMEN FINAL
# ==============================================================================

print("\n" + "="*80)
print("✅ ANÁLISIS 1 COMPLETADO")
print("="*80)
print(f"📄 Reporte guardado en: outputs/ANALISIS_1_METRICAS.json")
print(f"\n📊 Resumen:")
print(f"   • Archivos analizados: {len(resultados)}")
print(f"   • Targets diferentes: {len(targets_unicos)} → {sorted(list(targets_unicos))}")
print(f"   • Métodos split diferentes: {len(splits_unicos)} → {sorted(list(splits_unicos))}")
print(f"   • Modelos diferentes: {len(modelos_unicos)} → {sorted(list(modelos_unicos))}")
print(f"   • Periodos diferentes: {len(periodos_unicos)}")

if reporte['recomendaciones']:
    print(f"\n💡 RECOMENDACIONES ({len(reporte['recomendaciones'])}):")
    for i, rec in enumerate(reporte['recomendaciones'], 1):
        print(f"\n   {i}. [{rec['severidad']}] {rec['problema']}")
        print(f"      → {rec['descripcion']}")
        print(f"      ✓ {rec['solucion']}")
else:
    print(f"\n✅ No se detectaron inconsistencias mayores")

print("\n" + "="*80)
print("🎯 CONCLUSIÓN PRINCIPAL:")
print("="*80)
if len(targets_unicos) > 1 or len(splits_unicos) > 1:
    print("⚠️  Las métricas son DIFERENTES porque:")
    print("   1. Se están evaluando TARGETS diferentes")
    print("   2. Se están evaluando PERÍODOS diferentes")
    print("   3. Esto es ESPERADO y NO es un error de los modelos")
    print("\n💡 Solución: Estandarizar target y split en todos los scripts")
else:
    print("✅ El proyecto tiene configuración consistente")
    print("   Si hay diferencias en métricas, revisar:")
    print("   1. Versiones de modelos (reentrenados)")
    print("   2. Orden de datos (shuffle vs ordenado)")
    print("   3. Manejo de valores faltantes")

print("="*80)
