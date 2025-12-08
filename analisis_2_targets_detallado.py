"""
Análisis 2: Verificación detallada de targets por archivo
- Lee código línea por línea
- Identifica EXACTAMENTE qué target usa cada archivo
- Verifica si los datos existen
- NO modifica ningún archivo
"""

import pandas as pd
from pathlib import Path
import json
import re

print("="*80)
print("ANÁLISIS 2: VERIFICACIÓN DETALLADA DE TARGETS")
print("="*80)

# ==============================================================================
# 1. VERIFICAR QUÉ COLUMNAS EXISTEN EN LOS DATOS
# ==============================================================================

print("\n🔍 Paso 1: Verificando columnas disponibles en datasets...")

datasets = {
    'data_test.csv': 'data/processed/data_test.csv',
    'data_train.csv': 'data/processed/data_train.csv',
    'data_validation.csv': 'data/processed/data_validation.csv',
}

columnas_disponibles = {}

for nombre, path in datasets.items():
    if Path(path).exists():
        df = pd.read_csv(path, nrows=1)
        columnas = df.columns.tolist()
        columnas_disponibles[nombre] = columnas
        
        # Buscar columnas relacionadas con targets
        targets_relacionados = [col for col in columnas if any(x in col.lower() for x in 
                               ['qout', 'q_net', 'volumen', 'vtotal', 'demanda'])]
        
        print(f"\n📄 {nombre}:")
        print(f"   Total columnas: {len(columnas)}")
        print(f"   Targets relacionados encontrados:")
        for col in targets_relacionados:
            print(f"      • {col}")
    else:
        print(f"\n⚠️  {nombre}: NO ENCONTRADO")
        columnas_disponibles[nombre] = []

# ==============================================================================
# 2. ANALIZAR CADA ARCHIVO EN DETALLE
# ==============================================================================

print("\n" + "="*80)
print("🔍 Paso 2: Análisis detallado de código por archivo")
print("="*80)

archivos = {
    'modelo_gradio.py': 'modelo_gradio.py',
    'interfaz_planificacion_qin_v1.py': 'interfaz_planificacion_qin_v1.py',
    'graficos_comparativa_test.py': 'graficos_comparativa_test.py',
    'scatter_plots_test.py': 'scatter_plots_test.py',
    'check_r2.py': 'check_r2.py',
}

analisis_detallado = {}

for nombre, archivo in archivos.items():
    print(f"\n{'='*80}")
    print(f"📄 {nombre}")
    print('='*80)
    
    if not Path(archivo).exists():
        print(f"⚠️  Archivo no encontrado")
        continue
    
    with open(archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()
    
    info = {
        'archivo': archivo,
        'lineas_totales': len(lineas),
        'target_asignaciones': [],
        'target_usado': None,
        'columnas_requeridas': [],
        'lectura_datos': [],
        'calculo_qout': [],
        'problemas_potenciales': [],
    }
    
    # Buscar líneas donde se define el target
    for i, linea in enumerate(lineas, 1):
        linea_clean = linea.strip()
        
        # Buscar asignación de target
        if 'target' in linea_clean.lower() and '=' in linea_clean:
            if not linea_clean.startswith('#'):
                info['target_asignaciones'].append({
                    'linea': i,
                    'codigo': linea_clean[:100]
                })
        
        # Buscar y_test o y_train
        if re.search(r'y_(test|train|val)\s*=', linea_clean):
            info['target_usado'] = {
                'linea': i,
                'codigo': linea_clean[:100]
            }
        
        # Buscar lectura de datos
        if 'read_csv' in linea_clean or 'load_csv' in linea_clean:
            info['lectura_datos'].append({
                'linea': i,
                'codigo': linea_clean[:100]
            })
        
        # Buscar cálculo de Qout
        if 'qin' in linea_clean.lower() and '-' in linea_clean and 'qnet' in linea_clean.lower():
            info['calculo_qout'].append({
                'linea': i,
                'codigo': linea_clean[:100]
            })
        
        # Buscar groupby hora (perfil Qin)
        if 'groupby' in linea_clean and 'hora' in linea_clean.lower():
            info['calculo_qout'].append({
                'linea': i,
                'codigo': linea_clean[:100],
                'tipo': 'Perfil horario Qin'
            })
    
    # Identificar target usado
    if info['target_asignaciones']:
        ultima_asignacion = info['target_asignaciones'][-1]['codigo']
        if 'Volumen_Total' in ultima_asignacion:
            info['columnas_requeridas'].append(' Volumen_Total_m3')
            info['problemas_potenciales'].append({
                'severidad': 'CRÍTICA',
                'problema': 'Busca columna " Volumen_Total_m3" que NO existe en data_test.csv',
                'linea': info['target_asignaciones'][-1]['linea']
            })
        elif 'Q_net' in ultima_asignacion:
            info['columnas_requeridas'].append('Q_net_m3h')
        elif 'sist_Vtotal' in ultima_asignacion:
            info['columnas_requeridas'].append('sist_Vtotal_m3')
    
    if info['calculo_qout']:
        info['columnas_requeridas'].extend(['sist_Qin_m3h', 'Q_net_m3h'])
    
    # Verificar si las columnas requeridas existen
    for col_req in info['columnas_requeridas']:
        existe = any(col_req in cols for cols in columnas_disponibles.values())
        if not existe and col_req != ' Volumen_Total_m3':  # Ya sabemos que esta no existe
            info['problemas_potenciales'].append({
                'severidad': 'ALTA',
                'problema': f'Requiere columna "{col_req}" que podría no existir',
                'columna': col_req
            })
    
    analisis_detallado[nombre] = info
    
    # Imprimir resultados
    print(f"\n📊 Target asignaciones encontradas: {len(info['target_asignaciones'])}")
    if info['target_asignaciones']:
        for asig in info['target_asignaciones'][-3:]:  # Últimas 3
            print(f"   Línea {asig['linea']}: {asig['codigo']}")
    
    print(f"\n🎯 Target usado (y_test/y_train):")
    if info['target_usado']:
        print(f"   Línea {info['target_usado']['linea']}: {info['target_usado']['codigo']}")
    else:
        print("   ❓ No detectado")
    
    print(f"\n📦 Columnas requeridas: {info['columnas_requeridas'] or 'Ninguna específica'}")
    
    print(f"\n🧮 Cálculo de Qout: {len(info['calculo_qout'])} ocurrencias")
    if info['calculo_qout']:
        for calc in info['calculo_qout'][:2]:  # Primeras 2
            tipo = calc.get('tipo', 'Cálculo Qout')
            print(f"   Línea {calc['linea']}: {tipo}")
    
    print(f"\n⚠️  Problemas potenciales: {len(info['problemas_potenciales'])}")
    for prob in info['problemas_potenciales']:
        print(f"   [{prob['severidad']}] {prob['problema']}")

# ==============================================================================
# 3. VERIFICAR EXISTENCIA DE COLUMNAS
# ==============================================================================

print("\n" + "="*80)
print("🔍 Paso 3: Verificación de existencia de columnas")
print("="*80)

todas_columnas_requeridas = set()
for info in analisis_detallado.values():
    todas_columnas_requeridas.update(info['columnas_requeridas'])

print(f"\nColumnas requeridas por los scripts: {len(todas_columnas_requeridas)}")

for col in sorted(todas_columnas_requeridas):
    existe_en = []
    for nombre_ds, cols in columnas_disponibles.items():
        if col in cols:
            existe_en.append(nombre_ds)
    
    if existe_en:
        print(f"   ✅ {col}")
        print(f"      Existe en: {', '.join(existe_en)}")
    else:
        print(f"   ❌ {col}")
        print(f"      NO EXISTE en ningún dataset")
        print(f"      ⚠️  Scripts que la requieren:")
        for nombre, info in analisis_detallado.items():
            if col in info['columnas_requeridas']:
                print(f"         - {nombre}")

# ==============================================================================
# 4. TABLA RESUMEN
# ==============================================================================

print("\n" + "="*80)
print("📊 TABLA RESUMEN: Target por Archivo")
print("="*80)

print(f"\n{'Archivo':<40} {'Target Principal':<25} {'¿Calcula Qout?':<15} {'Status'}")
print("-" * 100)

for nombre, info in analisis_detallado.items():
    # Determinar target principal
    if ' Volumen_Total_m3' in info['columnas_requeridas']:
        target = ' Volumen_Total_m3'
        status = '❌ NO EXISTE'
    elif 'Q_net_m3h' in info['columnas_requeridas'] and info['calculo_qout']:
        target = 'Qout (Qin - Q_net)'
        status = '✅ OK'
    elif 'Q_net_m3h' in info['columnas_requeridas']:
        target = 'Q_net_m3h'
        status = '✅ OK'
    elif 'sist_Vtotal_m3' in info['columnas_requeridas']:
        target = 'sist_Vtotal_m3'
        status = '✅ OK'
    else:
        target = '❓ Indefinido'
        status = '⚠️  REVISAR'
    
    calcula_qout = '✅ Sí' if info['calculo_qout'] else '❌ No'
    
    print(f"{nombre:<40} {target:<25} {calcula_qout:<15} {status}")

# ==============================================================================
# 5. GENERAR REPORTE JSON
# ==============================================================================

Path('outputs').mkdir(exist_ok=True)

reporte = {
    'fecha_analisis': pd.Timestamp.now().isoformat(),
    'datasets_verificados': {
        nombre: {
            'existe': len(cols) > 0,
            'total_columnas': len(cols),
            'columnas_target': [col for col in cols if any(x in col.lower() for x in 
                               ['qout', 'q_net', 'volumen', 'vtotal'])]
        }
        for nombre, cols in columnas_disponibles.items()
    },
    'analisis_archivos': analisis_detallado,
    'columnas_requeridas': list(todas_columnas_requeridas),
    'problemas_criticos': [],
}

# Recopilar problemas críticos
for nombre, info in analisis_detallado.items():
    for prob in info['problemas_potenciales']:
        if prob['severidad'] == 'CRÍTICA':
            reporte['problemas_criticos'].append({
                'archivo': nombre,
                **prob
            })

with open('outputs/ANALISIS_2_TARGETS_DETALLADO.json', 'w', encoding='utf-8') as f:
    json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)

# ==============================================================================
# 6. CONCLUSIONES
# ==============================================================================

print("\n" + "="*80)
print("🎯 CONCLUSIONES DEL ANÁLISIS 2")
print("="*80)

problemas_criticos = sum(1 for info in analisis_detallado.values() 
                        for p in info['problemas_potenciales'] if p['severidad'] == 'CRÍTICA')

print(f"\n📊 Estadísticas:")
print(f"   • Archivos analizados: {len(analisis_detallado)}")
print(f"   • Problemas críticos: {problemas_criticos}")
print(f"   • Columnas inexistentes requeridas: {sum(1 for col in todas_columnas_requeridas if not any(col in cols for cols in columnas_disponibles.values()))}")

if problemas_criticos > 0:
    print(f"\n⚠️  PROBLEMAS CRÍTICOS DETECTADOS:")
    for nombre, info in analisis_detallado.items():
        for prob in info['problemas_potenciales']:
            if prob['severidad'] == 'CRÍTICA':
                print(f"\n   📄 {nombre}")
                print(f"      {prob['problema']}")
                if 'linea' in prob:
                    print(f"      Línea: {prob['linea']}")

print(f"\n💡 RECOMENDACIONES:")

if ' Volumen_Total_m3' in todas_columnas_requeridas:
    print(f"\n   1. CORREGIR modelo_gradio.py:")
    print(f"      • Cambiar target de ' Volumen_Total_m3' a 'sist_Vtotal_m3'")
    print(f"      • O cambiar a 'Q_net_m3h' para consistencia con otros scripts")

archivos_sin_qout = [n for n, i in analisis_detallado.items() if not i['calculo_qout']]
if archivos_sin_qout:
    print(f"\n   2. Archivos que NO calculan Qout:")
    for arch in archivos_sin_qout:
        print(f"      • {arch}")
    print(f"      → Si el objetivo es predecir DEMANDA (Qout), añadir cálculo de Qout")

print("\n" + "="*80)
print("✅ ANÁLISIS 2 COMPLETADO")
print("="*80)
print(f"📄 Reporte guardado en: outputs/ANALISIS_2_TARGETS_DETALLADO.json")
print("="*80)
