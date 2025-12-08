"""
Análisis 4: Verificación de modelos utilizados
- Verifica qué modelos carga cada script
- Compara versiones, timestamps de archivos
- Verifica hiperparámetros si están disponibles
- NO modifica ningún archivo
"""

import pandas as pd
from pathlib import Path
import json
import joblib
import os
from datetime import datetime

print("="*80)
print("ANÁLISIS 4: VERIFICACIÓN DE MODELOS UTILIZADOS")
print("="*80)

# ==============================================================================
# 1. INVENTARIO DE MODELOS DISPONIBLES
# ==============================================================================

print("\n🔍 Paso 1: Inventario de modelos disponibles...")

modelos_paths = {
    'LightGBM': 'models/forecasting/modelo_forecasting_lgbm.pkl',
    'XGBoost': 'models/forecasting/modelo_forecasting_xgboost.pkl',
    'RandomForest': 'models/forecasting/modelo_forecasting_randomforest.pkl',
}

modelos_info = {}

for nombre, path in modelos_paths.items():
    print(f"\n📦 {nombre}:")
    
    if not Path(path).exists():
        print(f"   ❌ NO ENCONTRADO: {path}")
        modelos_info[nombre] = {'existe': False, 'path': path}
        continue
    
    info = {
        'existe': True,
        'path': path,
        'size_mb': os.path.getsize(path) / (1024 * 1024),
        'fecha_modificacion': datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
        'edad_horas': (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).total_seconds() / 3600,
    }
    
    # Intentar cargar y extraer información
    try:
        modelo = joblib.load(path)
        
        # Información del modelo
        info['tipo'] = type(modelo).__name__
        
        # Features
        if hasattr(modelo, 'feature_name_'):
            info['num_features'] = len(modelo.feature_name_)
            info['features'] = modelo.feature_name_[:10]  # Primeras 10
        elif hasattr(modelo, 'n_features_in_'):
            info['num_features'] = modelo.n_features_in_
        
        # Hiperparámetros comunes
        if hasattr(modelo, 'get_params'):
            params = modelo.get_params()
            info['hiperparametros_clave'] = {
                k: v for k, v in params.items() 
                if k in ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves', 
                        'min_child_samples', 'subsample', 'colsample_bytree']
                and v is not None
            }
        
        # Para LightGBM específicamente
        if 'LGBMRegressor' in str(type(modelo)):
            info['lgbm_params'] = {
                'num_iterations': modelo.n_estimators,
                'num_leaves': modelo.num_leaves if hasattr(modelo, 'num_leaves') else None,
                'learning_rate': modelo.learning_rate if hasattr(modelo, 'learning_rate') else None,
            }
        
        # Para XGBoost
        if 'XGBRegressor' in str(type(modelo)):
            info['xgb_params'] = {
                'n_estimators': modelo.n_estimators,
                'max_depth': modelo.max_depth,
                'learning_rate': modelo.learning_rate if hasattr(modelo, 'learning_rate') else None,
            }
        
        # Para RandomForest
        if 'RandomForest' in str(type(modelo)):
            info['rf_params'] = {
                'n_estimators': modelo.n_estimators,
                'max_depth': modelo.max_depth,
            }
        
        info['carga_exitosa'] = True
        
    except Exception as e:
        info['carga_exitosa'] = False
        info['error_carga'] = str(e)[:200]
    
    modelos_info[nombre] = info
    
    # Imprimir resumen
    print(f"   ✅ Encontrado: {path}")
    print(f"   📊 Tamaño: {info['size_mb']:.2f} MB")
    print(f"   📅 Modificado: {info['fecha_modificacion']}")
    print(f"   ⏰ Edad: {info['edad_horas']:.1f} horas")
    
    if info['carga_exitosa']:
        print(f"   🤖 Tipo: {info['tipo']}")
        if 'num_features' in info:
            print(f"   📈 Features: {info['num_features']}")
        if 'hiperparametros_clave' in info:
            print(f"   ⚙️  Hiperparámetros clave:")
            for k, v in info['hiperparametros_clave'].items():
                print(f"      • {k}: {v}")
    else:
        print(f"   ❌ Error al cargar: {info.get('error_carga', 'Desconocido')}")

# ==============================================================================
# 2. ANALIZAR QUÉ MODELOS USA CADA SCRIPT
# ==============================================================================

print("\n" + "="*80)
print("🔍 Paso 2: Analizando modelos usados por cada script")
print("="*80)

archivos = {
    'graficos_comparativa_test.py': 'graficos_comparativa_test.py',
    'scatter_plots_test.py': 'scatter_plots_test.py',
    'check_r2.py': 'check_r2.py',
    'interfaz_planificacion_qin_v1.py': 'interfaz_planificacion_qin_v1.py',
}

uso_modelos = {}

for nombre, archivo in archivos.items():
    print(f"\n{'='*80}")
    print(f"📄 {nombre}")
    print('='*80)
    
    if not Path(archivo).exists():
        print(f"⚠️  Archivo no encontrado")
        continue
    
    with open(archivo, 'r', encoding='utf-8') as f:
        contenido = f.read()
    
    info = {
        'archivo': archivo,
        'modelos_cargados': [],
        'modelos_entrenados': [],
        'paths_modelos': [],
    }
    
    # Buscar cargas de modelos
    import re
    
    # Patrón para joblib.load
    loads = re.findall(r'joblib\.load\([\'"]([^\'"]+)[\'"]\)', contenido)
    for path_modelo in loads:
        modelo_nombre = Path(path_modelo).stem
        info['modelos_cargados'].append({
            'tipo': 'Cargado desde pickle',
            'nombre': modelo_nombre,
            'path': path_modelo
        })
        info['paths_modelos'].append(path_modelo)
    
    # Buscar entrenamiento de modelos
    if 'RandomForestRegressor' in contenido and '.fit(' in contenido:
        info['modelos_entrenados'].append('RandomForest')
    
    if 'LGBMRegressor' in contenido and '.fit(' in contenido:
        info['modelos_entrenados'].append('LightGBM')
    
    if 'XGBRegressor' in contenido and '.fit(' in contenido:
        info['modelos_entrenados'].append('XGBoost')
    
    # Buscar referencias a modelos específicos
    if 'lgbm' in contenido.lower():
        if 'LightGBM' not in info['modelos_entrenados'] and \
           not any('lgbm' in m['path'] for m in info['modelos_cargados']):
            info['menciones'] = info.get('menciones', []) + ['LightGBM']
    
    if 'xgboost' in contenido.lower() or 'xgb' in contenido.lower():
        if 'XGBoost' not in info['modelos_entrenados'] and \
           not any('xgb' in m['path'] for m in info['modelos_cargados']):
            info['menciones'] = info.get('menciones', []) + ['XGBoost']
    
    uso_modelos[nombre] = info
    
    # Imprimir resultados
    print(f"\n📦 Modelos cargados: {len(info['modelos_cargados'])}")
    for modelo in info['modelos_cargados']:
        print(f"   • {modelo['nombre']} ({modelo['path']})")
    
    print(f"\n🏋️  Modelos entrenados en memoria: {len(info['modelos_entrenados'])}")
    if info['modelos_entrenados']:
        for modelo in info['modelos_entrenados']:
            print(f"   • {modelo}")
    
    if 'menciones' in info:
        print(f"\n💬 Menciones adicionales: {', '.join(info['menciones'])}")

# ==============================================================================
# 3. VERIFICAR CONSISTENCIA DE MODELOS
# ==============================================================================

print("\n" + "="*80)
print("🔍 Paso 3: Verificación de consistencia de modelos")
print("="*80)

# Verificar si todos usan los mismos modelos guardados
todos_paths = []
for info in uso_modelos.values():
    todos_paths.extend(info['paths_modelos'])

paths_unicos = list(set(todos_paths))

print(f"\nPaths de modelos únicos encontrados: {len(paths_unicos)}")
for path in paths_unicos:
    archivos_que_usan = [nombre for nombre, info in uso_modelos.items() 
                         if path in info['paths_modelos']]
    existe = '✅' if Path(path).exists() else '❌'
    print(f"\n   {existe} {path}")
    print(f"      Usado por:")
    for arch in archivos_que_usan:
        print(f"         - {arch}")

# Verificar si hay entrenamiento en memoria
archivos_con_entrenamiento = {nombre: info['modelos_entrenados'] 
                               for nombre, info in uso_modelos.items() 
                               if info['modelos_entrenados']}

if archivos_con_entrenamiento:
    print(f"\n⚠️  Archivos que entrenan modelos en memoria:")
    for nombre, modelos in archivos_con_entrenamiento.items():
        print(f"\n   📄 {nombre}")
        print(f"      Entrena: {', '.join(modelos)}")
        print(f"      ⚠️  Las métricas pueden variar por:")
        print(f"         • Aleatoriedad en entrenamiento")
        print(f"         • Diferentes features seleccionadas")
        print(f"         • Diferentes hiperparámetros")

# ==============================================================================
# 4. TABLA RESUMEN
# ==============================================================================

print("\n" + "="*80)
print("📊 TABLA RESUMEN: Modelos por Archivo")
print("="*80)

print(f"\n{'Archivo':<40} {'LightGBM':<15} {'XGBoost':<15} {'RandomForest':<15}")
print("-" * 90)

for nombre, info in uso_modelos.items():
    lgbm_status = '❓'
    xgb_status = '❓'
    rf_status = '❓'
    
    # Verificar cada modelo
    for modelo in info['modelos_cargados']:
        if 'lgbm' in modelo['path'].lower():
            lgbm_status = '💾 Cargado'
        elif 'xgb' in modelo['path'].lower():
            xgb_status = '💾 Cargado'
    
    if 'LightGBM' in info['modelos_entrenados']:
        lgbm_status = '🏋️  Entrenado'
    if 'XGBoost' in info['modelos_entrenados']:
        xgb_status = '🏋️  Entrenado'
    if 'RandomForest' in info['modelos_entrenados']:
        rf_status = '🏋️  Entrenado'
    
    print(f"{nombre:<40} {lgbm_status:<15} {xgb_status:<15} {rf_status:<15}")

# ==============================================================================
# 5. GENERAR REPORTE JSON
# ==============================================================================

Path('outputs').mkdir(exist_ok=True)

reporte = {
    'fecha_analisis': pd.Timestamp.now().isoformat(),
    'modelos_disponibles': modelos_info,
    'uso_por_archivo': uso_modelos,
    'paths_unicos': paths_unicos,
    'archivos_con_entrenamiento': archivos_con_entrenamiento,
    'conclusiones': {
        'todos_usan_mismos_pickles': len(set(tuple(sorted(info['paths_modelos'])) 
                                                    for info in uso_modelos.values())) == 1,
        'hay_entrenamiento_en_memoria': len(archivos_con_entrenamiento) > 0,
        'modelos_pickles_actualizados_recientemente': any(
            info['existe'] and info['edad_horas'] < 24 
            for info in modelos_info.values()
        ),
    }
}

with open('outputs/ANALISIS_4_MODELOS.json', 'w', encoding='utf-8') as f:
    json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)

# ==============================================================================
# 6. CONCLUSIONES FINALES
# ==============================================================================

print("\n" + "="*80)
print("🎯 CONCLUSIONES DEL ANÁLISIS 4")
print("="*80)

print(f"\n📊 Estadísticas:")
print(f"   • Modelos pickle disponibles: {sum(1 for m in modelos_info.values() if m['existe'])}/3")
print(f"   • Archivos que entrenan modelos: {len(archivos_con_entrenamiento)}")
print(f"   • Paths únicos de modelos: {len(paths_unicos)}")

# Verificar si LightGBM fue reentrenado recientemente
lgbm_info = modelos_info.get('LightGBM', {})
if lgbm_info.get('existe') and lgbm_info.get('edad_horas', 999) < 24:
    print(f"\n⚠️  IMPORTANTE:")
    print(f"   LightGBM fue reentrenado hace {lgbm_info['edad_horas']:.1f} horas")
    print(f"   Esto explica por qué las métricas son diferentes a runs anteriores")

if archivos_con_entrenamiento:
    print(f"\n⚠️  CAUSA DE VARIACIÓN EN MÉTRICAS:")
    print(f"   Los siguientes archivos ENTRENAN modelos en cada ejecución:")
    for nombre in archivos_con_entrenamiento.keys():
        print(f"      • {nombre}")
    print(f"\n   💡 Esto significa:")
    print(f"      • Cada ejecución puede dar métricas LIGERAMENTE diferentes")
    print(f"      • Debido a aleatoriedad en entrenamiento (seed)")
    print(f"      • Especialmente RandomForest (bootstrap sampling)")

if not reporte['conclusiones']['todos_usan_mismos_pickles']:
    print(f"\n⚠️  INCONSISTENCIA DETECTADA:")
    print(f"   Los scripts NO usan los mismos modelos pickle")
    print(f"   Esto puede causar diferencias en métricas")

print("\n💡 RECOMENDACIONES:")
print(f"\n   1. Para métricas CONSISTENTES:")
print(f"      • Usar modelos PICKLE guardados (no entrenar en cada run)")
print(f"      • Fijar random_state en todos los modelos")
print(f"      • Usar los mismos paths de modelos")

print(f"\n   2. Para reportar en tesis:")
print(f"      • Usar UN script oficial para calcular métricas")
print(f"      • Documentar versión de modelo (timestamp)")
print(f"      • Ejecutar múltiples veces y reportar promedio ± std")

print("\n" + "="*80)
print("✅ ANÁLISIS 4 COMPLETADO")
print("="*80)
print(f"📄 Reporte guardado en: outputs/ANALISIS_4_MODELOS.json")
print("="*80)
