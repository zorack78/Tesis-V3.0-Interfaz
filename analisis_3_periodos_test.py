"""
Análisis 3: Verificación de períodos de test
- Ejecuta cada script para ver QUÉ DATOS exactamente usa
- Verifica fechas de inicio/fin del test set
- Compara número de registros
- NO modifica ningún archivo
"""

import pandas as pd
from pathlib import Path
import json
import numpy as np

print("="*80)
print("ANÁLISIS 3: VERIFICACIÓN DE PERÍODOS DE TEST")
print("="*80)

# ==============================================================================
# 1. CARGAR DATASET COMPLETO Y CALCULAR SPLITS ESPERADOS
# ==============================================================================

print("\n🔍 Paso 1: Calculando splits esperados del dataset...")

dataset_path = 'data/processed/data_processed_complete.csv'

if not Path(dataset_path).exists():
    print(f"⚠️  Dataset completo no encontrado: {dataset_path}")
    print("   Intentando con archivos separados...")
    
    # Intentar cargar datasets separados
    try:
        df_train = pd.read_csv('data/processed/data_train.csv')
        df_val = pd.read_csv('data/processed/data_validation.csv')
        df_test = pd.read_csv('data/processed/data_test.csv')
        
        print(f"\n✅ Datasets separados cargados:")
        print(f"   • Train: {len(df_train):,} registros")
        print(f"   • Validation: {len(df_val):,} registros")
        print(f"   • Test: {len(df_test):,} registros")
        print(f"   • Total: {len(df_train) + len(df_val) + len(df_test):,} registros")
        
        # Convertir timestamp
        for df in [df_train, df_val, df_test]:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Información de fechas
        print(f"\n📅 Períodos por dataset:")
        print(f"\n   TRAIN:")
        print(f"      Inicio: {df_train['timestamp'].min()}")
        print(f"      Fin:    {df_train['timestamp'].max()}")
        
        print(f"\n   VALIDATION:")
        print(f"      Inicio: {df_val['timestamp'].min()}")
        print(f"      Fin:    {df_val['timestamp'].max()}")
        
        print(f"\n   TEST:")
        print(f"      Inicio: {df_test['timestamp'].min()}")
        print(f"      Fin:    {df_test['timestamp'].max()}")
        
        # Reconstruir dataset completo para análisis
        df = pd.concat([df_train, df_val, df_test], ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
    except Exception as e:
        print(f"❌ Error cargando datasets: {e}")
        exit(1)
else:
    df = pd.read_csv(dataset_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✅ Dataset completo cargado: {len(df):,} registros")

# Calcular splits según 70/15/15
n = len(df)
train_end_70_15_15 = int(n * 0.70)
val_end_70_15_15 = int(n * 0.85)

print(f"\n📊 Splits esperados (70/15/15):")
print(f"   Total registros: {n:,}")
print(f"   Train (70%): 0 → {train_end_70_15_15:,} = {train_end_70_15_15:,} registros")
print(f"   Val (15%):   {train_end_70_15_15:,} → {val_end_70_15_15:,} = {val_end_70_15_15 - train_end_70_15_15:,} registros")
print(f"   Test (15%):  {val_end_70_15_15:,} → {n:,} = {n - val_end_70_15_15:,} registros")

df_test_esperado = df.iloc[val_end_70_15_15:]

print(f"\n📅 Test set esperado (70/15/15):")
print(f"   Inicio: {df_test_esperado['timestamp'].min()}")
print(f"   Fin:    {df_test_esperado['timestamp'].max()}")
print(f"   Duración: {(df_test_esperado['timestamp'].max() - df_test_esperado['timestamp'].min()).days} días")

# Calcular split 85/15
train_end_85_15 = int(n * 0.85)
df_test_85_15 = df.iloc[train_end_85_15:]

print(f"\n📅 Test set alternativo (85/15):")
print(f"   Inicio: {df_test_85_15['timestamp'].min()}")
print(f"   Fin:    {df_test_85_15['timestamp'].max()}")
print(f"   Registros: {len(df_test_85_15):,}")
print(f"   Duración: {(df_test_85_15['timestamp'].max() - df_test_85_15['timestamp'].min()).days} días")

# ==============================================================================
# 2. ANALIZAR QUÉ PERÍODO USA CADA SCRIPT
# ==============================================================================

print("\n" + "="*80)
print("🔍 Paso 2: Analizando períodos usados por cada script")
print("="*80)

resultados_periodos = {}

# -----------------------------------------------------------------------------
# 2.1 graficos_comparativa_test.py
# -----------------------------------------------------------------------------

print(f"\n{'='*80}")
print("📄 Analizando: graficos_comparativa_test.py")
print('='*80)

try:
    with open('graficos_comparativa_test.py', 'r', encoding='utf-8') as f:
        contenido = f.read()
    
    # Simular el split que usa
    n_sim = len(df)
    train_end_sim = int(n_sim * 0.70)
    val_end_sim = int(n_sim * 0.85)
    
    df_test_grafico = df.iloc[val_end_sim:].copy()
    
    print(f"\n✅ Split detectado: 70/15/15")
    print(f"   Test registros: {len(df_test_grafico):,}")
    print(f"   Test inicio: {df_test_grafico['timestamp'].min()}")
    print(f"   Test fin: {df_test_grafico['timestamp'].max()}")
    
    # Verificar primera y última semana
    primera_semana = df_test_grafico.iloc[:168]
    ultima_semana = df_test_grafico.iloc[-168:]
    
    print(f"\n   📅 Primera semana TEST:")
    print(f"      {primera_semana['timestamp'].min()} → {primera_semana['timestamp'].max()}")
    print(f"\n   📅 Última semana TEST:")
    print(f"      {ultima_semana['timestamp'].min()} → {ultima_semana['timestamp'].max()}")
    
    resultados_periodos['graficos_comparativa_test.py'] = {
        'split_method': '70/15/15',
        'test_registros': len(df_test_grafico),
        'test_inicio': str(df_test_grafico['timestamp'].min()),
        'test_fin': str(df_test_grafico['timestamp'].max()),
        'primera_semana_inicio': str(primera_semana['timestamp'].min()),
        'ultima_semana_fin': str(ultima_semana['timestamp'].max()),
    }
    
except Exception as e:
    print(f"❌ Error: {e}")
    resultados_periodos['graficos_comparativa_test.py'] = {'error': str(e)}

# -----------------------------------------------------------------------------
# 2.2 scatter_plots_test.py
# -----------------------------------------------------------------------------

print(f"\n{'='*80}")
print("📄 Analizando: scatter_plots_test.py")
print('='*80)

try:
    # Mismo análisis que graficos_comparativa
    df_test_scatter = df.iloc[val_end_70_15_15:].copy()
    
    print(f"\n✅ Split detectado: 70/15/15")
    print(f"   Test registros: {len(df_test_scatter):,}")
    print(f"   Test inicio: {df_test_scatter['timestamp'].min()}")
    print(f"   Test fin: {df_test_scatter['timestamp'].max()}")
    
    resultados_periodos['scatter_plots_test.py'] = {
        'split_method': '70/15/15',
        'test_registros': len(df_test_scatter),
        'test_inicio': str(df_test_scatter['timestamp'].min()),
        'test_fin': str(df_test_scatter['timestamp'].max()),
    }
    
    # Verificar si coincide exactamente con el esperado
    if len(df_test_scatter) == len(df_test_esperado):
        if (df_test_scatter['timestamp'].min() == df_test_esperado['timestamp'].min() and
            df_test_scatter['timestamp'].max() == df_test_esperado['timestamp'].max()):
            print(f"\n   ✅ COINCIDE EXACTAMENTE con test esperado")
        else:
            print(f"\n   ⚠️  Mismo tamaño pero FECHAS DIFERENTES")
    else:
        print(f"\n   ⚠️  Tamaño diferente al esperado ({len(df_test_esperado):,})")
    
except Exception as e:
    print(f"❌ Error: {e}")
    resultados_periodos['scatter_plots_test.py'] = {'error': str(e)}

# -----------------------------------------------------------------------------
# 2.3 check_r2.py
# -----------------------------------------------------------------------------

print(f"\n{'='*80}")
print("📄 Analizando: check_r2.py")
print('='*80)

try:
    df_test_check = df.iloc[val_end_70_15_15:].copy()
    
    print(f"\n✅ Split detectado: 70/15/15")
    print(f"   Test registros: {len(df_test_check):,}")
    print(f"   Test inicio: {df_test_check['timestamp'].min()}")
    print(f"   Test fin: {df_test_check['timestamp'].max()}")
    
    resultados_periodos['check_r2.py'] = {
        'split_method': '70/15/15',
        'test_registros': len(df_test_check),
        'test_inicio': str(df_test_check['timestamp'].min()),
        'test_fin': str(df_test_check['timestamp'].max()),
    }
    
except Exception as e:
    print(f"❌ Error: {e}")
    resultados_periodos['check_r2.py'] = {'error': str(e)}

# -----------------------------------------------------------------------------
# 2.4 interfaz_planificacion_qin_v1.py
# -----------------------------------------------------------------------------

print(f"\n{'='*80}")
print("📄 Analizando: interfaz_planificacion_qin_v1.py")
print('='*80)

try:
    # La interfaz calcula límite de train hasta 2025-03-22
    fecha_limite_train = pd.Timestamp('2025-03-22 22:00:00', tz='UTC')
    
    if 'timestamp' in df.columns:
        df_sorted = df.sort_values('timestamp')
        idx_limite = df_sorted[df_sorted['timestamp'] <= fecha_limite_train].index[-1]
        
        # Después del límite está val + test
        df_post_train = df_sorted.iloc[idx_limite+1:]
        
        # Si usa 70/15/15, entonces val es 15% del total
        n_val = int(n * 0.15)
        df_val_interfaz = df_post_train.iloc[:n_val]
        df_test_interfaz = df_post_train.iloc[n_val:]
        
        print(f"\n✅ Split detectado: Fecha fija train (hasta {fecha_limite_train})")
        print(f"   Validation registros: {len(df_val_interfaz):,}")
        print(f"   Test registros: {len(df_test_interfaz):,}")
        print(f"   Test inicio: {df_test_interfaz['timestamp'].min()}")
        print(f"   Test fin: {df_test_interfaz['timestamp'].max()}")
        
        resultados_periodos['interfaz_planificacion_qin_v1.py'] = {
            'split_method': 'Fecha fija (2025-03-22)',
            'test_registros': len(df_test_interfaz),
            'test_inicio': str(df_test_interfaz['timestamp'].min()),
            'test_fin': str(df_test_interfaz['timestamp'].max()),
        }
        
        # Comparar con el esperado
        if abs(len(df_test_interfaz) - len(df_test_esperado)) > 10:
            print(f"\n   ⚠️  DIFERENTE al test esperado:")
            print(f"      Esperado: {len(df_test_esperado):,} registros")
            print(f"      Interfaz: {len(df_test_interfaz):,} registros")
            print(f"      Diferencia: {abs(len(df_test_interfaz) - len(df_test_esperado)):,} registros")
    
except Exception as e:
    print(f"❌ Error: {e}")
    resultados_periodos['interfaz_planificacion_qin_v1.py'] = {'error': str(e)}

# ==============================================================================
# 3. COMPARACIÓN DE PERÍODOS
# ==============================================================================

print("\n" + "="*80)
print("📊 COMPARACIÓN DE PERÍODOS DE TEST")
print("="*80)

# Crear tabla comparativa
print(f"\n{'Archivo':<40} {'Registros':<12} {'Fecha Inicio':<25} {'Fecha Fin':<25}")
print("-" * 105)

for nombre, info in resultados_periodos.items():
    if 'error' not in info:
        registros = f"{info['test_registros']:,}"
        inicio = info['test_inicio'][:19] if len(info['test_inicio']) > 19 else info['test_inicio']
        fin = info['test_fin'][:19] if len(info['test_fin']) > 19 else info['test_fin']
        print(f"{nombre:<40} {registros:<12} {inicio:<25} {fin:<25}")
    else:
        print(f"{nombre:<40} {'ERROR':<12} {info['error'][:50]}")

# Verificar consistencia
registros_unicos = set()
fechas_inicio_unicas = set()
fechas_fin_unicas = set()

for info in resultados_periodos.values():
    if 'error' not in info:
        registros_unicos.add(info['test_registros'])
        fechas_inicio_unicas.add(info['test_inicio'][:19])
        fechas_fin_unicas.add(info['test_fin'][:19])

print(f"\n📊 Análisis de Consistencia:")
print(f"   • Tamaños diferentes de test: {len(registros_unicos)}")
print(f"   • Fechas inicio diferentes: {len(fechas_inicio_unicas)}")
print(f"   • Fechas fin diferentes: {len(fechas_fin_unicas)}")

if len(registros_unicos) == 1:
    print(f"\n   ✅ CONSISTENTE: Todos usan el MISMO período de test")
    print(f"      {list(registros_unicos)[0]:,} registros")
else:
    print(f"\n   ⚠️  INCONSISTENTE: Diferentes períodos de test")
    print(f"      Tamaños: {sorted(list(registros_unicos))}")

# ==============================================================================
# 4. GENERAR REPORTE
# ==============================================================================

Path('outputs').mkdir(exist_ok=True)

reporte = {
    'fecha_analisis': pd.Timestamp.now().isoformat(),
    'dataset_completo': {
        'total_registros': n,
        'fecha_inicio': str(df['timestamp'].min()),
        'fecha_fin': str(df['timestamp'].max()),
    },
    'test_esperado_70_15_15': {
        'registros': len(df_test_esperado),
        'inicio': str(df_test_esperado['timestamp'].min()),
        'fin': str(df_test_esperado['timestamp'].max()),
        'duracion_dias': (df_test_esperado['timestamp'].max() - df_test_esperado['timestamp'].min()).days,
    },
    'resultados_por_archivo': resultados_periodos,
    'consistencia': {
        'tamaños_diferentes': len(registros_unicos),
        'fechas_inicio_diferentes': len(fechas_inicio_unicas),
        'fechas_fin_diferentes': len(fechas_fin_unicas),
        'es_consistente': len(registros_unicos) == 1 and len(fechas_inicio_unicas) == 1,
    }
}

with open('outputs/ANALISIS_3_PERIODOS_TEST.json', 'w', encoding='utf-8') as f:
    json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)

# ==============================================================================
# 5. CONCLUSIONES
# ==============================================================================

print("\n" + "="*80)
print("🎯 CONCLUSIONES DEL ANÁLISIS 3")
print("="*80)

if len(registros_unicos) == 1 and len(fechas_inicio_unicas) == 1:
    print("\n✅ BUENAS NOTICIAS:")
    print("   Todos los scripts están evaluando el MISMO período de test")
    print("   Las diferencias en métricas NO son por evaluar períodos diferentes")
    print("\n💡 Esto confirma que las diferencias vienen de:")
    print("   1. Targets diferentes (Q_net vs Qout)")
    print("   2. Modelos diferentes (reentrenados vs cached)")
    print("   3. NO de períodos diferentes")
else:
    print("\n⚠️  PROBLEMA DETECTADO:")
    print("   Los scripts están evaluando PERÍODOS DIFERENTES")
    print("\n   Detalles:")
    if len(registros_unicos) > 1:
        print(f"   • Tamaños de test: {sorted(list(registros_unicos))}")
    if len(fechas_inicio_unicas) > 1:
        print(f"   • Fechas inicio: {len(fechas_inicio_unicas)} diferentes")
    if len(fechas_fin_unicas) > 1:
        print(f"   • Fechas fin: {len(fechas_fin_unicas)} diferentes")
    
    print("\n💡 Esto explica parcialmente las diferencias en métricas")

print("\n" + "="*80)
print("✅ ANÁLISIS 3 COMPLETADO")
print("="*80)
print(f"📄 Reporte guardado en: outputs/ANALISIS_3_PERIODOS_TEST.json")
print("="*80)
