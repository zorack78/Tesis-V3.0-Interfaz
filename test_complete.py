#!/usr/bin/env python3
"""
Script de prueba paso a paso para validar cada componente del sistema.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_step_1_basic_imports():
    """Paso 1: Verificar que se pueden importar todos los módulos."""
    print("=" * 60)
    print("PASO 1: VERIFICANDO IMPORTACIONES")
    print("=" * 60)
    
    try:
        from utils import load_config, calculate_metrics
        from data_processing import DataProcessor
        from feature_engineering import FeatureEngineer
        print("✅ Todos los módulos importados correctamente")
        return True
    except Exception as e:
        print(f"❌ Error al importar: {e}")
        return False

def test_step_2_load_config():
    """Paso 2: Cargar y verificar configuración."""
    print("\n" + "=" * 60)
    print("PASO 2: CARGANDO CONFIGURACIÓN")
    print("=" * 60)
    
    try:
        from utils import load_config
        config = load_config()
        
        print("✅ Configuración cargada")
        print(f"   📁 Directorio de datos: {config['data']['raw_dir']}")
        print(f"   📊 Archivo de volumen: {config['data']['volume_file']}")
        print(f"   📅 Archivo de calendario: {config['data']['calendar_file']}")
        print(f"   🎯 Métricas a usar: {config['validation']['metrics']}")
        
        return config
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return None

def test_step_3_load_data(config):
    """Paso 3: Cargar datos y mostrar información básica."""
    print("\n" + "=" * 60)
    print("PASO 3: CARGANDO Y ANALIZANDO DATOS")
    print("=" * 60)
    
    if not config:
        print("❌ No se puede cargar datos sin configuración")
        return None
    
    try:
        from data_processing import DataProcessor
        processor = DataProcessor(config)
        
        # Cargar datos
        print("📊 Cargando datos de volumen...")
        volume_df = processor.load_volume_data()
        
        print("📅 Cargando datos de calendario...")
        calendar_df = processor.load_calendar_data()
        
        # Análisis básico
        print("\n📈 ANÁLISIS DE DATOS DE VOLUMEN:")
        print(f"   • Registros: {len(volume_df):,}")
        print(f"   • Columnas: {list(volume_df.columns)}")
        print(f"   • Fecha inicial: {volume_df['timestamp'].min()}")
        print(f"   • Fecha final: {volume_df['timestamp'].max()}")
        print(f"   • Volumen promedio: {volume_df[' Volumen_Total_m3'].mean():,.2f} m³")
        print(f"   • Volumen máximo: {volume_df[' Volumen_Total_m3'].max():,.2f} m³")
        print(f"   • Volumen mínimo: {volume_df[' Volumen_Total_m3'].min():,.2f} m³")
        
        print("\n📅 ANÁLISIS DE DATOS DE CALENDARIO:")
        print(f"   • Registros: {len(calendar_df):,}")
        print(f"   • Feriados únicos: {calendar_df['feriado'].sum():,}")
        print(f"   • Días de fin de semana: {calendar_df['es_fin_de_semana'].sum():,}")
        print(f"   • Días con elecciones: {calendar_df['elecciones'].sum():,}")
        print(f"   • Festival de Viña: {calendar_df['festival_vina'].sum():,} días")
        
        return volume_df, calendar_df, processor
        
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return None

def test_step_4_data_processing(data_result):
    """Paso 4: Procesar y limpiar datos."""
    print("\n" + "=" * 60)
    print("PASO 4: PROCESANDO Y LIMPIANDO DATOS")
    print("=" * 60)
    
    if not data_result:
        print("❌ No se pueden procesar datos")
        return None
    
    try:
        volume_df, calendar_df, processor = data_result
        
        print("🔄 Combinando datos...")
        merged_df = processor.merge_volume_calendar(volume_df, calendar_df)
        print(f"   ✅ Datos combinados: {len(merged_df):,} registros")
        
        print("🧹 Limpiando datos...")
        clean_df = processor.clean_data(merged_df)
        print(f"   ✅ Datos limpios: {len(clean_df):,} registros")
        
        print("📊 Dividiendo datos...")
        train_df, val_df, test_df = processor.split_data(clean_df)
        print(f"   📚 Entrenamiento: {len(train_df):,} registros ({len(train_df)/len(clean_df)*100:.1f}%)")
        print(f"   🔍 Validación: {len(val_df):,} registros ({len(val_df)/len(clean_df)*100:.1f}%)")
        print(f"   🧪 Prueba: {len(test_df):,} registros ({len(test_df)/len(clean_df)*100:.1f}%)")
        
        return clean_df, train_df, val_df, test_df
        
    except Exception as e:
        print(f"❌ Error en procesamiento: {e}")
        return None

def test_step_5_feature_engineering(config, processed_data):
    """Paso 5: Crear features."""
    print("\n" + "=" * 60)
    print("PASO 5: CREANDO FEATURES")
    print("=" * 60)
    
    if not processed_data or not config:
        print("❌ No se pueden crear features")
        return None
    
    try:
        from feature_engineering import FeatureEngineer
        clean_df, train_df, val_df, test_df = processed_data
        
        # Tomar una muestra pequeña para probar
        sample_df = clean_df.head(100).copy()
        
        feature_engineer = FeatureEngineer(config)
        
        print("⏰ Creando features temporales...")
        features_df = feature_engineer.create_temporal_features(sample_df)
        print(f"   ✅ Features temporales: {features_df.shape[1]} columnas")
        print(f"   📊 Nuevas columnas: {[col for col in features_df.columns if col not in sample_df.columns][:10]}")
        
        print("\n🔙 Creando features de lag...")
        lag_features_df = feature_engineer.create_lag_features(features_df, lag_hours=[1, 24])
        print(f"   ✅ Features con lags: {lag_features_df.shape[1]} columnas")
        
        return lag_features_df
        
    except Exception as e:
        print(f"❌ Error en feature engineering: {e}")
        return None

def test_step_6_metrics_calculation():
    """Paso 6: Probar cálculo de métricas."""
    print("\n" + "=" * 60)
    print("PASO 6: PROBANDO CÁLCULO DE MÉTRICAS")
    print("=" * 60)
    
    try:
        from utils import calculate_metrics
        
        # Crear datos de prueba
        np.random.seed(42)
        y_true = np.random.normal(100000, 20000, 100)  # Valores típicos de m³
        y_pred = y_true + np.random.normal(0, 5000, 100)  # Predicción con ruido
        
        metrics = calculate_metrics(y_true, y_pred)
        
        print("✅ Métricas calculadas correctamente:")
        print(f"   📊 RMSE: {metrics['rmse']:,.2f} m³")
        print(f"   📊 MAE: {metrics['mae']:,.2f} m³")
        print(f"   📊 MAPE: {metrics['mape']:.2f}%")
        print(f"   📊 R²: {metrics['r2']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en métricas: {e}")
        return False

def main():
    """Ejecuta todas las pruebas paso a paso."""
    print("🚀 INICIANDO PRUEBAS COMPLETAS DEL SISTEMA")
    print("🕐 Esto puede tomar unos minutos...")
    
    # Ejecutar todas las pruebas
    step1_ok = test_step_1_basic_imports()
    if not step1_ok:
        return False
    
    config = test_step_2_load_config()
    
    data_result = test_step_3_load_data(config)
    
    processed_data = test_step_4_data_processing(data_result)
    
    features_result = test_step_5_feature_engineering(config, processed_data)
    
    metrics_ok = test_step_6_metrics_calculation()
    
    # Resumen final
    print("\n" + "=" * 60)
    print("🎯 RESUMEN FINAL DE PRUEBAS")
    print("=" * 60)
    
    tests_results = [
        ("Importaciones", step1_ok),
        ("Configuración", config is not None),
        ("Carga de datos", data_result is not None),
        ("Procesamiento", processed_data is not None),
        ("Feature engineering", features_result is not None),
        ("Cálculo de métricas", metrics_ok)
    ]
    
    passed = sum(1 for _, result in tests_results if result)
    total = len(tests_results)
    
    for test_name, result in tests_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20} {status}")
    
    print(f"\n🏆 RESULTADO: {passed}/{total} pruebas exitosas")
    
    if passed == total:
        print("\n🎉 ¡SISTEMA COMPLETAMENTE FUNCIONAL!")
        print("✅ Puedes proceder a usar los notebooks y entrenar modelos")
    else:
        print("\n⚠️ Algunas pruebas fallaron")
        print("❗ Revisa los errores anteriores antes de continuar")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)