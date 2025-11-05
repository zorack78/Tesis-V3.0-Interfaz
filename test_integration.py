#!/usr/bin/env python3
"""
Script de prueba para validar la integración de todos los módulos.
"""

import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from utils import load_config
    from data_processing import DataProcessor
    from feature_engineering import FeatureEngineer
    print("✓ Todos los módulos se importaron correctamente")
except ImportError as e:
    print(f"✗ Error al importar módulos: {e}")
    sys.exit(1)


def test_config_loading():
    """Prueba la carga de configuración."""
    try:
        config = load_config("config/config.yaml")
        print("✓ Configuración cargada correctamente")
        print(f"  - Directorio de datos raw: {config['data']['raw_dir']}")
        print(f"  - Archivo de volumen: {config['data']['volume_file']}")
        print(f"  - Archivo de calendario: {config['data']['calendar_file']}")
        return config
    except Exception as e:
        print(f"✗ Error al cargar configuración: {e}")
        return None

def test_data_files_exist(config):
    """Verifica que los archivos de datos existan."""
    if not config:
        return False
    
    try:
        raw_dir = Path(config['data']['raw_dir'])
        volume_file = raw_dir / config['data']['volume_file']
        calendar_file = raw_dir / config['data']['calendar_file']
        
        if volume_file.exists():
            print(f"✓ Archivo de volumen encontrado: {volume_file}")
        else:
            print(f"✗ Archivo de volumen no encontrado: {volume_file}")
            return False
            
        if calendar_file.exists():
            print(f"✓ Archivo de calendario encontrado: {calendar_file}")
        else:
            print(f"✗ Archivo de calendario no encontrado: {calendar_file}")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Error al verificar archivos: {e}")
        return False

def test_data_processor(config):
    """Prueba el procesador de datos."""
    if not config:
        return None
    
    try:
        processor = DataProcessor(config)
        print("✓ DataProcessor inicializado correctamente")
        
        # Intentar cargar una muestra pequeña de datos
        volume_df = processor.load_volume_data()
        print(f"✓ Datos de volumen cargados: {len(volume_df)} registros")
        print(f"  - Columnas: {list(volume_df.columns)}")
        date_range = (f"{volume_df['timestamp'].min()} a "
                      f"{volume_df['timestamp'].max()}")
        print(f"  - Rango de fechas: {date_range}")
        
        calendar_df = processor.load_calendar_data()
        print(f"✓ Datos de calendario cargados: {len(calendar_df)} registros")
        print(f"  - Columnas: {list(calendar_df.columns)}")
        
        return volume_df, calendar_df
    except Exception as e:
        print(f"✗ Error en DataProcessor: {e}")
        return None

def test_feature_engineer(config):
    """Prueba el ingeniero de features."""
    if not config:
        return False
    
    try:
        FeatureEngineer(config)
        print("✓ FeatureEngineer inicializado correctamente")
        return True
    except Exception as e:
        print(f"✗ Error en FeatureEngineer: {e}")
        return False


def main():
    """Función principal de pruebas."""
    print("=" * 60)
    print("PRUEBAS DE INTEGRACIÓN - MODELO PREDICTIVO AGUA POTABLE")
    print("=" * 60)
    
    # Test 1: Configuración
    print("\n1. PRUEBA DE CONFIGURACIÓN")
    print("-" * 30)
    config = test_config_loading()
    
    # Test 2: Archivos de datos
    print("\n2. PRUEBA DE ARCHIVOS DE DATOS")
    print("-" * 30)
    files_ok = test_data_files_exist(config)
    
    # Test 3: Procesador de datos
    print("\n3. PRUEBA DE PROCESADOR DE DATOS")
    print("-" * 30)
    data_result = test_data_processor(config) if files_ok else None
    
    # Test 4: Ingeniero de features
    print("\n4. PRUEBA DE INGENIERO DE FEATURES")
    print("-" * 30)
    feature_ok = test_feature_engineer(config)
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    tests = [
        ("Configuración", config is not None),
        ("Archivos de datos", files_ok),
        ("Procesador de datos", data_result is not None),
        ("Ingeniero de features", feature_ok)
    ]
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    for test_name, success in tests:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nResultado: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! El proyecto está listo.")
    else:
        print("⚠️  Algunas pruebas fallaron. Revisa los errores anteriores.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)