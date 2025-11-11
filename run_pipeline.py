#!/usr/bin/env python3
"""
Script para ejecutar el pipeline completo de procesamiento de datos.
"""

import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import load_config, setup_logging
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer


def main():
    """Ejecuta el pipeline completo de procesamiento."""
    print("=" * 60)
    print("PIPELINE DE PROCESAMIENTO DE DATOS V3.0")
    print("=" * 60)
    
    # Cargar configuración
    config = load_config()
    setup_logging(config)
    
    # Inicializar procesador y feature engineer
    processor = DataProcessor(config)
    feature_engineer = FeatureEngineer(config)
    
    print("\n1. Cargando datos...")
    volume_df = processor.load_volume_data()
    calendar_df = processor.load_calendar_data()
    
    print("\n2. Combinando datos...")
    merged_df = processor.merge_volume_calendar(volume_df, calendar_df)
    
    print("\n3. Limpiando datos...")
    clean_df = processor.clean_data(merged_df)
    
    print("\n4. Creando features avanzadas...")
    print("   → Features temporales...")
    clean_df = feature_engineer.create_temporal_features(clean_df)
    
    print("   → Features LAG (1h, 24h, 168h)...")
    clean_df = feature_engineer.create_lag_features(
        clean_df, 
        target_col=' Volumen_Total_m3',
        lag_hours=[1, 2, 3, 24, 48, 168]
    )
    
    print("   → Features ROLLING (6h, 24h, 168h)...")
    clean_df = feature_engineer.create_rolling_features(
        clean_df,
        target_col=' Volumen_Total_m3',
        windows=[6, 24, 168]
    )
    
    print("   → Features de DIFERENCIA...")
    clean_df = feature_engineer.create_difference_features(
        clean_df,
        target_col=' Volumen_Total_m3'
    )
    
    print("   → Interacciones de calendario...")
    clean_df = feature_engineer.create_calendar_interactions(clean_df)
    
    # Manejar NaN generados por LAG/ROLLING de manera inteligente
    n_before = len(clean_df)
    
    # Identificar columnas con features avanzadas
    lag_cols = [col for col in clean_df.columns if 'lag_' in col or 'rolling_' in col or 'diff_' in col or 'pct_change' in col]
    
    # Rellenar NaN en features LAG/ROLLING con forward fill y luego con 0
    for col in lag_cols:
        clean_df[col] = clean_df[col].fillna(method='ffill').fillna(0)
    
    # Solo eliminar filas donde el TARGET sea NaN
    clean_df = clean_df.dropna(subset=[' Volumen_Total_m3'])
    
    n_after = len(clean_df)
    print(f"   ✅ Features creadas: {clean_df.shape[1]} columnas")
    print(f"   ℹ️  Filas procesadas: {n_after}/{n_before} (conservadas {100*n_after/n_before:.1f}%)")
    
    print("\n5. Dividiendo datos...")
    train_df, val_df, test_df = processor.split_data(clean_df)
    
    print("\n6. Guardando datos procesados...")
    processor.save_processed_data(clean_df, 'data_processed_complete.csv')
    processor.save_processed_data(train_df, 'data_train.csv')
    processor.save_processed_data(val_df, 'data_validation.csv')
    processor.save_processed_data(test_df, 'data_test.csv')
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print(f"Total de registros procesados: {len(clean_df)}")
    print(f"Entrenamiento: {len(train_df)} registros")
    print(f"Validación: {len(val_df)} registros")
    print(f"Prueba: {len(test_df)} registros")
    print("\nArchivos guardados en: data/processed/")


if __name__ == "__main__":
    main()