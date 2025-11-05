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


def main():
    """Ejecuta el pipeline completo de procesamiento."""
    print("=" * 60)
    print("PIPELINE DE PROCESAMIENTO DE DATOS")
    print("=" * 60)
    
    # Cargar configuración
    config = load_config()
    setup_logging(config)
    
    # Inicializar procesador
    processor = DataProcessor(config)
    
    print("\n1. Cargando datos...")
    volume_df = processor.load_volume_data()
    calendar_df = processor.load_calendar_data()
    
    print("\n2. Combinando datos...")
    merged_df = processor.merge_volume_calendar(volume_df, calendar_df)
    
    print("\n3. Limpiando datos...")
    clean_df = processor.clean_data(merged_df)
    
    print("\n4. Dividiendo datos...")
    train_df, val_df, test_df = processor.split_data(clean_df)
    
    print("\n5. Guardando datos procesados...")
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