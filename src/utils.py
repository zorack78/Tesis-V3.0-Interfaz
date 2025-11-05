"""
Utilidades generales para el proyecto de predicción de demanda de agua.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Carga el archivo de configuración YAML.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Diccionario con la configuración
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any] = None):
    """
    Configura el sistema de logging.
    
    Args:
        config: Diccionario de configuración
    """
    if config is None:
        config = load_config()
    
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get(
        'format',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format
    )


def create_directories(config: Dict[str, Any]):
    """
    Crea los directorios necesarios para el proyecto.
    
    Args:
        config: Diccionario de configuración
    """
    directories = [
        config['data']['raw_dir'],
        config['data']['processed_dir'],
        config['data']['features_dir'],
        config['models']['output_dir'],
        'logs',
        'outputs/figures'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def convert_utc_to_local(df: pd.DataFrame,
                         timestamp_col: str = 'timestamp',
                         local_tz: str = 'America/Santiago') -> pd.DataFrame:
    """
    Convierte timestamps UTC a hora local chilena.
    
    Args:
        df: DataFrame con columna de timestamp
        timestamp_col: Nombre de la columna con timestamps
        local_tz: Timezone local
        
    Returns:
        DataFrame con columna adicional de hora local
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['timestamp_local'] = (df[timestamp_col]
                             .dt.tz_localize('UTC')
                             .dt.tz_convert(local_tz))
    return df


def calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de evaluación.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        Diccionario con métricas
    """
    from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                                 r2_score)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


def save_model(model, model_name: str, output_dir: str = "models/trained"):
    """
    Guarda un modelo entrenado.
    
    Args:
        model: Modelo a guardar
        model_name: Nombre del modelo
        output_dir: Directorio de salida
    """
    import joblib
    from datetime import datetime
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = Path(output_dir) / filename
    
    joblib.dump(model, filepath)
    logging.info(f"Modelo guardado en: {filepath}")
    
    return filepath


def load_model(model_path: str):
    """
    Carga un modelo guardado.
    
    Args:
        model_path: Ruta al archivo del modelo
        
    Returns:
        Modelo cargado
    """
    import joblib
    model = joblib.load(model_path)
    logging.info(f"Modelo cargado desde: {model_path}")
    return model


def plot_predictions(y_true: pd.Series,
                     y_pred: pd.Series,
                     title: str = "Predicciones vs Valores Reales",
                     save_path: str = None):
    """
    Grafica predicciones vs valores reales.
    
    Args:
        y_true: Serie con valores reales
        y_pred: Serie con valores predichos
        title: Título del gráfico
        save_path: Ruta para guardar la figura (opcional)
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 6))
    plt.plot(y_true.index, y_true.values, label='Real', alpha=0.7)
    plt.plot(y_pred.index, y_pred.values, label='Predicción', alpha=0.7)
    plt.xlabel('Fecha')
    plt.ylabel('Volumen (m³)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logging.info(f"Figura guardada en: {save_path}")
    
    plt.show()


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Genera un resumen de un DataFrame.
    
    Args:
        df: DataFrame a resumir
        
    Returns:
        Diccionario con información del DataFrame
    """
    summary = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    return summary
