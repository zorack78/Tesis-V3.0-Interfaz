"""
Módulo para procesamiento de datos de demanda de agua potable.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
import logging


class DataProcessor:
    """
    Clase para procesar datos de demanda de agua y calendario.
    """
    
    def __init__(self, config: dict):
        """
        Inicializa el procesador de datos.
        
        Args:
            config: Diccionario de configuración
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_volume_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Carga datos de volumen de agua.
        
        Args:
            filepath: Ruta al archivo CSV de volumen
            
        Returns:
            DataFrame con datos de volumen
        """
        if filepath is None:
            raw_dir = Path(self.config['data']['raw_dir'])
            volume_file = self.config['data']['volume_file']
            filepath = raw_dir / volume_file
        
        self.logger.info(f"Cargando datos de volumen desde: {filepath}")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        self.logger.info(f"Datos de volumen cargados: {len(df)} registros")
        
        return df
    
    def load_calendar_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Carga datos de calendario social.
        
        Args:
            filepath: Ruta al archivo CSV de calendario
            
        Returns:
            DataFrame con datos de calendario
        """
        if filepath is None:
            raw_dir = Path(self.config['data']['raw_dir'])
            calendar_file = self.config['data']['calendar_file']
            filepath = raw_dir / calendar_file
        
        self.logger.info(f"Cargando datos de calendario desde: {filepath}")
        
        df = pd.read_csv(filepath)
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        df = df.sort_values('timestamp_utc').reset_index(drop=True)
        
        self.logger.info(f"Datos de calendario cargados: {len(df)} registros")
        
        return df
    
    def merge_volume_calendar(self,
                              volume_df: pd.DataFrame,
                              calendar_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combina datos de volumen y calendario.
        
        Args:
            volume_df: DataFrame de volumen
            calendar_df: DataFrame de calendario
            
        Returns:
            DataFrame combinado
        """
        self.logger.info("Combinando datos de volumen y calendario")
        
        # Renombrar columnas para evitar conflictos
        volume_df = volume_df.rename(columns={'timestamp': 'timestamp_utc'})
        
        # Merge en timestamp UTC
        merged = pd.merge(
            volume_df,
            calendar_df,
            on='timestamp_utc',
            how='left'
        )
        
        self.logger.info(f"Datos combinados: {len(merged)} registros")
        
        return merged
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y preprocesa los datos.
        
        Args:
            df: DataFrame a limpiar
            
        Returns:
            DataFrame limpio
        """
        self.logger.info("Limpiando datos...")
        
        df_clean = df.copy()
        
        # Eliminar duplicados
        n_duplicates = df_clean.duplicated(subset=['timestamp_utc']).sum()
        if n_duplicates > 0:
            msg = f"Eliminando {n_duplicates} registros duplicados"
            self.logger.warning(msg)
            df_clean = df_clean.drop_duplicates(
                subset=['timestamp_utc'], keep='first'
            )
        
        # Verificar valores faltantes en volumen
        if 'Volumen_Total_m3' in df_clean.columns:
            n_missing = df_clean['Volumen_Total_m3'].isnull().sum()
            if n_missing > 0:
                msg = f"Encontrados {n_missing} valores faltantes en volumen"
                self.logger.warning(msg)
                # Interpolar valores faltantes
                vol_col = 'Volumen_Total_m3'
                df_clean[vol_col] = df_clean[vol_col].interpolate(
                    method='time'
                )
        
        # Eliminar outliers extremos (opcional)
        if 'Volumen_Total_m3' in df_clean.columns:
            Q1 = df_clean['Volumen_Total_m3'].quantile(0.01)
            Q3 = df_clean['Volumen_Total_m3'].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            vol_col = 'Volumen_Total_m3'
            n_outliers = ((df_clean[vol_col] < lower_bound) |
                          (df_clean[vol_col] > upper_bound)).sum()
            
            if n_outliers > 0:
                self.logger.info(f"Encontrados {n_outliers} outliers extremos")
        
        # Ordenar por timestamp
        df_clean = df_clean.sort_values('timestamp_utc').reset_index(drop=True)
        
        self.logger.info("Limpieza de datos completada")
        
        return df_clean
    
    def resample_data(self,
                      df: pd.DataFrame,
                      freq: str = 'H',
                      agg_func: str = 'mean') -> pd.DataFrame:
        """
        Remuestrea los datos a una frecuencia específica.
        
        Args:
            df: DataFrame con datos
            freq: Frecuencia de remuestreo ('H' para horario, 'D' para diario)
            agg_func: Función de agregación
            
        Returns:
            DataFrame remuestreado
        """
        self.logger.info(f"Remuestreando datos a frecuencia {freq}")
        
        df_resampled = df.copy()
        df_resampled = df_resampled.set_index('timestamp_utc')
        
        # Remuestrear volumen
        vol_col = 'Volumen_Total_m3'
        if vol_col in df_resampled.columns:
            volume_resampled = (df_resampled[[vol_col]]
                                .resample(freq).agg(agg_func))
            
            # Para otras columnas, tomar el primer valor
            other_cols = [col for col in df_resampled.columns
                          if col != vol_col]
            if other_cols:
                other_resampled = (df_resampled[other_cols]
                                   .resample(freq).first())
                df_resampled = pd.concat([volume_resampled, other_resampled],
                                         axis=1)
            else:
                df_resampled = volume_resampled
        
        df_resampled = df_resampled.reset_index()
        
        msg = f"Remuestreo completado: {len(df_resampled)} registros"
        self.logger.info(msg)
        
        return df_resampled
    
    def split_data(self,
                   df: pd.DataFrame,
                   test_size: float = 0.2,
                   validation_size: float = 0.1) -> Tuple[pd.DataFrame,
                                                          pd.DataFrame,
                                                          pd.DataFrame]:
        """
        Divide los datos en conjuntos de entrenamiento, validación y prueba.
        
        Args:
            df: DataFrame completo
            test_size: Proporción para conjunto de prueba
            validation_size: Proporción para conjunto de validación
            
        Returns:
            Tupla (train_df, val_df, test_df)
        """
        self.logger.info("Dividiendo datos en train/validation/test")
        
        n = len(df)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))
        
        train_df = df.iloc[:val_idx].copy()
        val_df = df.iloc[val_idx:test_idx].copy()
        test_df = df.iloc[test_idx:].copy()
        
        msg = (f"Train: {len(train_df)}, Validation: {len(val_df)}, "
               f"Test: {len(test_df)}")
        self.logger.info(msg)
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """
        Guarda datos procesados.
        
        Args:
            df: DataFrame a guardar
            filename: Nombre del archivo
        """
        output_path = Path(self.config['data']['processed_dir']) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        self.logger.info(f"Datos guardados en: {output_path}")


def load_and_prepare_data(config: dict) -> pd.DataFrame:
    """
    Función de conveniencia para cargar y preparar todos los datos.
    
    Args:
        config: Diccionario de configuración
        
    Returns:
        DataFrame procesado y listo para análisis
    """
    processor = DataProcessor(config)
    
    # Cargar datos
    volume_df = processor.load_volume_data()
    calendar_df = processor.load_calendar_data()
    
    # Combinar
    merged_df = processor.merge_volume_calendar(volume_df, calendar_df)
    
    # Limpiar
    clean_df = processor.clean_data(merged_df)
    
    # Guardar datos procesados
    processor.save_processed_data(clean_df, 'data_processed.csv')
    
    return clean_df
