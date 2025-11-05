"""
Módulo para ingeniería de features (características) del modelo.
"""

import pandas as pd
import numpy as np
from typing import List
import logging


class FeatureEngineer:
    """
    Clase para crear features a partir de los datos de series temporales.
    """
    
    def __init__(self, config: dict):
        """
        Inicializa el ingeniero de features.
        
        Args:
            config: Diccionario de configuración
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_config = config.get('features', {})
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features temporales básicas.
        
        Args:
            df: DataFrame con columna timestamp_utc
            
        Returns:
            DataFrame con features temporales
        """
        self.logger.info("Creando features temporales...")
        
        df_features = df.copy()
        
        # Asegurar que timestamp es datetime
        if 'timestamp_utc' in df_features.columns:
            timestamp_col = 'timestamp_utc'
            df_features[timestamp_col] = pd.to_datetime(
                df_features[timestamp_col]
            )
            
            # Features básicas
            dt = df_features[timestamp_col].dt
            df_features['hour'] = dt.hour
            df_features['day_of_week'] = dt.dayofweek
            df_features['day_of_month'] = dt.day
            df_features['month'] = dt.month
            df_features['quarter'] = dt.quarter
            df_features['year'] = dt.year
            df_features['week_of_year'] = dt.isocalendar().week
            
            # Encoding cíclico para hora
            hour_radians = 2 * np.pi * df_features['hour'] / 24
            df_features['hour_sin'] = np.sin(hour_radians)
            df_features['hour_cos'] = np.cos(hour_radians)
            
            # Encoding cíclico para día de la semana
            dow_radians = 2 * np.pi * df_features['day_of_week'] / 7
            df_features['day_of_week_sin'] = np.sin(dow_radians)
            df_features['day_of_week_cos'] = np.cos(dow_radians)
            
            # Encoding cíclico para mes
            month_radians = 2 * np.pi * df_features['month'] / 12
            df_features['month_sin'] = np.sin(month_radians)
            df_features['month_cos'] = np.cos(month_radians)
            
            # Features binarias
            hour_col = df_features['hour']
            dow_col = df_features['day_of_week']
            df_features['is_weekend'] = (dow_col >= 5).astype(int)
            df_features['is_morning'] = ((hour_col >= 6) &
                                         (hour_col < 12)).astype(int)
            df_features['is_afternoon'] = ((hour_col >= 12) &
                                           (hour_col < 18)).astype(int)
            df_features['is_evening'] = ((hour_col >= 18) &
                                         (hour_col < 22)).astype(int)
            df_features['is_night'] = ((hour_col >= 22) |
                                       (hour_col < 6)).astype(int)
        
        msg = f"Features temporales creadas: {df_features.shape[1]} columnas"
        self.logger.info(msg)
        
        return df_features
    
    def create_lag_features(self,
                            df: pd.DataFrame,
                            target_col: str = 'Volumen_Total_m3',
                            lag_hours: List[int] = None) -> pd.DataFrame:
        """
        Crea features de lags (valores pasados).
        
        Args:
            df: DataFrame con datos
            target_col: Columna objetivo para crear lags
            lag_hours: Lista de horas para crear lags
            
        Returns:
            DataFrame con features de lag
        """
        self.logger.info("Creando features de lag...")
        
        if lag_hours is None:
            default_lags = [1, 2, 3, 6, 12, 24, 48, 168]
            lag_hours = self.feature_config.get('lag_hours', default_lags)
        
        df_lags = df.copy()
        
        if target_col in df_lags.columns:
            for lag in lag_hours:
                col_name = f'{target_col}_lag_{lag}h'
                df_lags[col_name] = df_lags[target_col].shift(lag)
                self.logger.debug(f"Creado lag de {lag} horas")
        
        self.logger.info(f"Features de lag creadas: {len(lag_hours)} lags")
        
        return df_lags
    
    def create_rolling_features(self,
                               df: pd.DataFrame,
                               target_col: str = 'Volumen_Total_m3',
                               windows: List[int] = None) -> pd.DataFrame:
        """
        Crea features de ventanas móviles (rolling).
        
        Args:
            df: DataFrame con datos
            target_col: Columna objetivo
            windows: Lista de tamaños de ventana (en horas)
            
        Returns:
            DataFrame con features rolling
        """
        self.logger.info("Creando features de ventanas móviles...")
        
        if windows is None:
            windows = self.feature_config.get('rolling_windows', [6, 12, 24, 168])
        
        df_rolling = df.copy()
        
        if target_col in df_rolling.columns:
            for window in windows:
                # Media móvil
                col_mean = f'{target_col}_rolling_mean_{window}h'
                df_rolling[col_mean] = df_rolling[target_col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Desviación estándar móvil
                col_std = f'{target_col}_rolling_std_{window}h'
                df_rolling[col_std] = df_rolling[target_col].rolling(
                    window=window, min_periods=1
                ).std()
                
                # Máximo móvil
                col_max = f'{target_col}_rolling_max_{window}h'
                df_rolling[col_max] = df_rolling[target_col].rolling(
                    window=window, min_periods=1
                ).max()
                
                # Mínimo móvil
                col_min = f'{target_col}_rolling_min_{window}h'
                df_rolling[col_min] = df_rolling[target_col].rolling(
                    window=window, min_periods=1
                ).min()
                
                self.logger.debug(f"Creadas features rolling de {window} horas")
        
        self.logger.info(f"Features rolling creadas para {len(windows)} ventanas")
        
        return df_rolling
    
    def create_difference_features(self,
                                   df: pd.DataFrame,
                                   target_col: str = 'Volumen_Total_m3') -> pd.DataFrame:
        """
        Crea features de diferencias (cambios).
        
        Args:
            df: DataFrame con datos
            target_col: Columna objetivo
            
        Returns:
            DataFrame con features de diferencia
        """
        self.logger.info("Creando features de diferencia...")
        
        df_diff = df.copy()
        
        if target_col in df_diff.columns:
            # Diferencia de 1 hora
            df_diff[f'{target_col}_diff_1h'] = df_diff[target_col].diff(1)
            
            # Diferencia de 24 horas (día)
            df_diff[f'{target_col}_diff_24h'] = df_diff[target_col].diff(24)
            
            # Diferencia de 168 horas (semana)
            df_diff[f'{target_col}_diff_168h'] = df_diff[target_col].diff(168)
            
            # Cambio porcentual
            df_diff[f'{target_col}_pct_change_1h'] = df_diff[target_col].pct_change(1)
            df_diff[f'{target_col}_pct_change_24h'] = df_diff[target_col].pct_change(24)
        
        self.logger.info("Features de diferencia creadas")
        
        return df_diff
    
    def create_calendar_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea interacciones con features de calendario.
        
        Args:
            df: DataFrame con features de calendario
            
        Returns:
            DataFrame con interacciones
        """
        self.logger.info("Creando interacciones de calendario...")
        
        df_inter = df.copy()
        
        # Interacción feriado x fin de semana
        if 'feriado' in df_inter.columns and 'is_weekend' in df_inter.columns:
            df_inter['feriado_weekend'] = (
                df_inter['feriado'] * df_inter['is_weekend']
            )
        
        # Interacción vacaciones x fin de semana
        if 'vacaciones_escolares' in df_inter.columns and 'is_weekend' in df_inter.columns:
            df_inter['vacaciones_weekend'] = (
                df_inter['vacaciones_escolares'] * df_inter['is_weekend']
            )
        
        # Eventos especiales (cualquier tipo de evento)
        event_cols = ['feriado', 'elecciones', 'festival_vina', 
                      'fiestas_patrias', 'anio_nuevo']
        existing_event_cols = [col for col in event_cols if col in df_inter.columns]
        
        if existing_event_cols:
            df_inter['any_special_event'] = df_inter[existing_event_cols].max(axis=1)
        
        self.logger.info("Interacciones de calendario creadas")
        
        return df_inter
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea todas las features configuradas.
        
        Args:
            df: DataFrame con datos base
            
        Returns:
            DataFrame con todas las features
        """
        self.logger.info("Creando todas las features...")
        
        df_features = df.copy()
        
        # Features temporales
        df_features = self.create_temporal_features(df_features)
        
        # Features de lag
        df_features = self.create_lag_features(df_features)
        
        # Features rolling
        df_features = self.create_rolling_features(df_features)
        
        # Features de diferencia
        df_features = self.create_difference_features(df_features)
        
        # Interacciones de calendario
        df_features = self.create_calendar_interactions(df_features)
        
        # Eliminar filas con NaN generadas por lags/rolling
        n_before = len(df_features)
        df_features = df_features.dropna()
        n_after = len(df_features)
        
        if n_before != n_after:
            self.logger.info(f"Eliminadas {n_before - n_after} filas con NaN")
        
        self.logger.info(f"Features totales creadas: {df_features.shape[1]} columnas")
        
        return df_features


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Función de conveniencia para crear todas las features.
    
    Args:
        df: DataFrame con datos procesados
        config: Diccionario de configuración
        
    Returns:
        DataFrame con features engineered
    """
    engineer = FeatureEngineer(config)
    df_features = engineer.create_all_features(df)
    return df_features
