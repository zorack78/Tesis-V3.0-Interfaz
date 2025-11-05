"""
Módulo de definición y configuración de modelos predictivos.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class BaseModel:
    """
    Clase base para modelos predictivos.
    """
    
    def __init__(self, config: dict):
        """
        Inicializa el modelo base.
        
        Args:
            config: Diccionario de configuración
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Entrena el modelo.
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
        """
        raise NotImplementedError("Debe implementarse en subclase")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones.
        
        Args:
            X: Features para predicción
            
        Returns:
            Array de predicciones
        """
        raise NotImplementedError("Debe implementarse en subclase")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Obtiene la importancia de features.
        
        Returns:
            DataFrame con importancia de features
        """
        raise NotImplementedError("Debe implementarse en subclase")


class RandomForestModel(BaseModel):
    """
    Modelo Random Forest para predicción.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        model_config = self.config['models']['random_forest']
        self.model = RandomForestRegressor(**model_config)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Entrena el modelo Random Forest."""
        self.logger.info("Entrenando Random Forest...")
        self.feature_names = X.columns.tolist()
        
        # Escalar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar
        self.model.fit(X_scaled, y)
        
        self.logger.info("Random Forest entrenado exitosamente")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Realiza predicciones con Random Forest."""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Obtiene importancia de features."""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


class XGBoostModel(BaseModel):
    """
    Modelo XGBoost para predicción.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        try:
            import xgboost as xgb
            model_config = self.config['models']['xgboost']
            self.model = xgb.XGBRegressor(**model_config)
        except ImportError:
            self.logger.error("XGBoost no está instalado")
            raise
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Entrena el modelo XGBoost."""
        self.logger.info("Entrenando XGBoost...")
        self.feature_names = X.columns.tolist()
        
        # XGBoost puede manejar datos sin escalar
        self.model.fit(X, y)
        
        self.logger.info("XGBoost entrenado exitosamente")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Realiza predicciones con XGBoost."""
        predictions = self.model.predict(X)
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Obtiene importancia de features."""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


class LightGBMModel(BaseModel):
    """
    Modelo LightGBM para predicción.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        try:
            import lightgbm as lgb
            model_config = self.config['models']['lightgbm']
            self.model = lgb.LGBMRegressor(**model_config)
        except ImportError:
            self.logger.error("LightGBM no está instalado")
            raise
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Entrena el modelo LightGBM."""
        self.logger.info("Entrenando LightGBM...")
        self.feature_names = X.columns.tolist()
        
        # LightGBM puede manejar datos sin escalar
        self.model.fit(X, y)
        
        self.logger.info("LightGBM entrenado exitosamente")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Realiza predicciones con LightGBM."""
        predictions = self.model.predict(X)
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Obtiene importancia de features."""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


class ProphetModel:
    """
    Wrapper para modelo Facebook Prophet.
    """
    
    def __init__(self, config: dict):
        """
        Inicializa Prophet.
        
        Args:
            config: Diccionario de configuración
        """
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    def fit(self, df: pd.DataFrame, target_col: str = 'Volumen_Total_m3'):
        """
        Entrena Prophet.
        
        Args:
            df: DataFrame con columnas 'timestamp_utc' y target
            target_col: Nombre de la columna objetivo
        """
        try:
            from prophet import Prophet
            
            self.logger.info("Entrenando Prophet...")
            
            # Preparar datos para Prophet (requiere 'ds' y 'y')
            prophet_df = pd.DataFrame({
                'ds': df['timestamp_utc'],
                'y': df[target_col]
            })
            
            # Configuración de Prophet
            prophet_config = self.config['models']['prophet']
            self.model = Prophet(**prophet_config)
            
            # Agregar feriados si están disponibles
            if 'feriado' in df.columns:
                holidays = df[df['feriado'] == 1][['timestamp_utc']].copy()
                holidays.columns = ['ds']
                holidays['holiday'] = 'feriado'
                self.model.add_country_holidays(country_name='CL')
            
            # Entrenar
            self.model.fit(prophet_df)
            
            self.logger.info("Prophet entrenado exitosamente")
            
        except ImportError:
            self.logger.error("Prophet no está instalado")
            raise
    
    def predict(self, periods: int = 24, freq: str = 'H') -> pd.DataFrame:
        """
        Realiza predicciones futuras.
        
        Args:
            periods: Número de períodos a predecir
            freq: Frecuencia ('H' para horario)
            
        Returns:
            DataFrame con predicciones
        """
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast
    
    def plot_components(self):
        """Grafica componentes de la serie temporal."""
        from prophet.plot import plot_components_plotly
        return plot_components_plotly(self.model, self.model.predict(
            self.model.make_future_dataframe(periods=0)
        ))


def get_model(model_name: str, config: dict) -> BaseModel:
    """
    Factory para crear modelos.
    
    Args:
        model_name: Nombre del modelo ('random_forest', 'xgboost', 'lightgbm', 'prophet')
        config: Diccionario de configuración
        
    Returns:
        Instancia del modelo
    """
    models = {
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'prophet': ProphetModel
    }
    
    if model_name not in models:
        raise ValueError(f"Modelo desconocido: {model_name}")
    
    return models[model_name](config)
