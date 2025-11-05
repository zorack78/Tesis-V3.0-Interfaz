#!/usr/bin/env python3
"""
Modelo mejorado con features avanzadas antes de crear interfaz Gradio.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import load_config, calculate_metrics
from feature_engineering import FeatureEngineer


class AdvancedWaterDemandModel:
    """Modelo avanzado para predicción de demanda de agua."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
    def create_advanced_features(self, df):
        """Crea features avanzadas para mejor predicción."""
        print("⚙️ Creando features avanzadas...")
        
        # Features temporales básicas
        feature_engineer = FeatureEngineer(self.config)
        df_features = feature_engineer.create_temporal_features(df)
        
        # Convertir timestamp
        df_features['timestamp_utc'] = pd.to_datetime(
            df_features['timestamp_utc']
        )
        df_features = df_features.sort_values('timestamp_utc').reset_index(
            drop=True
        )
        
        target_col = ' Volumen_Total_m3'
        
        # 1. Features LAG (valores pasados)
        print("   📊 Creando features LAG...")
        lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h a 1 semana
        for lag in lag_hours:
            df_features[f'lag_{lag}h'] = df_features[target_col].shift(lag)
        
        # 2. Rolling features (promedios móviles)
        print("   📈 Creando rolling features...")
        windows = [6, 12, 24, 168]  # 6h, 12h, 1d, 1w
        for window in windows:
            df_features[f'rolling_mean_{window}h'] = (
                df_features[target_col].rolling(
                    window=window, min_periods=1
                ).mean()
            )
            df_features[f'rolling_std_{window}h'] = (
                df_features[target_col].rolling(
                    window=window, min_periods=1
                ).std()
            )
            df_features[f'rolling_max_{window}h'] = (
                df_features[target_col].rolling(
                    window=window, min_periods=1
                ).max()
            )
            df_features[f'rolling_min_{window}h'] = (
                df_features[target_col].rolling(
                    window=window, min_periods=1
                ).min()
            )
        
        # 3. Features de diferencias
        print("   🔄 Creando features de cambios...")
        df_features['diff_1h'] = df_features[target_col].diff(1)
        df_features['diff_24h'] = df_features[target_col].diff(24)
        df_features['diff_168h'] = df_features[target_col].diff(168)
        
        # 4. Features de ratios
        print("   📊 Creando ratios...")
        df_features['ratio_vs_24h'] = (
            df_features[target_col] / df_features['lag_24h']
        ).fillna(1)
        df_features['ratio_vs_168h'] = (
            df_features[target_col] / df_features['lag_168h']
        ).fillna(1)
        
        # 5. Features de interacción
        print("   🔗 Creando interacciones...")
        if 'feriado' in df_features.columns:
            df_features['hour_x_feriado'] = (
                df_features['hour'] * df_features['feriado']
            )
            df_features['weekend_x_feriado'] = (
                df_features['is_weekend'] * df_features['feriado']
            )
        
        # 6. Features estacionales específicas
        print("   🌊 Creando features estacionales...")
        df_features['is_summer'] = df_features['month'].isin(
            [12, 1, 2]
        ).astype(int)
        df_features['is_winter'] = df_features['month'].isin(
            [6, 7, 8]
        ).astype(int)
        df_features['is_peak_hour'] = df_features['hour'].isin(
            [7, 8, 9, 19, 20, 21]
        ).astype(int)
        df_features['is_low_hour'] = df_features['hour'].isin(
            [0, 1, 2, 3, 4, 5]
        ).astype(int)
        
        # 7. Features de calendario avanzadas
        if 'temporada_turistica_alta' in df_features.columns:
            df_features['turismo_x_weekend'] = (
                df_features['temporada_turistica_alta'] * 
                df_features['is_weekend']
            )
        
        print(f"✅ Features avanzadas creadas: "
              f"{df_features.shape[1]} columnas totales")
        return df_features
    
    def select_best_features(self, df_features, target_col):
        """Selecciona las mejores features para el modelo."""
        print("🎯 Seleccionando mejores features...")
        
        # Features básicas siempre incluidas
        basic_features = [
            'hour', 'day_of_week', 'month', 'quarter',
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'is_weekend', 'is_morning', 'is_afternoon', 'is_evening'
        ]
        
        # Features LAG
        lag_features = [col for col in df_features.columns if col.startswith('lag_')]
        
        # Features rolling
        rolling_features = [col for col in df_features.columns if col.startswith('rolling_')]
        
        # Features de diferencias
        diff_features = [col for col in df_features.columns if col.startswith('diff_')]
        
        # Features de ratios
        ratio_features = [col for col in df_features.columns if col.startswith('ratio_')]
        
        # Features estacionales
        seasonal_features = ['is_summer', 'is_winter', 'is_peak_hour', 'is_low_hour']
        
        # Features de calendario
        calendar_features = []
        for feature in ['feriado', 'es_fin_de_semana', 'temporada_turistica_alta', 
                       'festival_vina', 'elecciones', 'vacaciones_escolares']:
            if feature in df_features.columns:
                calendar_features.append(feature)
        
        # Features de interacción
        interaction_features = [col for col in df_features.columns if '_x_' in col]
        
        # Combinar todas las features
        all_features = (basic_features + lag_features + rolling_features + 
                       diff_features + ratio_features + seasonal_features + 
                       calendar_features + interaction_features)
        
        # Filtrar features que existen
        selected_features = [f for f in all_features if f in df_features.columns]
        
        print(f"   📋 Features seleccionadas: {len(selected_features)}")
        print(f"      • Básicas: {len(basic_features)}")
        print(f"      • LAG: {len(lag_features)}")
        print(f"      • Rolling: {len(rolling_features)}")
        print(f"      • Diferencias: {len(diff_features)}")
        print(f"      • Ratios: {len(ratio_features)}")
        print(f"      • Estacionales: {len(seasonal_features)}")
        print(f"      • Calendario: {len(calendar_features)}")
        print(f"      • Interacciones: {len(interaction_features)}")
        
        return selected_features
    
    def train_advanced_model(self, X_train, y_train, X_val, y_val):
        """Entrena modelo con grid search."""
        print("🤖 Entrenando modelo avanzado con Grid Search...")
        
        # Parámetros para grid search
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Usar TimeSeriesSplit para validación temporal
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        print("   🔍 Ejecutando Grid Search...")
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        print(f"✅ Mejores parámetros encontrados:")
        for param, value in grid_search.best_params_.items():
            print(f"   • {param}: {value}")
        
        # Evaluar en validación
        y_val_pred = self.model.predict(X_val)
        val_metrics = calculate_metrics(y_val.values, y_val_pred)
        
        print(f"📊 MÉTRICAS EN VALIDACIÓN:")
        print(f"   • RMSE: {val_metrics['rmse']:,.0f} m³")
        print(f"   • MAE: {val_metrics['mae']:,.0f} m³")
        print(f"   • MAPE: {val_metrics['mape']:.2f}%")
        print(f"   • R²: {val_metrics['r2']:.4f}")
        
        self.is_trained = True
        return val_metrics
    
    def save_model(self, model_dir="models/advanced"):
        """Guarda el modelo y componentes."""
        print("💾 Guardando modelo avanzado...")
        
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar modelo
        model_file = model_path / f"water_demand_model_{timestamp}.pkl"
        joblib.dump(self.model, model_file)
        
        # Guardar scaler
        scaler_file = model_path / f"scaler_{timestamp}.pkl"
        joblib.dump(self.scaler, scaler_file)
        
        # Guardar features
        features_file = model_path / f"features_{timestamp}.txt"
        with open(features_file, 'w') as f:
            for feature in self.feature_columns:
                f.write(f"{feature}\n")
        
        print(f"   ✅ Modelo guardado: {model_file}")
        print(f"   ✅ Scaler guardado: {scaler_file}")
        print(f"   ✅ Features guardadas: {features_file}")
        
        return model_file, scaler_file, features_file
    
    def predict_with_confidence(self, X_test):
        """Hace predicciones con intervalos de confianza."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        # Predicción principal
        y_pred = self.model.predict(X_test)
        
        # Intervalos de confianza usando árboles individuales
        tree_predictions = np.array([tree.predict(X_test) for tree in self.model.estimators_])
        
        # Estadísticas de los árboles
        pred_std = np.std(tree_predictions, axis=0)
        pred_lower = np.percentile(tree_predictions, 25, axis=0)
        pred_upper = np.percentile(tree_predictions, 75, axis=0)
        
        return {
            'prediction': y_pred,
            'std': pred_std,
            'lower_25': pred_lower,
            'upper_75': pred_upper,
            'confidence_width': pred_upper - pred_lower
        }

def main():
    """Función principal para entrenar modelo avanzado."""
    print("🚀 ENTRENANDO MODELO AVANZADO PARA GRADIO")
    print("=" * 60)
    
    # Cargar configuración
    config = load_config()
    
    # Cargar datos
    print("📊 Cargando datos procesados...")
    data_path = Path("data/processed")
    
    if not (data_path / "data_train.csv").exists():
        print("❌ Ejecuta primero: python run_pipeline.py")
        return False
    
    train_df = pd.read_csv(data_path / "data_train.csv")
    val_df = pd.read_csv(data_path / "data_validation.csv")
    test_df = pd.read_csv(data_path / "data_test.csv")
    
    print(f"✅ Datos cargados: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    # Inicializar modelo
    model = AdvancedWaterDemandModel(config)
    
    # Crear features avanzadas
    train_features = model.create_advanced_features(train_df)
    val_features = model.create_advanced_features(val_df)
    test_features = model.create_advanced_features(test_df)
    
    # Seleccionar mejores features
    target_col = ' Volumen_Total_m3'
    feature_columns = model.select_best_features(train_features, target_col)
    model.feature_columns = feature_columns
    
    # Preparar datos
    X_train = train_features[feature_columns].fillna(0)
    y_train = train_features[target_col]
    
    X_val = val_features[feature_columns].fillna(0)
    y_val = val_features[target_col]
    
    X_test = test_features[feature_columns].fillna(0)
    y_test = test_features[target_col]
    
    # Normalizar features (opcional, para algunos modelos)
    # X_train_scaled = model.scaler.fit_transform(X_train)
    # X_val_scaled = model.scaler.transform(X_val)
    # X_test_scaled = model.scaler.transform(X_test)
    
    # Entrenar modelo
    val_metrics = model.train_advanced_model(X_train, y_train, X_val, y_val)
    
    # Evaluar en test
    print("\n🧪 EVALUACIÓN EN CONJUNTO DE PRUEBA:")
    predictions_result = model.predict_with_confidence(X_test)
    y_pred = predictions_result['prediction']
    
    test_metrics = calculate_metrics(y_test.values, y_pred)
    
    print(f"📊 MÉTRICAS FINALES:")
    print(f"   • RMSE: {test_metrics['rmse']:,.0f} m³")
    print(f"   • MAE: {test_metrics['mae']:,.0f} m³")
    print(f"   • MAPE: {test_metrics['mape']:.2f}%")
    print(f"   • R²: {test_metrics['r2']:.4f}")
    
    # Análisis de importancia de features
    print(f"\n🔍 TOP 15 FEATURES MÁS IMPORTANTES:")
    importances = model.model.feature_importances_
    feature_importance = list(zip(feature_columns, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance[:15]):
        print(f"   {i+1:2d}. {feature:<25} {importance:.4f}")
    
    # Guardar modelo
    model_files = model.save_model()
    
    # Crear gráfica avanzada con intervalos de confianza
    print("\n📈 Creando gráfica avanzada...")
    create_advanced_prediction_plot(test_features, y_test, predictions_result, test_metrics)
    
    print(f"\n🎉 MODELO AVANZADO COMPLETADO")
    print(f"✅ Listo para integrar con Gradio")
    print(f"📁 Archivos del modelo: {model_files[0].parent}")
    
    return True

def create_advanced_prediction_plot(test_features, y_test, predictions_result, metrics):
    """Crea gráfica avanzada con intervalos de confianza."""
    
    y_pred = predictions_result['prediction']
    pred_lower = predictions_result['lower_25']
    pred_upper = predictions_result['upper_75']
    
    # Tomar muestra para visualización
    n_points = min(200, len(y_test))
    indices = np.linspace(0, len(y_test)-1, n_points, dtype=int)
    
    timestamps_sample = test_features['timestamp_utc'].iloc[indices]
    y_test_sample = y_test.iloc[indices]
    y_pred_sample = y_pred[indices]
    pred_lower_sample = pred_lower[indices]
    pred_upper_sample = pred_upper[indices]
    
    # Crear figura
    plt.figure(figsize=(18, 12))
    
    # Subplot 1: Serie temporal con intervalos de confianza
    plt.subplot(2, 3, (1, 2))
    
    # Banda de confianza
    plt.fill_between(range(len(y_test_sample)), pred_lower_sample, pred_upper_sample,
                     alpha=0.3, color='orange', label='Intervalo de Confianza (25-75%)')
    
    # Líneas principales
    plt.plot(range(len(y_test_sample)), y_test_sample, 'b-', 
             label='Valores Reales', linewidth=2, alpha=0.8)
    plt.plot(range(len(y_pred_sample)), y_pred_sample, 'r-', 
             label='Predicciones', linewidth=2, alpha=0.8)
    
    plt.title('🎯 Predicción Avanzada con Intervalos de Confianza', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Observaciones (Orden Temporal)')
    plt.ylabel('Volumen (m³)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Scatter con intervalos
    plt.subplot(2, 3, 3)
    scatter = plt.scatter(y_test, y_pred, c=predictions_result['confidence_width'], 
                         alpha=0.6, s=25, cmap='viridis')
    plt.colorbar(scatter, label='Ancho Intervalo Confianza')
    
    # Línea diagonal
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.title('📊 Scatter con Incertidumbre', fontweight='bold')
    plt.xlabel('Valores Reales (m³)')
    plt.ylabel('Predicciones (m³)')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Distribución de errores
    plt.subplot(2, 3, 4)
    errors = np.abs(y_test - y_pred)
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black', color='lightcoral')
    plt.title('📈 Distribución de Errores', fontweight='bold')
    plt.xlabel('Error Absoluto (m³)')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Errores vs confianza
    plt.subplot(2, 3, 5)
    plt.scatter(predictions_result['confidence_width'], errors, alpha=0.6, s=25)
    plt.title('🔍 Error vs Incertidumbre', fontweight='bold')
    plt.xlabel('Ancho Intervalo Confianza')
    plt.ylabel('Error Absoluto (m³)')
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Métricas resumen
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    metrics_text = f"""
    📊 MÉTRICAS DEL MODELO AVANZADO
    
    • R² Score: {metrics['r2']:.4f}
    • RMSE: {metrics['rmse']:,.0f} m³
    • MAE: {metrics['mae']:,.0f} m³
    • MAPE: {metrics['mape']:.2f}%
    
    📈 ESTADÍSTICAS DE CONFIANZA
    
    • Ancho promedio IC: {np.mean(predictions_result['confidence_width']):,.0f} m³
    • Desviación estándar: {np.mean(predictions_result['std']):,.0f} m³
    
    🎯 CALIDAD PREDICTIVA
    
    {"🎉 EXCELENTE" if metrics['r2'] > 0.8 else 
     "👍 BUENO" if metrics['r2'] > 0.6 else 
     "⚠️ MEJORABLE"}
    """
    
    plt.text(0.1, 0.9, metrics_text, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Guardar
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"modelo_avanzado_{timestamp}.png"
    filepath = output_dir / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   💾 Gráfica avanzada guardada: {filepath}")
    
    plt.show()

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)