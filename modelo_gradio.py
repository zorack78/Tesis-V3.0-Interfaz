#!/usr/bin/env python3
"""
Modelo mejorado simplificado para interface Gradio.
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
from sklearn.preprocessing import StandardScaler
import joblib

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import load_config, calculate_metrics
from feature_engineering import FeatureEngineer


class GradioWaterDemandModel:
    """Modelo optimizado para interfaz Gradio."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
    def create_enhanced_features(self, df):
        """Crea features mejoradas sin ser muy pesado."""
        print("⚙️ Creando features mejoradas...")
        
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
        
        # 1. Features LAG importantes
        print("   📊 Creando features LAG...")
        lag_hours = [1, 24, 168]  # 1h, 1d, 1w
        for lag in lag_hours:
            df_features[f'lag_{lag}h'] = df_features[target_col].shift(lag)
        
        # 2. Rolling features selectivas
        print("   📈 Creando rolling features...")
        windows = [24, 168]  # 1d, 1w
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
        
        # 3. Features de cambios
        print("   🔄 Creando cambios...")
        df_features['diff_24h'] = df_features[target_col].diff(24)
        df_features['ratio_vs_24h'] = (
            df_features[target_col] / df_features['lag_24h']
        ).fillna(1)
        
        # 4. Features estacionales
        print("   🌊 Creando features estacionales...")
        df_features['is_summer'] = df_features['month'].isin(
            [12, 1, 2]
        ).astype(int)
        df_features['is_peak_hour'] = df_features['hour'].isin(
            [7, 8, 9, 19, 20, 21]
        ).astype(int)
        
        # 5. Feature de interacción principal
        if 'feriado' in df_features.columns:
            df_features['hour_x_feriado'] = (
                df_features['hour'] * df_features['feriado']
            )
        
        print(f"✅ Features mejoradas creadas: {df_features.shape[1]} columnas")
        return df_features
    
    def select_features(self, df_features):
        """Selecciona features para el modelo."""
        
        # Features básicas
        basic_features = [
            'hour', 'day_of_week', 'month', 'quarter',
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'is_weekend', 'is_morning', 'is_afternoon', 'is_evening'
        ]
        
        # Features adicionales
        additional_features = [
            'lag_1h', 'lag_24h', 'lag_168h',
            'rolling_mean_24h', 'rolling_mean_168h',
            'rolling_std_24h', 'rolling_std_168h',
            'diff_24h', 'ratio_vs_24h',
            'is_summer', 'is_peak_hour'
        ]
        
        # Features de calendario
        calendar_features = []
        for feature in ['feriado', 'es_fin_de_semana', 
                       'temporada_turistica_alta', 'hour_x_feriado']:
            if feature in df_features.columns:
                calendar_features.append(feature)
        
        # Combinar todas
        all_features = basic_features + additional_features + calendar_features
        
        # Filtrar las que existen
        selected_features = [
            f for f in all_features if f in df_features.columns
        ]
        
        print(f"📋 Features seleccionadas: {len(selected_features)}")
        return selected_features
    
    def train_optimized_model(self, X_train, y_train, X_val, y_val):
        """Entrena modelo optimizado sin grid search."""
        print("🤖 Entrenando modelo optimizado...")
        
        # Parámetros optimizados (basados en experiencia previa)
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Entrenar
        self.model.fit(X_train, y_train)
        
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
    
    def save_model_for_gradio(self, model_dir="models/gradio"):
        """Guarda modelo para Gradio."""
        print("💾 Guardando modelo para Gradio...")
        
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Usar nombres fijos para Gradio
        model_file = model_path / "water_demand_model.pkl"
        features_file = model_path / "features.txt"
        
        # Guardar modelo
        joblib.dump(self.model, model_file)
        
        # Guardar features
        with open(features_file, 'w') as f:
            for feature in self.feature_columns:
                f.write(f"{feature}\n")
        
        print(f"   ✅ Modelo: {model_file}")
        print(f"   ✅ Features: {features_file}")
        
        return model_file, features_file
    
    def predict_with_uncertainty(self, X_test):
        """Predicciones con estimación de incertidumbre."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        # Predicción principal
        y_pred = self.model.predict(X_test)
        
        # Estimación simple de incertidumbre usando árboles
        tree_predictions = np.array([
            tree.predict(X_test) for tree in self.model.estimators_[:50]  # Solo 50 árboles
        ])
        
        pred_std = np.std(tree_predictions, axis=0)
        
        return {
            'prediction': y_pred,
            'std': pred_std,
            'lower': y_pred - pred_std,
            'upper': y_pred + pred_std
        }


def main():
    """Función principal."""
    print("🚀 MODELO OPTIMIZADO PARA GRADIO")
    print("=" * 50)
    
    # Cargar configuración
    config = load_config()
    
    # Cargar datos
    print("📊 Cargando datos...")
    data_path = Path("data/processed")
    
    if not (data_path / "data_train.csv").exists():
        print("❌ Ejecuta primero: python run_pipeline.py")
        return False
    
    train_df = pd.read_csv(data_path / "data_train.csv")
    val_df = pd.read_csv(data_path / "data_validation.csv")
    test_df = pd.read_csv(data_path / "data_test.csv")
    
    print(f"✅ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Inicializar modelo
    model = GradioWaterDemandModel(config)
    
    # Crear features
    print("\n🔧 Procesando features...")
    train_features = model.create_enhanced_features(train_df)
    val_features = model.create_enhanced_features(val_df)
    test_features = model.create_enhanced_features(test_df)
    
    # Seleccionar features
    target_col = ' Volumen_Total_m3'
    feature_columns = model.select_features(train_features)
    model.feature_columns = feature_columns
    
    # Preparar datos
    X_train = train_features[feature_columns].fillna(0)
    y_train = train_features[target_col]
    
    X_val = val_features[feature_columns].fillna(0)
    y_val = val_features[target_col]
    
    X_test = test_features[feature_columns].fillna(0)
    y_test = test_features[target_col]
    
    print(f"📐 Dimensiones: X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Entrenar
    print("\n🎯 Entrenando modelo...")
    val_metrics = model.train_optimized_model(X_train, y_train, X_val, y_val)
    
    # Evaluar en test
    print("\n🧪 Evaluación final...")
    predictions_result = model.predict_with_uncertainty(X_test)
    y_pred = predictions_result['prediction']
    
    test_metrics = calculate_metrics(y_test.values, y_pred)
    
    print(f"📊 MÉTRICAS FINALES:")
    print(f"   • RMSE: {test_metrics['rmse']:,.0f} m³")
    print(f"   • MAE: {test_metrics['mae']:,.0f} m³")
    print(f"   • MAPE: {test_metrics['mape']:.2f}%")
    print(f"   • R²: {test_metrics['r2']:.4f}")
    
    # Top features
    print(f"\n🔍 TOP 10 FEATURES:")
    importances = model.model.feature_importances_
    feature_importance = list(zip(feature_columns, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance[:10]):
        print(f"   {i+1:2d}. {feature:<20} {importance:.4f}")
    
    # Guardar modelo
    model_files = model.save_model_for_gradio()
    
    # Gráfica simple
    print("\n📈 Creando gráfica...")
    create_simple_plot(y_test, predictions_result, test_metrics)
    
    print(f"\n🎉 MODELO LISTO PARA GRADIO")
    print(f"📁 Archivos: {model_files[0].parent}")
    
    return True


def create_simple_plot(y_test, predictions_result, metrics):
    """Crea gráfica simple pero informativa."""
    
    y_pred = predictions_result['prediction']
    y_std = predictions_result['std']
    
    # Tomar muestra
    n_points = min(150, len(y_test))
    indices = np.linspace(0, len(y_test)-1, n_points, dtype=int)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Serie temporal
    ax1.plot(y_test.iloc[indices], 'b-', label='Real', alpha=0.8)
    ax1.plot(y_pred[indices], 'r-', label='Predicción', alpha=0.8)
    ax1.fill_between(range(len(indices)), 
                     y_pred[indices] - y_std[indices],
                     y_pred[indices] + y_std[indices],
                     alpha=0.3, color='orange', label='Incertidumbre')
    ax1.set_title('🎯 Predicción vs Realidad')
    ax1.set_xlabel('Observaciones')
    ax1.set_ylabel('Volumen (m³)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter
    ax2.scatter(y_test, y_pred, alpha=0.6, s=20)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_title('📊 Correlación')
    ax2.set_xlabel('Real (m³)')
    ax2.set_ylabel('Predicción (m³)')
    ax2.grid(True, alpha=0.3)
    
    # Errores
    errors = np.abs(y_test - y_pred)
    ax3.hist(errors, bins=30, alpha=0.7, color='lightcoral')
    ax3.set_title('📈 Distribución de Errores')
    ax3.set_xlabel('Error Absoluto (m³)')
    ax3.set_ylabel('Frecuencia')
    ax3.grid(True, alpha=0.3)
    
    # Métricas
    ax4.axis('off')
    metrics_text = f"""
📊 MÉTRICAS DEL MODELO

• R² Score: {metrics['r2']:.4f}
• RMSE: {metrics['rmse']:,.0f} m³
• MAE: {metrics['mae']:,.0f} m³
• MAPE: {metrics['mape']:.2f}%

🎯 EVALUACIÓN

{"🎉 EXCELENTE" if metrics['r2'] > 0.8 else 
 "👍 BUENO" if metrics['r2'] > 0.6 else 
 "⚠️ MEJORABLE"}

✅ Modelo listo para Gradio
    """
    
    ax4.text(0.1, 0.9, metrics_text, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Guardar
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"modelo_gradio_{timestamp}.png"
    filepath = output_dir / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   💾 Gráfica guardada: {filepath}")
    
    plt.show()


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)