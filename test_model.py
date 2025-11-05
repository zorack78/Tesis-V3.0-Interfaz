#!/usr/bin/env python3
"""
Script de prueba para entrenar un modelo simple y hacer predicciones.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import load_config, calculate_metrics
from feature_engineering import FeatureEngineer

def load_processed_data():
    """Carga los datos ya procesados."""
    print("📊 Cargando datos procesados...")
    
    data_path = Path("data/processed")
    
    try:
        train_df = pd.read_csv(data_path / "data_train.csv")
        val_df = pd.read_csv(data_path / "data_validation.csv")
        test_df = pd.read_csv(data_path / "data_test.csv")
        
        print(f"✅ Datos cargados:")
        print(f"   📚 Train: {len(train_df)} registros")
        print(f"   🔍 Val: {len(val_df)} registros")
        print(f"   🧪 Test: {len(test_df)} registros")
        
        return train_df, val_df, test_df
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        print("💡 Ejecuta primero: python run_pipeline.py")
        return None, None, None

def create_features(train_df, val_df, test_df, config):
    """Crea features para los datasets."""
    print("\n⚙️ Creando features...")
    
    try:
        feature_engineer = FeatureEngineer(config)
        
        # Crear features temporales básicas
        print("   ⏰ Features temporales...")
        train_features = feature_engineer.create_temporal_features(train_df)
        val_features = feature_engineer.create_temporal_features(val_df)
        test_features = feature_engineer.create_temporal_features(test_df)
        
        print(f"✅ Features creadas: {train_features.shape[1]} columnas")
        
        return train_features, val_features, test_features
        
    except Exception as e:
        print(f"❌ Error creando features: {e}")
        return None, None, None

def prepare_model_data(train_features, val_features, test_features):
    """Prepara los datos para el modelo."""
    print("\n🔧 Preparando datos para el modelo...")
    
    # Columna objetivo
    target_col = ' Volumen_Total_m3'
    
    # Features básicas para el modelo
    feature_cols = [
        'hour', 'day_of_week', 'month', 'quarter',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'is_weekend', 'is_morning', 'is_afternoon', 'is_evening'
    ]
    
    # Agregar features de calendario si existen
    calendar_features = ['feriado', 'es_fin_de_semana', 'anio_nuevo', 
                        'fiestas_patrias', 'temporada_turistica_alta']
    
    for feature in calendar_features:
        if feature in train_features.columns:
            feature_cols.append(feature)
    
    print(f"   📋 Features seleccionadas: {len(feature_cols)}")
    for feature in feature_cols:
        print(f"      • {feature}")
    
    # Preparar conjuntos de datos
    X_train = train_features[feature_cols].fillna(0)
    y_train = train_features[target_col]
    
    X_val = val_features[feature_cols].fillna(0)
    y_val = val_features[target_col]
    
    X_test = test_features[feature_cols].fillna(0)
    y_test = test_features[target_col]
    
    print(f"✅ Datos preparados:")
    print(f"   📊 X_train: {X_train.shape}")
    print(f"   📊 X_val: {X_val.shape}")
    print(f"   📊 X_test: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

def train_simple_model(X_train, y_train, X_val, y_val):
    """Entrena un modelo Random Forest simple."""
    print("\n🤖 Entrenando modelo Random Forest...")
    
    try:
        # Modelo simple pero efectivo
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Entrenar
        print("   🔄 Entrenando...")
        model.fit(X_train, y_train)
        
        # Predecir en validación
        print("   🔍 Validando...")
        y_val_pred = model.predict(X_val)
        
        # Calcular métricas
        val_metrics = calculate_metrics(y_val.values, y_val_pred)
        
        print(f"✅ Modelo entrenado:")
        print(f"   📊 RMSE: {val_metrics['rmse']:,.0f} m³")
        print(f"   📊 MAE: {val_metrics['mae']:,.0f} m³")
        print(f"   📊 MAPE: {val_metrics['mape']:.2f}%")
        print(f"   📊 R²: {val_metrics['r2']:.4f}")
        
        return model, val_metrics
        
    except Exception as e:
        print(f"❌ Error entrenando modelo: {e}")
        return None, None

def test_model(model, X_test, y_test, feature_cols):
    """Prueba el modelo en el conjunto de test."""
    print("\n🧪 Evaluando en conjunto de prueba...")
    
    try:
        # Predicciones
        y_test_pred = model.predict(X_test)
        
        # Métricas
        test_metrics = calculate_metrics(y_test.values, y_test_pred)
        
        print(f"📊 RESULTADOS FINALES:")
        print(f"   📈 RMSE: {test_metrics['rmse']:,.0f} m³")
        print(f"   📈 MAE: {test_metrics['mae']:,.0f} m³")
        print(f"   📈 MAPE: {test_metrics['mape']:.2f}%")
        print(f"   📈 R²: {test_metrics['r2']:.4f}")
        
        # Importancia de features
        print(f"\n🔍 TOP 10 FEATURES MÁS IMPORTANTES:")
        importances = model.feature_importances_
        feature_importance = list(zip(feature_cols, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"   {i+1:2d}. {feature:<20} {importance:.4f}")
        
        # Ejemplos de predicciones
        print(f"\n📋 EJEMPLOS DE PREDICCIONES:")
        n_examples = min(5, len(y_test))
        for i in range(n_examples):
            real = y_test.iloc[i]
            pred = y_test_pred[i]
            error = abs(real - pred)
            error_pct = (error / real) * 100
            print(f"   Real: {real:6.0f} m³ | Pred: {pred:6.0f} m³ | Error: {error:4.0f} m³ ({error_pct:.1f}%)")
        
        return test_metrics, y_test_pred
        
    except Exception as e:
        print(f"❌ Error evaluando modelo: {e}")
        return None, None

def main():
    """Función principal para entrenar y probar un modelo."""
    print("🚀 ENTRENAMIENTO DE MODELO DE PRUEBA")
    print("=" * 50)
    
    # Cargar configuración
    config = load_config()
    
    # Cargar datos
    train_df, val_df, test_df = load_processed_data()
    if train_df is None:
        return False
    
    # Crear features
    train_features, val_features, test_features = create_features(
        train_df, val_df, test_df, config
    )
    if train_features is None:
        return False
    
    # Preparar datos
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_model_data(
        train_features, val_features, test_features
    )
    
    # Entrenar modelo
    model, val_metrics = train_simple_model(X_train, y_train, X_val, y_val)
    if model is None:
        return False
    
    # Evaluar modelo
    test_metrics, predictions = test_model(model, X_test, y_test, feature_cols)
    if test_metrics is None:
        return False
    
    print("\n" + "=" * 50)
    print("🎯 ENTRENAMIENTO COMPLETADO")
    print("=" * 50)
    print("✅ Modelo entrenado y evaluado exitosamente")
    print("✅ El sistema está listo para modelos más complejos")
    
    # Guardar modelo simple
    from utils import save_model
    model_path = save_model(model, "random_forest_simple")
    print(f"💾 Modelo guardado en: {model_path}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)