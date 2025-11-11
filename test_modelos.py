#!/usr/bin/env python3
"""
Script para probar y comparar modelos antes de integrar en interfaz.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred):
    """Calcula métricas de evaluación"""
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

def test_modelo(nombre, modelo, X_train, y_train, X_test, y_test):
    """Prueba un modelo y retorna métricas"""
    print(f"\n{'='*60}")
    print(f"🔧 Probando: {nombre}")
    print(f"{'='*60}")
    
    # Entrenar
    print("   Entrenando...")
    modelo.fit(X_train, y_train)
    
    # Predecir
    print("   Prediciendo...")
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    # Métricas
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    # Mostrar resultados
    print(f"\n   📊 TRAIN:")
    print(f"      • RMSE: {train_metrics['rmse']:,.0f} m³")
    print(f"      • MAE:  {train_metrics['mae']:,.0f} m³")
    print(f"      • MAPE: {train_metrics['mape']:.2f}%")
    print(f"      • R²:   {train_metrics['r2']:.4f}")
    
    print(f"\n   🎯 TEST:")
    print(f"      • RMSE: {test_metrics['rmse']:,.0f} m³")
    print(f"      • MAE:  {test_metrics['mae']:,.0f} m³")
    print(f"      • MAPE: {test_metrics['mape']:.2f}%")
    print(f"      • R²:   {test_metrics['r2']:.4f}")
    
    # Verificar overfitting
    diff_r2 = train_metrics['r2'] - test_metrics['r2']
    if diff_r2 > 0.1:
        print(f"\n   ⚠️  ADVERTENCIA: Posible overfitting (diff R²: {diff_r2:.3f})")
    else:
        print(f"\n   ✅ Buena generalización (diff R²: {diff_r2:.3f})")
    
    return test_metrics

def main():
    print("🚀 PRUEBA DE MODELOS - COMPARACIÓN CON DATOS ESTANDARIZADOS")
    print("="*60)
    
    # Cargar datos
    print("\n📊 Cargando datos procesados con features completas...")
    data_path = Path("data/processed")
    
    if not (data_path / "data_train.csv").exists():
        print("❌ No se encontraron datos procesados.")
        print("   Ejecuta primero: python run_pipeline.py")
        return False
    
    train_df = pd.read_csv(data_path / "data_train.csv")
    test_df = pd.read_csv(data_path / "data_test.csv")
    
    print(f"   ✅ Train: {train_df.shape}")
    print(f"   ✅ Test:  {test_df.shape}")
    
    # Verificar target
    target_col = ' Volumen_Total_m3'
    if target_col not in train_df.columns:
        print(f"❌ No se encontró columna target: {target_col}")
        return False
    
    # Seleccionar features (excluyendo fechas, nombres, target)
    exclude_cols = [
        'timestamp_utc', ' Volumen_Total_m3', 'fecha_hora_local', 
        'nombre_feriado', 'fecha_feriado_oficial', 'fecha_feriado_observado'
    ]
    
    available_features = [col for col in train_df.columns if col not in exclude_cols]
    
    # Verificar que tenemos features LAG/ROLLING
    lag_features = [f for f in available_features if 'lag_' in f]
    rolling_features = [f for f in available_features if 'rolling_' in f]
    
    print(f"\n📋 Features disponibles:")
    print(f"   • Total: {len(available_features)}")
    print(f"   • LAG features: {len(lag_features)}")
    print(f"   • ROLLING features: {len(rolling_features)}")
    print(f"   • Ejemplos: {', '.join(available_features[:5])}...")
    
    # Preparar datos
    X_train = train_df[available_features].fillna(0)
    y_train = train_df[target_col]
    
    X_test = test_df[available_features].fillna(0)
    y_test = test_df[target_col]
    
    print(f"\n📐 Dimensiones finales:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    
    # Definir modelos a probar
    modelos = {
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
    }
    
    # Probar cada modelo
    resultados = {}
    for nombre, modelo in modelos.items():
        try:
            metricas = test_modelo(nombre, modelo, X_train, y_train, X_test, y_test)
            resultados[nombre] = metricas
        except Exception as e:
            print(f"\n❌ Error con {nombre}: {e}")
    
    # Comparación final
    print(f"\n\n{'='*60}")
    print("📊 COMPARACIÓN FINAL DE MODELOS")
    print(f"{'='*60}")
    print(f"\n{'Modelo':<20} {'R²':<10} {'RMSE':<12} {'MAE':<12}")
    print("-" * 60)
    
    for nombre, metricas in resultados.items():
        print(f"{nombre:<20} {metricas['r2']:<10.4f} {metricas['rmse']:<12,.0f} {metricas['mae']:<12,.0f}")
    
    # Mejor modelo
    if resultados:
        mejor = max(resultados.items(), key=lambda x: x[1]['r2'])
        print(f"\n🏆 MEJOR MODELO: {mejor[0]} (R² = {mejor[1]['r2']:.4f})")
    
    print("\n✅ Prueba completada exitosamente")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
