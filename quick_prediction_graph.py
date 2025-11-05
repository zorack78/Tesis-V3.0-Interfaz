#!/usr/bin/env python3
"""
Script rápido para entrenar modelo y generar gráfica predicción vs real.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import load_config, calculate_metrics
from feature_engineering import FeatureEngineer

def quick_train_and_predict():
    """Función rápida para entrenar y predecir."""
    print("🚀 ENTRENANDO MODELO Y GENERANDO GRÁFICA...")
    
    # 1. Cargar datos procesados
    print("📊 Cargando datos...")
    data_path = Path("data/processed")
    
    if not (data_path / "data_train.csv").exists():
        print("❌ Ejecuta primero: python run_pipeline.py")
        return False
    
    train_df = pd.read_csv(data_path / "data_train.csv")
    test_df = pd.read_csv(data_path / "data_test.csv")
    
    print(f"✅ Train: {len(train_df)}, Test: {len(test_df)}")
    
    # 2. Crear features básicas
    print("⚙️ Creando features...")
    config = load_config()
    feature_engineer = FeatureEngineer(config)
    
    train_features = feature_engineer.create_temporal_features(train_df)
    test_features = feature_engineer.create_temporal_features(test_df)
    
    # Convertir timestamps
    train_features['timestamp_utc'] = pd.to_datetime(train_features['timestamp_utc'])
    test_features['timestamp_utc'] = pd.to_datetime(test_features['timestamp_utc'])
    
    # 3. Preparar datos
    print("🔧 Preparando datos...")
    target_col = ' Volumen_Total_m3'
    
    feature_cols = [
        'hour', 'day_of_week', 'month', 'quarter',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'is_weekend', 'is_morning', 'is_afternoon', 'is_evening'
    ]
    
    # Agregar features de calendario disponibles
    calendar_features = ['feriado', 'es_fin_de_semana', 'temporada_turistica_alta']
    for feature in calendar_features:
        if feature in train_features.columns:
            feature_cols.append(feature)
    
    X_train = train_features[feature_cols].fillna(0)
    y_train = train_features[target_col]
    X_test = test_features[feature_cols].fillna(0)
    y_test = test_features[target_col]
    
    print(f"📋 Usando {len(feature_cols)} features")
    
    # 4. Entrenar modelo
    print("🤖 Entrenando Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 5. Hacer predicciones
    print("🔍 Generando predicciones...")
    y_pred = model.predict(X_test)
    
    # 6. Calcular métricas
    metrics = calculate_metrics(y_test.values, y_pred)
    
    print(f"📊 RESULTADOS:")
    print(f"   RMSE: {metrics['rmse']:,.0f} m³")
    print(f"   MAE: {metrics['mae']:,.0f} m³")
    print(f"   MAPE: {metrics['mape']:.2f}%")
    print(f"   R²: {metrics['r2']:.4f}")
    
    # 7. Crear gráfica
    print("📈 Creando gráfica...")
    
    # Tomar muestra para visualización clara
    n_points = min(200, len(y_test))  # Máximo 200 puntos
    indices = np.linspace(0, len(y_test)-1, n_points, dtype=int)
    
    timestamps_sample = test_features['timestamp_utc'].iloc[indices]
    y_test_sample = y_test.iloc[indices]
    y_pred_sample = y_pred[indices]
    
    # Crear figura
    plt.figure(figsize=(16, 10))
    
    # Subplot 1: Serie temporal
    plt.subplot(2, 2, 1)
    plt.plot(timestamps_sample, y_test_sample, 'b-o', label='Real', 
             linewidth=2, markersize=4, alpha=0.8)
    plt.plot(timestamps_sample, y_pred_sample, 'r-s', label='Predicción', 
             linewidth=2, markersize=4, alpha=0.8)
    plt.title('Predicción vs Realidad - Serie Temporal', fontsize=14, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Volumen (m³)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Subplot 2: Scatter plot
    plt.subplot(2, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.6, s=30)
    
    # Línea diagonal perfecta
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.title('Scatter: Predicción vs Real', fontsize=14, fontweight='bold')
    plt.xlabel('Valores Reales (m³)')
    plt.ylabel('Predicciones (m³)')
    plt.grid(True, alpha=0.3)
    
    # Agregar métricas
    textstr = f'R² = {metrics["r2"]:.3f}\nRMSE = {metrics["rmse"]:,.0f}\nMAE = {metrics["mae"]:,.0f}\nMAPE = {metrics["mape"]:.1f}%'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    # Subplot 3: Errores
    plt.subplot(2, 2, 3)
    errors = np.abs(y_test - y_pred)
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black', color='orange')
    plt.title('Distribución de Errores Absolutos', fontsize=14, fontweight='bold')
    plt.xlabel('Error Absoluto (m³)')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)
    plt.axvline(errors.mean(), color='red', linestyle='--', 
                label=f'Error Promedio: {errors.mean():,.0f} m³')
    plt.legend()
    
    # Subplot 4: Patrón horario
    plt.subplot(2, 2, 4)
    test_features_copy = test_features.copy()
    test_features_copy['y_test'] = y_test.values
    test_features_copy['y_pred'] = y_pred
    
    hourly_real = test_features_copy.groupby('hour')['y_test'].mean()
    hourly_pred = test_features_copy.groupby('hour')['y_pred'].mean()
    
    plt.plot(hourly_real.index, hourly_real.values, 'b-o', label='Real', linewidth=2)
    plt.plot(hourly_pred.index, hourly_pred.values, 'r-s', label='Predicción', linewidth=2)
    plt.title('Patrón Horario Promedio', fontsize=14, fontweight='bold')
    plt.xlabel('Hora del Día')
    plt.ylabel('Volumen Promedio (m³)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))
    
    # Ajustar layout y guardar
    plt.tight_layout()
    
    # Guardar gráfico
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prediccion_vs_real_{timestamp}.png"
    filepath = output_dir / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Gráfica guardada: {filepath}")
    
    # También crear una gráfica simple enfocada
    plt.figure(figsize=(14, 8))
    
    # Tomar solo los primeros 100 puntos para mayor claridad
    n_display = min(100, len(y_test))
    x_range = range(n_display)
    
    plt.plot(x_range, y_test.iloc[:n_display], 'b-o', label='Valores Reales', 
             linewidth=3, markersize=6, alpha=0.8)
    plt.plot(x_range, y_pred[:n_display], 'r-s', label='Predicciones del Modelo', 
             linewidth=3, markersize=6, alpha=0.8)
    
    plt.title('Comparación: Predicción vs Comportamiento Real\nModelo de Demanda de Agua Potable - Gran Valparaíso', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Observación (Orden Temporal)', fontsize=12)
    plt.ylabel('Volumen de Agua (m³)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Agregar información de rendimiento
    info_text = f'Rendimiento del Modelo:\n• R² = {metrics["r2"]:.3f}\n• RMSE = {metrics["rmse"]:,.0f} m³\n• Error Promedio = {metrics["mape"]:.1f}%'
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Guardar versión simple
    simple_filename = f"comparacion_simple_{timestamp}.png"
    simple_filepath = output_dir / simple_filename
    plt.savefig(simple_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Gráfica simple guardada: {simple_filepath}")
    
    # Imprimir resumen
    print(f"\n🎯 RESUMEN DEL ANÁLISIS:")
    print(f"   📊 Total predicciones: {len(y_test):,}")
    print(f"   📈 Precisión (R²): {metrics['r2']:.1%}")
    print(f"   🎯 Error promedio: {metrics['mape']:.1f}%")
    
    # Análisis de precisión por rangos
    errors_pct = np.abs((y_test.values - y_pred) / y_test.values) * 100
    excellent = (errors_pct <= 5).sum()
    good = ((errors_pct > 5) & (errors_pct <= 15)).sum()
    fair = (errors_pct > 15).sum()
    
    print(f"   ✅ Predicciones excelentes (≤5% error): {excellent} ({excellent/len(errors_pct)*100:.1f}%)")
    print(f"   👍 Predicciones buenas (5-15% error): {good} ({good/len(errors_pct)*100:.1f}%)")
    print(f"   ⚠️ Predicciones regulares (>15% error): {fair} ({fair/len(errors_pct)*100:.1f}%)")
    
    return True

if __name__ == "__main__":
    print("📊 GENERADOR RÁPIDO DE GRÁFICA PREDICCIÓN vs REAL")
    print("=" * 60)
    
    success = quick_train_and_predict()
    
    if success:
        print("\n🎉 ¡PROCESO COMPLETADO!")
        print("🔍 Revisa las gráficas en la carpeta: outputs/figures/")
        print("📈 Tienes dos versiones: completa y simple")
    else:
        print("\n❌ Error en el proceso")
        sys.exit(1)