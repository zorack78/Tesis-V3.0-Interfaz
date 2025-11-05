#!/usr/bin/env python3
"""
Script para entrenar un modelo y generar gráfica comparativa predicción vs real.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import load_config, calculate_metrics
from feature_engineering import FeatureEngineer

def load_and_prepare_data():
    """Carga y prepara los datos para el modelo."""
    print("📊 Cargando datos procesados...")
    
    data_path = Path("data/processed")
    
    # Verificar que existen los datos
    if not (data_path / "data_train.csv").exists():
        print("❌ No se encontraron datos procesados")
        print("💡 Ejecuta primero: python run_pipeline.py")
        return None
    
    # Cargar datos
    train_df = pd.read_csv(data_path / "data_train.csv")
    val_df = pd.read_csv(data_path / "data_validation.csv")
    test_df = pd.read_csv(data_path / "data_test.csv")
    
    print(f"✅ Datos cargados: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    return train_df, val_df, test_df

def create_features_for_model(train_df, val_df, test_df):
    """Crea features básicas para el modelo."""
    print("⚙️ Creando features...")
    
    config = load_config()
    feature_engineer = FeatureEngineer(config)
    
    # Crear features temporales
    train_features = feature_engineer.create_temporal_features(train_df)
    val_features = feature_engineer.create_temporal_features(val_df)
    test_features = feature_engineer.create_temporal_features(test_df)
    
    # Convertir timestamps para usar como índice después
    for df in [train_features, val_features, test_features]:
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    
    print(f"✅ Features creadas: {train_features.shape[1]} columnas")
    return train_features, val_features, test_features

def prepare_model_data(train_features, val_features, test_features):
    """Prepara los datos para entrenar el modelo."""
    print("🔧 Preparando datos del modelo...")
    
    # Columna objetivo
    target_col = ' Volumen_Total_m3'
    
    # Features seleccionadas
    feature_cols = [
        'hour', 'day_of_week', 'month', 'quarter',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'is_weekend', 'is_morning', 'is_afternoon', 'is_evening', 'is_night'
    ]
    
    # Agregar features de calendario si existen
    calendar_features = ['feriado', 'es_fin_de_semana', 'anio_nuevo', 
                        'fiestas_patrias', 'temporada_turistica_alta',
                        'vacaciones_escolares', 'festival_vina']
    
    for feature in calendar_features:
        if feature in train_features.columns:
            feature_cols.append(feature)
    
    print(f"📋 Usando {len(feature_cols)} features")
    
    # Preparar datasets
    X_train = train_features[feature_cols].fillna(0)
    y_train = train_features[target_col]
    
    X_val = val_features[feature_cols].fillna(0)
    y_val = val_features[target_col]
    
    X_test = test_features[feature_cols].fillna(0)
    y_test = test_features[target_col]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

def train_model(X_train, y_train, X_val, y_val):
    """Entrena un modelo Random Forest."""
    print("🤖 Entrenando modelo Random Forest...")
    
    # Modelo optimizado para este tipo de datos
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Entrenar
    model.fit(X_train, y_train)
    
    # Evaluar en validación
    y_val_pred = model.predict(X_val)
    val_metrics = calculate_metrics(y_val.values, y_val_pred)
    
    print(f"✅ Modelo entrenado - Validación:")
    print(f"   📊 RMSE: {val_metrics['rmse']:,.0f} m³")
    print(f"   📊 MAE: {val_metrics['mae']:,.0f} m³")
    print(f"   📊 MAPE: {val_metrics['mape']:.2f}%")
    print(f"   📊 R²: {val_metrics['r2']:.4f}")
    
    return model

def create_predictions_comparison(model, test_features, X_test, y_test, feature_cols):
    """Crea predicciones y compara con valores reales."""
    print("🔍 Generando predicciones para comparación...")
    
    # Hacer predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas finales
    test_metrics = calculate_metrics(y_test.values, y_pred)
    
    print(f"📊 MÉTRICAS FINALES EN TEST:")
    print(f"   📈 RMSE: {test_metrics['rmse']:,.0f} m³")
    print(f"   📈 MAE: {test_metrics['mae']:,.0f} m³")
    print(f"   📈 MAPE: {test_metrics['mape']:.2f}%")
    print(f"   📈 R²: {test_metrics['r2']:.4f}")
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame({
        'timestamp': test_features['timestamp_utc'].values,
        'real': y_test.values,
        'prediccion': y_pred,
        'error': np.abs(y_test.values - y_pred),
        'error_pct': np.abs((y_test.values - y_pred) / y_test.values) * 100
    })
    
    return results_df, test_metrics

def create_comparison_plots(results_df, test_metrics):
    """Crea gráficas comparativas detalladas."""
    print("📈 Creando gráficas comparativas...")
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Serie temporal completa
    ax1 = plt.subplot(3, 2, (1, 2))
    ax1.plot(results_df['timestamp'], results_df['real'], 
             label='Valores Reales', linewidth=1.5, alpha=0.8, color='blue')
    ax1.plot(results_df['timestamp'], results_df['prediccion'], 
             label='Predicciones', linewidth=1.5, alpha=0.8, color='red')
    ax1.set_title('Comparación Predicción vs Realidad - Serie Temporal Completa', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Volumen (m³)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Zoom a una semana
    sample_size = min(168, len(results_df))  # 1 semana o menos
    sample_df = results_df.head(sample_size)
    
    ax2 = plt.subplot(3, 2, 3)
    ax2.plot(range(len(sample_df)), sample_df['real'], 
             label='Real', marker='o', linewidth=2, markersize=4, alpha=0.8)
    ax2.plot(range(len(sample_df)), sample_df['prediccion'], 
             label='Predicción', marker='s', linewidth=2, markersize=4, alpha=0.8)
    ax2.set_title(f'Detalle - Primeras {len(sample_df)} Horas', fontweight='bold')
    ax2.set_xlabel('Horas')
    ax2.set_ylabel('Volumen (m³)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter plot predicción vs real
    ax3 = plt.subplot(3, 2, 4)
    ax3.scatter(results_df['real'], results_df['prediccion'], alpha=0.6, s=20)
    
    # Línea diagonal perfecta
    min_val = min(results_df['real'].min(), results_df['prediccion'].min())
    max_val = max(results_df['real'].max(), results_df['prediccion'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
    
    ax3.set_title('Scatter Plot: Predicción vs Real', fontweight='bold')
    ax3.set_xlabel('Valores Reales (m³)')
    ax3.set_ylabel('Predicciones (m³)')
    ax3.grid(True, alpha=0.3)
    
    # Agregar métricas al gráfico
    textstr = f'R² = {test_metrics["r2"]:.4f}\nRMSE = {test_metrics["rmse"]:,.0f}\nMAE = {test_metrics["mae"]:,.0f}\nMAPE = {test_metrics["mape"]:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # 4. Distribución de errores
    ax4 = plt.subplot(3, 2, 5)
    ax4.hist(results_df['error'], bins=50, alpha=0.7, edgecolor='black')
    ax4.set_title('Distribución de Errores Absolutos', fontweight='bold')
    ax4.set_xlabel('Error Absoluto (m³)')
    ax4.set_ylabel('Frecuencia')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(results_df['error'].mean(), color='red', linestyle='--', 
                label=f'Error Promedio: {results_df["error"].mean():,.0f} m³')
    ax4.legend()
    
    # 5. Errores a lo largo del tiempo
    ax5 = plt.subplot(3, 2, 6)
    ax5.plot(results_df['timestamp'], results_df['error'], alpha=0.7, linewidth=1)
    ax5.set_title('Error Absoluto a lo Largo del Tiempo', fontweight='bold')
    ax5.set_xlabel('Fecha')
    ax5.set_ylabel('Error Absoluto (m³)')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prediccion_vs_real_{timestamp}.png"
    filepath = output_dir / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Gráfico guardado: {filepath}")
    
    # Mostrar gráfico
    plt.show()
    
    return filepath

def print_detailed_analysis(results_df):
    """Imprime análisis detallado de los resultados."""
    print("\n" + "="*60)
    print("📊 ANÁLISIS DETALLADO DE RESULTADOS")
    print("="*60)
    
    print(f"\n📈 ESTADÍSTICAS GENERALES:")
    print(f"   • Total de predicciones: {len(results_df):,}")
    print(f"   • Volumen real promedio: {results_df['real'].mean():,.0f} m³")
    print(f"   • Volumen predicho promedio: {results_df['prediccion'].mean():,.0f} m³")
    print(f"   • Error absoluto promedio: {results_df['error'].mean():,.0f} m³")
    print(f"   • Error porcentual promedio: {results_df['error_pct'].mean():.2f}%")
    
    print(f"\n🎯 PRECISIÓN DEL MODELO:")
    # Categorizar errores
    excellent = (results_df['error_pct'] <= 5).sum()
    good = ((results_df['error_pct'] > 5) & (results_df['error_pct'] <= 10)).sum()
    fair = ((results_df['error_pct'] > 10) & (results_df['error_pct'] <= 20)).sum()
    poor = (results_df['error_pct'] > 20).sum()
    
    total = len(results_df)
    print(f"   • Excelente (≤5% error): {excellent:,} ({excellent/total*100:.1f}%)")
    print(f"   • Bueno (5-10% error): {good:,} ({good/total*100:.1f}%)")
    print(f"   • Regular (10-20% error): {fair:,} ({fair/total*100:.1f}%)")
    print(f"   • Malo (>20% error): {poor:,} ({poor/total*100:.1f}%)")
    
    print(f"\n🔍 CASOS EXTREMOS:")
    best_idx = results_df['error_pct'].idxmin()
    worst_idx = results_df['error_pct'].idxmax()
    
    print(f"   • Mejor predicción:")
    print(f"     Fecha: {results_df.loc[best_idx, 'timestamp']}")
    print(f"     Real: {results_df.loc[best_idx, 'real']:,.0f} m³")
    print(f"     Predicción: {results_df.loc[best_idx, 'prediccion']:,.0f} m³")
    print(f"     Error: {results_df.loc[best_idx, 'error_pct']:.2f}%")
    
    print(f"   • Peor predicción:")
    print(f"     Fecha: {results_df.loc[worst_idx, 'timestamp']}")
    print(f"     Real: {results_df.loc[worst_idx, 'real']:,.0f} m³")
    print(f"     Predicción: {results_df.loc[worst_idx, 'prediccion']:,.0f} m³")
    print(f"     Error: {results_df.loc[worst_idx, 'error_pct']:.2f}%")

def main():
    """Función principal."""
    print("🚀 GENERANDO COMPARACIÓN PREDICCIÓN vs REALIDAD")
    print("=" * 60)
    
    # 1. Cargar datos
    data = load_and_prepare_data()
    if data is None:
        return False
    train_df, val_df, test_df = data
    
    # 2. Crear features
    train_features, val_features, test_features = create_features_for_model(
        train_df, val_df, test_df
    )
    
    # 3. Preparar datos del modelo
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_model_data(
        train_features, val_features, test_features
    )
    
    # 4. Entrenar modelo
    model = train_model(X_train, y_train, X_val, y_val)
    
    # 5. Generar predicciones
    results_df, test_metrics = create_predictions_comparison(
        model, test_features, X_test, y_test, feature_cols
    )
    
    # 6. Crear gráficas
    graph_path = create_comparison_plots(results_df, test_metrics)
    
    # 7. Análisis detallado
    print_detailed_analysis(results_df)
    
    print(f"\n🎉 PROCESO COMPLETADO")
    print(f"📊 Gráfica guardada en: {graph_path}")
    print(f"✅ Modelo entrenado y evaluado exitosamente")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("❌ Error en el proceso")
        sys.exit(1)