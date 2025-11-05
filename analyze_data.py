#!/usr/bin/env python3
"""
Script para analizar los datos procesados y generar insights.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import load_config

def analyze_processed_data():
    """Analiza los datos ya procesados."""
    print("📊 ANÁLISIS DE DATOS PROCESADOS")
    print("=" * 50)
    
    # Cargar datos procesados
    data_path = Path("data/processed")
    
    if not (data_path / "data_processed_complete.csv").exists():
        print("❌ Primero ejecuta: python run_pipeline.py")
        return
    
    # Leer datos
    df = pd.read_csv(data_path / "data_processed_complete.csv")
    train_df = pd.read_csv(data_path / "data_train.csv")
    val_df = pd.read_csv(data_path / "data_validation.csv")
    test_df = pd.read_csv(data_path / "data_test.csv")
    
    print(f"✅ Datos cargados:")
    print(f"   📁 Completo: {len(df):,} registros")
    print(f"   📚 Entrenamiento: {len(train_df):,} registros")
    print(f"   🔍 Validación: {len(val_df):,} registros") 
    print(f"   🧪 Prueba: {len(test_df):,} registros")
    
    # Convertir timestamps
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    
    # Análisis básico
    vol_col = ' Volumen_Total_m3'
    print(f"\n📈 ESTADÍSTICAS DEL VOLUMEN:")
    print(f"   📊 Promedio: {df[vol_col].mean():,.0f} m³/hora")
    print(f"   📊 Mediana: {df[vol_col].median():,.0f} m³/hora")
    print(f"   📊 Desv. estándar: {df[vol_col].std():,.0f} m³/hora")
    print(f"   📊 Mínimo: {df[vol_col].min():,.0f} m³/hora")
    print(f"   📊 Máximo: {df[vol_col].max():,.0f} m³/hora")
    
    # Análisis por día de la semana
    df['day_name'] = df['timestamp_utc'].dt.day_name()
    daily_avg = df.groupby('day_name')[vol_col].mean().sort_values(ascending=False)
    
    print(f"\n📅 CONSUMO PROMEDIO POR DÍA:")
    for day, volume in daily_avg.items():
        print(f"   {day}: {volume:,.0f} m³/hora")
    
    # Análisis por hora
    df['hour'] = df['timestamp_utc'].dt.hour
    hourly_avg = df.groupby('hour')[vol_col].mean()
    
    print(f"\n⏰ PATRONES HORARIOS:")
    peak_hour = hourly_avg.idxmax()
    min_hour = hourly_avg.idxmin()
    print(f"   🔼 Pico máximo: {peak_hour}:00 ({hourly_avg[peak_hour]:,.0f} m³)")
    print(f"   🔽 Mínimo: {min_hour}:00 ({hourly_avg[min_hour]:,.0f} m³)")
    
    # Análisis de feriados
    if 'feriado' in df.columns:
        holiday_avg = df[df['feriado'] == 1][vol_col].mean()
        normal_avg = df[df['feriado'] == 0][vol_col].mean()
        print(f"\n🎉 IMPACTO DE FERIADOS:")
        print(f"   📊 Días normales: {normal_avg:,.0f} m³/hora")
        print(f"   📊 Feriados: {holiday_avg:,.0f} m³/hora")
        print(f"   📈 Diferencia: {((holiday_avg/normal_avg-1)*100):+.1f}%")
    
    return df

def create_simple_visualization(df):
    """Crea visualizaciones básicas."""
    print(f"\n📊 GENERANDO GRÁFICOS...")
    
    vol_col = ' Volumen_Total_m3'
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análisis de Demanda de Agua Potable - Gran Valparaíso', fontsize=16)
    
    # 1. Serie temporal
    sample_df = df.head(24*7)  # Una semana de muestra
    axes[0, 0].plot(sample_df['timestamp_utc'], sample_df[vol_col])
    axes[0, 0].set_title('Serie Temporal (1 semana)')
    axes[0, 0].set_ylabel('Volumen (m³)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Patrón diario promedio
    hourly_avg = df.groupby(df['timestamp_utc'].dt.hour)[vol_col].mean()
    axes[0, 1].plot(hourly_avg.index, hourly_avg.values, marker='o')
    axes[0, 1].set_title('Patrón Diario Promedio')
    axes[0, 1].set_xlabel('Hora del día')
    axes[0, 1].set_ylabel('Volumen promedio (m³)')
    axes[0, 1].grid(True)
    
    # 3. Distribución del volumen
    axes[1, 0].hist(df[vol_col], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribución del Volumen')
    axes[1, 0].set_xlabel('Volumen (m³)')
    axes[1, 0].set_ylabel('Frecuencia')
    
    # 4. Boxplot por día de la semana
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_name'] = df['timestamp_utc'].dt.day_name()
    daily_data = [df[df['day_name'] == day][vol_col].values for day in days_order]
    axes[1, 1].boxplot(daily_data, labels=[d[:3] for d in days_order])
    axes[1, 1].set_title('Distribución por Día de la Semana')
    axes[1, 1].set_ylabel('Volumen (m³)')
    
    plt.tight_layout()
    
    # Guardar gráfico
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "analisis_basico.png", dpi=100, bbox_inches='tight')
    print(f"   ✅ Gráfico guardado: {output_dir / 'analisis_basico.png'}")
    
    plt.show()

def main():
    """Función principal."""
    print("🚀 INICIANDO ANÁLISIS DE DATOS PROCESADOS")
    
    # Analizar datos
    df = analyze_processed_data()
    
    if df is not None:
        # Crear visualizaciones
        create_simple_visualization(df)
        
        print(f"\n🎯 ANÁLISIS COMPLETADO")
        print(f"✅ Los datos están listos para entrenar modelos")
        print(f"✅ Revisa el gráfico generado en outputs/figures/")
    
    return df is not None

if __name__ == "__main__":
    main()