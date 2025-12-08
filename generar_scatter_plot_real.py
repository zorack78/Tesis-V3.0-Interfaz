"""
Script para generar scatter plot de predicciones vs valores reales.
Utiliza datos reales exportados desde la interfaz (rigor científico).

Autor: Sistema de Planificación Qin
Fecha: Diciembre 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


def cargar_datos_reales():
    """Carga predicciones y métricas reales exportadas"""
    base_path = Path('outputs/metricas_ML')
    
    # Cargar predicciones
    csv_path = base_path / 'predicciones_reales.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f"❌ No se encontró {csv_path}\n"
            "   Ejecuta primero la comparación de modelos en Gradio:\n"
            "   Tab '🔬 Comparación Modelos' → 'Ejecutar Comparación'"
        )
    
    df_pred = pd.read_csv(csv_path)
    print(f"✅ Predicciones cargadas: {len(df_pred)} registros")
    
    # Cargar métricas
    json_path = base_path / 'metricas_reales.json'
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            metricas = json.load(f)
        print(f"✅ Métricas cargadas desde {json_path}")
    else:
        metricas = None
        print("⚠️ No se encontraron métricas JSON")
    
    return df_pred, metricas


def generar_scatter_plot(df_pred, metricas):
    """
    Genera scatter plot de 3 modelos vs valores reales.
    
    Args:
        df_pred: DataFrame con columnas demanda_real_m3, pred_*_m3
        metricas: Dict con métricas R² de cada modelo
    """
    # Configurar figura con 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        'Scatter Plot: Predicción vs Demanda Real (Test Set)',
        fontsize=16, fontweight='bold', y=1.02
    )
    
    # Definir modelos y colores
    # Buscar columnas dinámicamente (por si tienen sufijos como "(actual)")
    cols = df_pred.columns.tolist()
    col_xgb = [c for c in cols if 'xgboost' in c.lower()][0]
    col_rf = [c for c in cols if 'randomforest' in c.lower()][0]
    col_lgb = [c for c in cols if 'lightgbm' in c.lower()][0]
    
    modelos_config = [
        {
            'nombre': 'XGBoost V3.0',
            'columna': col_xgb,
            'color': '#1f77b4',
            'ax': axes[0]
        },
        {
            'nombre': 'RandomForest',
            'columna': col_rf,
            'color': '#ff7f0e',
            'ax': axes[1]
        },
        {
            'nombre': 'LightGBM',
            'columna': col_lgb,
            'color': '#2ca02c',
            'ax': axes[2]
        }
    ]
    
    y_real = df_pred['demanda_real_m3'].values
    
    # Calcular rango global para ejes consistentes
    min_val = min(y_real.min(), 
                  df_pred[[m['columna'] for m in modelos_config]].min().min())
    max_val = max(y_real.max(),
                  df_pred[[m['columna'] for m in modelos_config]].max().max())
    padding = (max_val - min_val) * 0.05
    lim = [min_val - padding, max_val + padding]
    
    for modelo in modelos_config:
        ax = modelo['ax']
        y_pred = df_pred[modelo['columna']].values
        
        # Scatter plot con transparencia
        ax.scatter(y_real, y_pred, alpha=0.4, s=20, 
                  color=modelo['color'], edgecolors='none')
        
        # Línea de referencia perfecta (y=x)
        ax.plot(lim, lim, 'k--', alpha=0.5, linewidth=1.5, 
               label='Predicción perfecta')
        
        # Obtener R² desde métricas o calcular
        r2 = None
        if metricas and 'modelos' in metricas:
            # Buscar modelo por coincidencia parcial del nombre
            for nombre_modelo, metricas_modelo in metricas['modelos'].items():
                # Comparar sin "(Actual)" y espacios extra
                if modelo['nombre'].lower().replace(' ', '') in nombre_modelo.lower().replace(' ', '').replace('(actual)', ''):
                    r2 = metricas_modelo.get('r2')
                    break
        
        # Si no se encontró en métricas, calcular
        if r2 is None:
            ss_res = np.sum((y_real - y_pred) ** 2)
            ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
        
        # Título con R²
        ax.set_title(f"{modelo['nombre']}\nR² = {r2:.4f}", 
                    fontsize=12, fontweight='bold')
        
        # Etiquetas
        ax.set_xlabel('Demanda Real (m³/hr)', fontsize=10)
        ax.set_ylabel('Demanda Predicha (m³/hr)', fontsize=10)
        
        # Ajustar límites
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Aspecto cuadrado
        ax.set_aspect('equal', adjustable='box')
        
        # Leyenda
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar
    output_path = Path('outputs/metricas_ML') / \
                  '05_scatter_predicho_vs_real_REAL.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white')
    print(f"\n💾 Scatter plot guardado: {output_path}")
    
    # Mostrar estadísticas
    if metricas and 'periodo' in metricas:
        print(f"\n📊 Información del Test Set:")
        print(f"   Periodo: {metricas['periodo']['inicio']} → "
              f"{metricas['periodo']['fin']}")
        print(f"   N samples: {metricas['n_samples']}")
        print(f"   Fecha generación: {metricas['fecha_generacion']}")
    
    plt.close()
    return output_path


def generar_reporte_estadistico(df_pred, metricas):
    """Genera reporte estadístico de las predicciones"""
    output_path = Path('outputs/metricas_ML') / \
                  'reporte_scatter_plot_REAL.txt'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("REPORTE ESTADÍSTICO - SCATTER PLOT PREDICCIONES REALES\n")
        f.write("=" * 70 + "\n\n")
        
        if metricas:
            f.write(f"Fecha generación: {metricas['fecha_generacion']}\n")
            f.write(f"Periodo test: {metricas['periodo']['inicio']} → "
                   f"{metricas['periodo']['fin']}\n")
            f.write(f"N samples: {metricas['n_samples']}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("MÉTRICAS DE LOS MODELOS\n")
        f.write("-" * 70 + "\n\n")
        
        if metricas and 'modelos' in metricas:
            for nombre, m in metricas['modelos'].items():
                f.write(f"📊 {nombre}:\n")
                f.write(f"   R²:     {m['r2']:.4f}\n")
                f.write(f"   RMSE:   {m['rmse']:,.0f} m³/hr\n")
                f.write(f"   MAE:    {m['mae']:,.0f} m³/hr\n")
                f.write(f"   MAPE:   {m['mape']:.2f}%\n\n")
        
        # Estadísticas de las predicciones
        f.write("-" * 70 + "\n")
        f.write("ESTADÍSTICAS DESCRIPTIVAS\n")
        f.write("-" * 70 + "\n\n")
        
        y_real = df_pred['demanda_real_m3']
        f.write(f"Demanda Real:\n")
        f.write(f"   Media:   {y_real.mean():,.0f} m³/hr\n")
        f.write(f"   Std:     {y_real.std():,.0f} m³/hr\n")
        f.write(f"   Min:     {y_real.min():,.0f} m³/hr\n")
        f.write(f"   Max:     {y_real.max():,.0f} m³/hr\n\n")
        
        # Buscar columnas de predicciones dinámicamente
        cols = df_pred.columns.tolist()
        col_xgb = [c for c in cols if 'xgboost' in c.lower()][0]
        col_rf = [c for c in cols if 'randomforest' in c.lower()][0]
        col_lgb = [c for c in cols if 'lightgbm' in c.lower()][0]
        
        modelos = [
            ('XGBoost V3.0', col_xgb),
            ('RandomForest', col_rf),
            ('LightGBM', col_lgb)
        ]
        
        for nombre, col in modelos:
            if col in df_pred.columns:
                y_pred = df_pred[col]
                error = y_pred - y_real
                f.write(f"{nombre}:\n")
                f.write(f"   Media pred:     {y_pred.mean():,.0f} m³/hr\n")
                f.write(f"   Error medio:    {error.mean():,.0f} m³/hr\n")
                f.write(f"   Error abs medio:{abs(error).mean():,.0f} m³/hr\n")
                f.write(f"   Std error:      {error.std():,.0f} m³/hr\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("RIGOR CIENTÍFICO: Todas las métricas y predicciones\n")
        f.write("provienen de datos reales del test set (Jun-Sep 2025).\n")
        f.write("=" * 70 + "\n")
    
    print(f"📄 Reporte estadístico: {output_path}")
    return output_path


def main():
    """Script principal"""
    print("\n" + "="*70)
    print("GENERACIÓN DE SCATTER PLOT CON PREDICCIONES REALES")
    print("="*70 + "\n")
    
    try:
        # 1. Cargar datos
        print("1️⃣ Cargando predicciones reales...")
        df_pred, metricas = cargar_datos_reales()
        
        # 2. Generar scatter plot
        print("\n2️⃣ Generando scatter plot...")
        scatter_path = generar_scatter_plot(df_pred, metricas)
        
        # 3. Generar reporte
        print("\n3️⃣ Generando reporte estadístico...")
        reporte_path = generar_reporte_estadistico(df_pred, metricas)
        
        print("\n" + "="*70)
        print("✅ PROCESO COMPLETADO CON ÉXITO")
        print("="*70)
        print(f"\n📊 Archivos generados:")
        print(f"   • {scatter_path}")
        print(f"   • {reporte_path}")
        print("\n💡 Estos gráficos utilizan predicciones reales del test set")
        print("   (Jun 26 - Sep 30, 2025) con rigor científico.\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}\n")
        print("📋 INSTRUCCIONES:")
        print("   1. Abre interfaz_gradio_v3.py")
        print("   2. Ve al tab '🔬 Comparación Modelos'")
        print("   3. Click en 'Ejecutar Comparación'")
        print("   4. Espera a que termine (exportará predicciones_reales.csv)")
        print("   5. Ejecuta este script nuevamente\n")
        
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
