"""
Script simple: Extrae métricas de la interfaz Gradio ya ejecutada
Genera solo las gráficas principales con valores REALES
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# Colores personalizados
COLORS = {
    'XGBoost V3.0': '#2E86AB',
    'RandomForest': '#A23B72',
    'LightGBM': '#F18701'
}

# MÉTRICAS REALES de la interfaz (04/12/2025 22:16)
METRICAS_REALES = {
    'XGBoost V3.0': {
        'r2': 0.9905,
        'rmse': 425,
        'mae': 269,
        'mape': 2.67
    },
    'RandomForest': {
        'r2': 0.9842,
        'rmse': 548,
        'mae': 295,
        'mape': 3.15
    },
    'LightGBM': {
        'r2': 0.9923,
        'rmse': 382,
        'mae': 261,
        'mape': 2.59
    }
}


def graficar_metricas_comparativas(metricas, output_dir):
    """Gráfica comparativa de las 4 métricas principales"""
    print("\n📊 Generando gráfica comparativa de métricas...")
    
    modelos = list(metricas.keys())
    r2_vals = [metricas[m]['r2'] for m in modelos]
    rmse_vals = [metricas[m]['rmse'] for m in modelos]
    mae_vals = [metricas[m]['mae'] for m in modelos]
    mape_vals = [metricas[m]['mape'] for m in modelos]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = ('Comparación de Métricas de Desempeño - Modelos ML\n' + 
             '(Datos Reales del Test Set: Jun-Sep 2025)')
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    
    colors_list = [COLORS[m] for m in modelos]
    
    # R²
    ax1 = axes[0, 0]
    bars1 = ax1.bar(modelos, r2_vals, color=colors_list, 
                    edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_title('R² - Coeficiente de Determinación', fontweight='bold')
    ax1.set_ylim([min(r2_vals) - 0.005, 1.0])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars1, r2_vals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # RMSE
    ax2 = axes[0, 1]
    bars2 = ax2.bar(modelos, rmse_vals, color=colors_list, 
                    edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('RMSE (m³/hr)', fontweight='bold')
    ax2.set_title('RMSE - Error Cuadrático Medio', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars2, rmse_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 15,
                f'{val:,.0f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # MAE
    ax3 = axes[1, 0]
    bars3 = ax3.bar(modelos, mae_vals, color=colors_list, 
                    edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('MAE (m³/hr)', fontweight='bold')
    ax3.set_title('MAE - Error Absoluto Medio', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars3, mae_vals):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{val:,.0f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # MAPE
    ax4 = axes[1, 1]
    bars4 = ax4.bar(modelos, mape_vals, color=colors_list, 
                    edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('MAPE (%)', fontweight='bold')
    ax4.set_title('MAPE - Error Porcentual Absoluto Medio', 
                 fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars4, mape_vals):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    
    output_path = output_dir / '01_comparacion_metricas_completa_REAL.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Guardada: {output_path.name}")
    plt.close()


def nota_scatter_plot():
    """Nota sobre scatter plots"""
    print("\n📊 Nota sobre Scatter Plots:")
    print("   El archivo '05_scatter_predicho_vs_real.png' requiere")
    print("   predicciones reales de cada modelo sobre el test set.")
    print("   Para obtenerlo:")
    print("   1. Abre la interfaz Gradio")
    print("   2. Ve al tab '🔬 Comparación Modelos'")
    print("   3. Click en 'Ejecutar Comparación'")
    print("   4. La gráfica interactiva mostrará el scatter plot real")
    print("   Alternativamente, los R² mostrados arriba son los correctos.")


def generar_reporte(metricas, output_dir):
    """Genera reporte con métricas reales"""
    print("\n📝 Generando reporte de métricas...")
    
    reporte = []
    reporte.append("=" * 80)
    reporte.append("REPORTE DE MÉTRICAS - COMPARACIÓN MODELOS ML (DATOS REALES)")
    reporte.append("Predicción de Demanda de Agua Potable - Gran Valparaíso")
    reporte.append("=" * 80)
    reporte.append(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    reporte.append("Test Set: Junio - Septiembre 2025 (2,256 registros)")
    reporte.append("Fuente: Interfaz Gradio - Comparación Modelos ML\n")
    
    reporte.append("-" * 80)
    reporte.append(f"{'MODELO':<25} {'R²':>12} {'RMSE':>12} {'MAE':>12} {'MAPE':>12}")
    reporte.append("-" * 80)
    
    for modelo, m in metricas.items():
        reporte.append(f"{modelo:<25} {m['r2']:>12.4f} {m['rmse']:>12,.0f} "
                      f"{m['mae']:>12,.0f} {m['mape']:>11.2f}%")
    
    reporte.append("-" * 80)
    
    reporte.append("\n" + "=" * 80)
    reporte.append("MEJOR MODELO POR MÉTRICA")
    reporte.append("=" * 80)
    
    mejor_r2 = max(metricas.items(), key=lambda x: x[1]['r2'])
    mejor_rmse = min(metricas.items(), key=lambda x: x[1]['rmse'])
    mejor_mae = min(metricas.items(), key=lambda x: x[1]['mae'])
    mejor_mape = min(metricas.items(), key=lambda x: x[1]['mape'])
    
    reporte.append(f"\n🥇 R² más alto: {mejor_r2[0]} ({mejor_r2[1]['r2']:.4f})")
    reporte.append(f"🥇 RMSE más bajo: {mejor_rmse[0]} "
                  f"({mejor_rmse[1]['rmse']:,.0f} m³/hr)")
    reporte.append(f"🥇 MAE más bajo: {mejor_mae[0]} "
                  f"({mejor_mae[1]['mae']:,.0f} m³/hr)")
    reporte.append(f"🥇 MAPE más bajo: {mejor_mape[0]} "
                  f"({mejor_mape[1]['mape']:.2f}%)")
    
    reporte.append("\n" + "=" * 80)
    reporte.append("OBSERVACIONES")
    reporte.append("=" * 80)
    reporte.append("\n• Estas métricas corresponden a los valores REALES obtenidos")
    reporte.append("  de la interfaz Gradio (Tab 'Comparación Modelos ML')")
    reporte.append("• Test set: 15% final del dataset (Jun-Sep 2025)")
    reporte.append("• Métricas calculadas sobre Demanda Qout (no Q_net directo)")
    reporte.append("• LightGBM muestra el mejor desempeño en todas las métricas")
    
    reporte.append("\n" + "=" * 80)
    
    output_path = output_dir / 'reporte_metricas_modelos_REAL.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(reporte))
    
    print(f"   ✅ Guardado: {output_path.name}")
    print("\n" + "\n".join(reporte))


def main():
    """Función principal"""
    print("\n" + "="*80)
    print("GENERADOR DE GRÁFICAS - MÉTRICAS REALES DE INTERFAZ")
    print("="*80)
    print("\nUsando métricas de: Interfaz Gradio (04/12/2025 22:16)")
    print("Test Set: Junio - Septiembre 2025")
    
    output_dir = Path('outputs/metricas_ML')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Usar métricas reales de la interfaz
        metricas = METRICAS_REALES
        print(f"\n✅ Métricas cargadas para {len(metricas)} modelos")
        
        # Generar gráfica comparativa
        graficar_metricas_comparativas(metricas, output_dir)
        
        # Generar reporte
        generar_reporte(metricas, output_dir)
        
        # Nota sobre scatter plots
        nota_scatter_plot()
        
        print("\n" + "="*80)
        print("✅ GENERACIÓN COMPLETADA")
        print(f"📁 Archivos guardados en: {output_dir.absolute()}")
        print("="*80)
        print("\n⚠️  IMPORTANTE:")
        print("   El archivo '05_scatter_predicho_vs_real.png' NO se actualizó")
        print("   porque contiene datos sintéticos del script anterior.")
        print("   Para scatter plots reales, elimina ese archivo y usa")
        print("   las gráficas de la interfaz Gradio directamente.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
