"""
Script para generar gráficas profesionales de métricas de modelos ML
para tesis - Comparación XGBoost, RandomForest y LightGBM

Genera gráficas individuales y comparativas de alta calidad (300 DPI)
guardadas en outputs/metricas_ML/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Colores personalizados para cada modelo
COLORS = {
    'XGBoost V3.0': '#2E86AB',      # Azul
    'RandomForest': '#A23B72',       # Morado
    'LightGBM': '#F18701'            # Naranja
}


def cargar_metricas_desde_interfaz():
    """
    Ejecuta la comparación de modelos y extrae las métricas
    
    Returns:
        dict: Métricas de cada modelo
        dict: Predicciones de cada modelo
        array: Valores reales (y_test)
        array: Timestamps
    """
    print("\n🔬 Ejecutando comparación de modelos...")
    
    # Importar la interfaz
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from interfaz_planificacion_qin_v1 import InterfazPlanificacionQin
    
    # Inicializar interfaz
    interfaz = InterfazPlanificacionQin()
    
    # Ejecutar comparación
    reporte_md, figura = interfaz.comparar_modelos_ml()
    
    # Cargar dataset de test para extraer datos reales
    df_completo_path = Path('data/processed/dataset_features_completo.csv')
    df = pd.read_csv(df_completo_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').copy()
    
    # Obtener periodo de test
    n = len(df)
    val_end = int(n * 0.85)
    df_test = df.iloc[val_end:].copy()
    
    # Calcular Qin histórico por hora
    df_test['hora'] = df_test['timestamp'].dt.hour
    qin_por_hora = df_test['hora'].map(interfaz.qin_perfil_hora).apply(
        lambda x: x['median']
    ).values
    
    # Calcular demanda real: Qout = Qin - Q_net
    y_test = qin_por_hora - df_test['Q_net_m3h'].values
    timestamps = df_test['timestamp'].values
    
    # Obtener predicciones de cada modelo
    X_test = df_test[interfaz.features]
    
    predicciones = {}
    
    # XGBoost
    y_pred_qnet_xgb = interfaz.modelo.predict(X_test)
    predicciones['XGBoost V3.0'] = qin_por_hora - y_pred_qnet_xgb
    
    # RandomForest
    if hasattr(interfaz, 'modelo_rf') and interfaz.modelo_rf is not None:
        y_pred_qnet_rf = interfaz.modelo_rf.predict(X_test)
        predicciones['RandomForest'] = qin_por_hora - y_pred_qnet_rf
    
    # LightGBM
    if hasattr(interfaz, 'modelo_lgb') and interfaz.modelo_lgb is not None:
        y_pred_qnet_lgb = interfaz.modelo_lgb.predict(X_test)
        predicciones['LightGBM'] = qin_por_hora - y_pred_qnet_lgb
    
    # Calcular métricas
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metricas = {}
    for nombre, y_pred in predicciones.items():
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE (ignorar valores muy pequeños)
        mask = np.abs(y_test) > 1000
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        
        metricas[nombre] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    print(f"✅ Métricas extraídas para {len(metricas)} modelos")
    return metricas, predicciones, y_test, timestamps


def graficar_metricas_comparativas(metricas, output_dir):
    """
    Gráfica comparativa de las 4 métricas principales
    
    Args:
        metricas (dict): Diccionario con métricas de cada modelo
        output_dir (Path): Directorio para guardar gráficas
    """
    print("\n📊 Generando gráfica comparativa de métricas...")
    
    modelos = list(metricas.keys())
    
    # Extraer valores
    r2_vals = [metricas[m]['r2'] for m in modelos]
    rmse_vals = [metricas[m]['rmse'] for m in modelos]
    mae_vals = [metricas[m]['mae'] for m in modelos]
    mape_vals = [metricas[m]['mape'] for m in modelos]
    
    # Crear figura con 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparación de Métricas de Desempeño - Modelos ML', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    colors_list = [COLORS[m] for m in modelos]
    
    # 1. R² Score
    ax1 = axes[0, 0]
    bars1 = ax1.bar(modelos, r2_vals, color=colors_list, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_title('R² - Coeficiente de Determinación', fontweight='bold')
    ax1.set_ylim([0.98, 1.0])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Añadir valores en las barras
    for bar, val in zip(bars1, r2_vals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0002,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. RMSE
    ax2 = axes[0, 1]
    bars2 = ax2.bar(modelos, rmse_vals, color=colors_list, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('RMSE (m³/hr)', fontweight='bold')
    ax2.set_title('RMSE - Error Cuadrático Medio', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars2, rmse_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. MAE
    ax3 = axes[1, 0]
    bars3 = ax3.bar(modelos, mae_vals, color=colors_list, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('MAE (m³/hr)', fontweight='bold')
    ax3.set_title('MAE - Error Absoluto Medio', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars3, mae_vals):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 4. MAPE
    ax4 = axes[1, 1]
    bars4 = ax4.bar(modelos, mape_vals, color=colors_list, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('MAPE (%)', fontweight='bold')
    ax4.set_title('MAPE - Error Porcentual Absoluto Medio', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars4, mape_vals):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Rotar etiquetas del eje x
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    
    # Guardar
    output_path = output_dir / '01_comparacion_metricas_completa.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Guardada: {output_path.name}")
    plt.close()


def graficar_metricas_individuales(metricas, output_dir):
    """
    Gráfica individual de cada métrica con ranking
    
    Args:
        metricas (dict): Diccionario con métricas de cada modelo
        output_dir (Path): Directorio para guardar gráficas
    """
    print("\n📊 Generando gráficas individuales por métrica...")
    
    metricas_config = [
        ('r2', 'R² Score', 'R² - Coeficiente de Determinación', 
         lambda x: f'{x:.4f}', True, [0.98, 1.0]),
        ('rmse', 'RMSE (m³/hr)', 'RMSE - Error Cuadrático Medio', 
         lambda x: f'{x:,.0f}', False, None),
        ('mae', 'MAE (m³/hr)', 'MAE - Error Absoluto Medio', 
         lambda x: f'{x:,.0f}', False, None),
        ('mape', 'MAPE (%)', 'MAPE - Error Porcentual Absoluto Medio', 
         lambda x: f'{x:.2f}%', False, None),
    ]
    
    for idx, (metrica_key, ylabel, title, formatter, maximize, ylim) in enumerate(metricas_config, 1):
        # Extraer valores y ordenar
        valores = {modelo: metricas[modelo][metrica_key] for modelo in metricas.keys()}
        valores_sorted = sorted(valores.items(), 
                               key=lambda x: x[1], 
                               reverse=maximize)
        
        modelos_sorted = [m for m, _ in valores_sorted]
        vals_sorted = [v for _, v in valores_sorted]
        colors_sorted = [COLORS[m] for m in modelos_sorted]
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Gráfica de barras horizontales para mejor legibilidad
        bars = ax.barh(modelos_sorted, vals_sorted, color=colors_sorted, 
                       edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel(ylabel, fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        if ylim:
            ax.set_xlim(ylim)
        
        # Añadir valores al final de cada barra
        for bar, val in zip(bars, vals_sorted):
            width = bar.get_width()
            ax.text(width + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01, 
                   bar.get_y() + bar.get_height()/2.,
                   formatter(val), ha='left', va='center', 
                   fontweight='bold', fontsize=11)
        
        # Añadir ranking
        for i, (bar, modelo) in enumerate(zip(bars, modelos_sorted), 1):
            emoji = '🥇' if i == 1 else ('🥈' if i == 2 else '🥉')
            ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02,
                   bar.get_y() + bar.get_height()/2.,
                   emoji, ha='left', va='center', fontsize=14)
        
        plt.tight_layout()
        
        # Guardar
        output_path = output_dir / f'02_{idx}_{metrica_key}_ranking.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Guardada: {output_path.name}")
        plt.close()


def graficar_predicciones_vs_real(predicciones, y_test, timestamps, output_dir):
    """
    Gráfica de predicciones vs valores reales para cada modelo
    
    Args:
        predicciones (dict): Predicciones de cada modelo
        y_test (array): Valores reales
        timestamps (array): Timestamps de los datos
        output_dir (Path): Directorio para guardar gráficas
    """
    print("\n📊 Generando gráficas de predicciones vs real...")
    
    # Convertir timestamps a datetime si no lo son
    if not isinstance(timestamps[0], pd.Timestamp):
        timestamps = pd.to_datetime(timestamps)
    
    # Tomar últimos 7 días (168 horas)
    n_points = min(168, len(y_test))
    timestamps_plot = timestamps[-n_points:]
    y_test_plot = y_test[-n_points:]
    
    for modelo, y_pred in predicciones.items():
        y_pred_plot = y_pred[-n_points:]
        
        # Calcular error
        error = y_pred_plot - y_test_plot
        
        # Crear figura con 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                                gridspec_kw={'height_ratios': [2, 1]})
        
        fig.suptitle(f'Predicciones vs Real - {modelo}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Subplot 1: Predicciones vs Real
        ax1 = axes[0]
        ax1.plot(timestamps_plot, y_test_plot, 
                color='black', linewidth=2, label='Real', alpha=0.8)
        ax1.plot(timestamps_plot, y_pred_plot, 
                color=COLORS[modelo], linewidth=1.5, 
                label=f'Predicción {modelo}', alpha=0.9, linestyle='--')
        
        ax1.set_ylabel('Demanda de Agua Qout (m³/hr)', fontweight='bold', fontsize=12)
        ax1.set_title('Comparación Predicción vs Real (Últimos 7 días)', 
                     fontweight='bold', fontsize=13)
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)
        
        # Subplot 2: Error residual
        ax2 = axes[1]
        ax2.plot(timestamps_plot, error, 
                color=COLORS[modelo], linewidth=1, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.fill_between(timestamps_plot, 0, error, 
                        color=COLORS[modelo], alpha=0.3)
        
        ax2.set_xlabel('Fecha y Hora', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Error (Pred - Real) m³/hr', fontweight='bold', fontsize=12)
        ax2.set_title('Error de Predicción', fontweight='bold', fontsize=13)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=45)
        
        # Añadir estadísticas de error
        mae_val = np.mean(np.abs(error))
        rmse_val = np.sqrt(np.mean(error**2))
        textstr = f'MAE: {mae_val:,.0f} m³/hr\nRMSE: {rmse_val:,.0f} m³/hr'
        ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Guardar
        modelo_clean = modelo.replace(' ', '_').replace('.', '')
        output_path = output_dir / f'03_prediccion_vs_real_{modelo_clean}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Guardada: {output_path.name}")
        plt.close()


def graficar_comparacion_predicciones_unificada(predicciones, y_test, timestamps, output_dir):
    """
    Gráfica unificada con todas las predicciones comparadas con real
    
    Args:
        predicciones (dict): Predicciones de cada modelo
        y_test (array): Valores reales
        timestamps (array): Timestamps de los datos
        output_dir (Path): Directorio para guardar gráficas
    """
    print("\n📊 Generando gráfica comparativa unificada...")
    
    # Convertir timestamps a datetime si no lo son
    if not isinstance(timestamps[0], pd.Timestamp):
        timestamps = pd.to_datetime(timestamps)
    
    # Tomar últimos 7 días
    n_points = min(168, len(y_test))
    timestamps_plot = timestamps[-n_points:]
    y_test_plot = y_test[-n_points:]
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot real (línea sólida negra)
    ax.plot(timestamps_plot, y_test_plot, 
           color='black', linewidth=2.5, label='Real', alpha=0.9, zorder=10)
    
    # Plot predicciones de cada modelo
    for modelo, y_pred in predicciones.items():
        y_pred_plot = y_pred[-n_points:]
        ax.plot(timestamps_plot, y_pred_plot, 
               color=COLORS[modelo], linewidth=1.5, 
               label=modelo, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Fecha y Hora', fontweight='bold', fontsize=13)
    ax.set_ylabel('Demanda de Agua Qout (m³/hr)', fontweight='bold', fontsize=13)
    ax.set_title('Comparación de Predicciones - Todos los Modelos (Últimos 7 días)', 
                fontweight='bold', fontsize=15, pad=15)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Guardar
    output_path = output_dir / '04_comparacion_predicciones_unificada.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Guardada: {output_path.name}")
    plt.close()


def graficar_scatter_predicciones(predicciones, y_test, output_dir):
    """
    Gráficas de dispersión (scatter) predicho vs real para cada modelo
    
    Args:
        predicciones (dict): Predicciones de cada modelo
        y_test (array): Valores reales
        output_dir (Path): Directorio para guardar gráficas
    """
    print("\n📊 Generando gráficas de dispersión...")
    
    # Figura con subplot para cada modelo
    n_models = len(predicciones)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('Valores Predichos vs Valores Reales - Scatter Plot', 
                fontsize=16, fontweight='bold', y=1.02)
    
    for idx, (modelo, y_pred) in enumerate(predicciones.items()):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.5, s=20, 
                  color=COLORS[modelo], edgecolors='black', linewidth=0.5)
        
        # Línea ideal (y = x)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='Ideal (y=x)', alpha=0.8)
        
        # Calcular R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        
        ax.set_xlabel('Demanda Real (m³/hr)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Demanda Predicha (m³/hr)', fontweight='bold', fontsize=11)
        ax.set_title(f'{modelo}\nR² = {r2:.4f}', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    # Guardar
    output_path = output_dir / '05_scatter_predicho_vs_real.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Guardada: {output_path.name}")
    plt.close()


def generar_reporte_metricas(metricas, output_dir):
    """
    Genera reporte en texto plano con todas las métricas
    
    Args:
        metricas (dict): Diccionario con métricas de cada modelo
        output_dir (Path): Directorio para guardar reporte
    """
    print("\n📝 Generando reporte de métricas...")
    
    reporte = []
    reporte.append("=" * 80)
    reporte.append("REPORTE DE MÉTRICAS - COMPARACIÓN MODELOS ML")
    reporte.append("Predicción de Demanda de Agua Potable - Gran Valparaíso")
    reporte.append("=" * 80)
    reporte.append(f"\nFecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Tabla de métricas
    reporte.append("-" * 80)
    reporte.append(f"{'MODELO':<25} {'R²':>12} {'RMSE':>12} {'MAE':>12} {'MAPE':>12}")
    reporte.append("-" * 80)
    
    for modelo, m in metricas.items():
        reporte.append(f"{modelo:<25} {m['r2']:>12.4f} {m['rmse']:>12,.0f} "
                      f"{m['mae']:>12,.0f} {m['mape']:>11.2f}%")
    
    reporte.append("-" * 80)
    
    # Mejor modelo por métrica
    reporte.append("\n" + "=" * 80)
    reporte.append("MEJOR MODELO POR MÉTRICA")
    reporte.append("=" * 80)
    
    mejor_r2 = max(metricas.items(), key=lambda x: x[1]['r2'])
    mejor_rmse = min(metricas.items(), key=lambda x: x[1]['rmse'])
    mejor_mae = min(metricas.items(), key=lambda x: x[1]['mae'])
    mejor_mape = min(metricas.items(), key=lambda x: x[1]['mape'])
    
    reporte.append(f"\n🥇 R² más alto: {mejor_r2[0]} ({mejor_r2[1]['r2']:.4f})")
    reporte.append(f"🥇 RMSE más bajo: {mejor_rmse[0]} ({mejor_rmse[1]['rmse']:,.0f} m³/hr)")
    reporte.append(f"🥇 MAE más bajo: {mejor_mae[0]} ({mejor_mae[1]['mae']:,.0f} m³/hr)")
    reporte.append(f"🥇 MAPE más bajo: {mejor_mape[0]} ({mejor_mape[1]['mape']:.2f}%)")
    
    # Interpretación
    reporte.append("\n" + "=" * 80)
    reporte.append("INTERPRETACIÓN DE MÉTRICAS")
    reporte.append("=" * 80)
    reporte.append("\n• R² (Coeficiente de Determinación):")
    reporte.append("  - Mide proporción de varianza explicada por el modelo")
    reporte.append("  - Rango: 0 a 1 (más cercano a 1 = mejor)")
    reporte.append("  - Interpretación: % de variabilidad de la demanda explicada")
    
    reporte.append("\n• RMSE (Root Mean Squared Error):")
    reporte.append("  - Error cuadrático medio en m³/hr")
    reporte.append("  - Penaliza errores grandes más fuertemente")
    reporte.append("  - Sensible a outliers")
    
    reporte.append("\n• MAE (Mean Absolute Error):")
    reporte.append("  - Error absoluto medio en m³/hr")
    reporte.append("  - Interpretación directa: error promedio de predicción")
    reporte.append("  - Más robusto a outliers que RMSE")
    
    reporte.append("\n• MAPE (Mean Absolute Percentage Error):")
    reporte.append("  - Error porcentual medio")
    reporte.append("  - Interpretación: % de error promedio respecto al valor real")
    reporte.append("  - Útil para comparar modelos en diferentes escalas")
    
    reporte.append("\n" + "=" * 80)
    
    # Guardar reporte
    output_path = output_dir / 'reporte_metricas_modelos.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(reporte))
    
    print(f"   ✅ Guardado: {output_path.name}")
    
    # También imprimir en consola
    print("\n" + "\n".join(reporte))


def main():
    """Función principal"""
    print("\n" + "="*80)
    print("GENERADOR DE GRÁFICAS DE MÉTRICAS - MODELOS ML")
    print("Predicción de Demanda de Agua Potable - Gran Valparaíso")
    print("="*80)
    
    # Directorio de salida
    output_dir = Path('outputs/metricas_ML')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Cargar métricas ejecutando comparación
        metricas, predicciones, y_test, timestamps = cargar_metricas_desde_interfaz()
        
        # 2. Generar gráficas
        graficar_metricas_comparativas(metricas, output_dir)
        graficar_metricas_individuales(metricas, output_dir)
        graficar_predicciones_vs_real(predicciones, y_test, timestamps, output_dir)
        graficar_comparacion_predicciones_unificada(predicciones, y_test, timestamps, output_dir)
        graficar_scatter_predicciones(predicciones, y_test, output_dir)
        
        # 3. Generar reporte
        generar_reporte_metricas(metricas, output_dir)
        
        print("\n" + "="*80)
        print("✅ GENERACIÓN COMPLETADA")
        print(f"📁 Gráficas guardadas en: {output_dir.absolute()}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
