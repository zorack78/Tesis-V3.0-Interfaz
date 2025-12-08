"""
Script para generar gráficas de métricas ML con DATOS REALES
Usa los modelos entrenados y predicciones reales del test set
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

# Colores personalizados
COLORS = {
    'XGBoost V3.0': '#2E86AB',
    'RandomForest': '#A23B72',
    'LightGBM': '#F18701'
}


def obtener_metricas_reales():
    """
    Obtiene métricas REALES ejecutando comparar_modelos_ml() desde la interfaz
    """
    print("\n🔬 Ejecutando comparación de modelos para obtener métricas reales...")
    
    # Importar dependencias
    import joblib
    import json
    from sklearn.ensemble import RandomForestRegressor
    
    # Cargar modelo XGBoost directamente
    print("   Cargando modelo XGBoost...")
    modelo_xgb = joblib.load('models/forecasting/modelo_forecasting_xgboost.pkl')
    
    # Cargar features
    with open('models/forecasting/features.txt') as f:
        features = [line.strip() for line in f]
    print(f"   ✅ {len(features)} features cargadas")
    
    # Cargar umbrales
    umbrales = joblib.load('models/forecasting/umbrales.pkl')
    print(f"   ✅ Umbrales cargados")
    
    # Cargar dataset de test
    df_completo_path = Path('data/processed/dataset_features_completo.csv')
    df = pd.read_csv(df_completo_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').copy()
    
    # Agregar features categóricas manualmente
    print("   Agregando features categóricas...")
    df['temp_nivel_Frio'] = (df['clima_temp_c'] < umbrales['temp_frio']).astype(int)
    df['temp_nivel_Calor'] = (df['clima_temp_c'] > umbrales['temp_calor']).astype(int)
    df['temp_nivel_Normal'] = ((df['clima_temp_c'] >= umbrales['temp_frio']) & 
                                 (df['clima_temp_c'] <= umbrales['temp_calor'])).astype(int)
    
    # Obtener periodo de test
    n = len(df)
    val_end = int(n * 0.85)
    df_test = df.iloc[val_end:].copy()
    
    # Preparar datos
    X_test = df_test[features]
    y_test_qnet = df_test['Q_net_m3h'].values
    timestamps = df_test['timestamp'].values
    
    # Calcular Qin histórico por hora (perfil estándar)
    df_test['hora'] = df_test['timestamp'].dt.hour
    # Calcular Qin promedio por hora desde datos de entrenamiento
    df_train = df.iloc[:val_end].copy()
    qin_perfil = df_train.groupby('hora')['sist_Qin_m3h'].median().to_dict()
    qin_por_hora = df_test['hora'].map(qin_perfil).values
    
    # Calcular demanda real: Qout = Qin - Q_net
    y_test_qout = qin_por_hora - y_test_qnet
    
    print(f"   Test set: {len(y_test_qout):,} registros")
    print(f"   Período: {df_test['timestamp'].min()} a {df_test['timestamp'].max()}")
    
    # Diccionarios para almacenar resultados
    metricas = {}
    predicciones = {}
    
    # 1. XGBoost V3.0
    print("\n📊 Evaluando XGBoost V3.0...")
    y_pred_qnet_xgb = modelo_xgb.predict(X_test)
    y_pred_qout_xgb = qin_por_hora - y_pred_qnet_xgb
    predicciones['XGBoost V3.0'] = y_pred_qout_xgb
    metricas['XGBoost V3.0'] = calcular_metricas(y_test_qout, y_pred_qout_xgb)
    print(f"   ✅ R²: {metricas['XGBoost V3.0']['r2']:.4f} | "
          f"MAE: {metricas['XGBoost V3.0']['mae']:,.0f}")
    
    # 2. RandomForest - Entrenar desde cero
    print("\n🌲 Entrenando RandomForest...")
    df_train = df.iloc[:val_end].copy()
    X_train = df_train[features]
    y_train = df_train['Q_net_m3h'].values
    
    # Rellenar NaN (RandomForest no acepta NaN)
    if X_train.isna().sum().sum() > 0:
        print("   Rellenando NaN con 0...")
        X_train = X_train.fillna(0)
        X_test_rf = X_test.fillna(0)
    else:
        X_test_rf = X_test
    
    modelo_rf = RandomForestRegressor(
        n_estimators=200, 
        max_depth=15, 
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    print(f"   Entrenando con {len(X_train):,} registros...")
    modelo_rf.fit(X_train, y_train)
    
    # Predecir con RandomForest
    y_pred_qnet_rf = modelo_rf.predict(X_test.fillna(0))
    y_pred_qout_rf = qin_por_hora - y_pred_qnet_rf
    predicciones['RandomForest'] = y_pred_qout_rf
    metricas['RandomForest'] = calcular_metricas(y_test_qout, y_pred_qout_rf)
    print(f"   ✅ R²: {metricas['RandomForest']['r2']:.4f} | "
          f"MAE: {metricas['RandomForest']['mae']:,.0f}")
    
    # 3. LightGBM
    try:
        import lightgbm as lgb
        print("\n💡 Entrenando LightGBM...")
        
        modelo_lgb = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        print(f"   Entrenando con {len(X_train):,} registros...")
        modelo_lgb.fit(X_train, y_train)
        
        y_pred_qnet_lgb = modelo_lgb.predict(X_test)
        y_pred_qout_lgb = qin_por_hora - y_pred_qnet_lgb
        predicciones['LightGBM'] = y_pred_qout_lgb
        metricas['LightGBM'] = calcular_metricas(y_test_qout, y_pred_qout_lgb)
        print(f"   ✅ R²: {metricas['LightGBM']['r2']:.4f} | "
              f"MAE: {metricas['LightGBM']['mae']:,.0f}")
    except ImportError:
        print("   ⚠️ LightGBM no disponible")
    
    print(f"\n✅ Métricas reales obtenidas para {len(metricas)} modelos")
    
    return metricas, predicciones, y_test_qout, timestamps


def calcular_metricas(y_true, y_pred):
    """Calcula métricas de evaluación"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (ignorar valores muy pequeños)
    mask = np.abs(y_true) > 1000
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
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
    fig.suptitle('Comparación de Métricas de Desempeño - Modelos ML\n(Datos Reales del Test Set)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    colors_list = [COLORS[m] for m in modelos]
    
    # R²
    ax1 = axes[0, 0]
    bars1 = ax1.bar(modelos, r2_vals, color=colors_list, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_title('R² - Coeficiente de Determinación', fontweight='bold')
    ax1.set_ylim([min(r2_vals) - 0.002, 1.0])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars1, r2_vals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0003,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # RMSE
    ax2 = axes[0, 1]
    bars2 = ax2.bar(modelos, rmse_vals, color=colors_list, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('RMSE (m³/hr)', fontweight='bold')
    ax2.set_title('RMSE - Error Cuadrático Medio', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars2, rmse_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # MAE
    ax3 = axes[1, 0]
    bars3 = ax3.bar(modelos, mae_vals, color=colors_list, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('MAE (m³/hr)', fontweight='bold')
    ax3.set_title('MAE - Error Absoluto Medio', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars3, mae_vals):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # MAPE
    ax4 = axes[1, 1]
    bars4 = ax4.bar(modelos, mape_vals, color=colors_list, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('MAPE (%)', fontweight='bold')
    ax4.set_title('MAPE - Error Porcentual Absoluto Medio', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars4, mape_vals):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    
    output_path = output_dir / '01_comparacion_metricas_completa_REAL.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Guardada: {output_path.name}")
    plt.close()


def graficar_scatter(predicciones, y_test, output_dir):
    """Gráficas de dispersión con datos reales"""
    print("\n📊 Generando gráficas de dispersión...")
    
    n_models = len(predicciones)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('Valores Predichos vs Valores Reales - Scatter Plot\n(Test Set Real: Jun-Sep 2025)', 
                fontsize=16, fontweight='bold', y=1.02)
    
    for idx, (modelo, y_pred) in enumerate(predicciones.items()):
        ax = axes[idx]
        
        ax.scatter(y_test, y_pred, alpha=0.5, s=20, 
                  color=COLORS[modelo], edgecolors='black', linewidth=0.5)
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='Ideal (y=x)', alpha=0.8)
        
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        
        ax.set_xlabel('Demanda Real (m³/hr)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Demanda Predicha (m³/hr)', fontweight='bold', fontsize=11)
        ax.set_title(f'{modelo}\nR² = {r2:.4f}', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    output_path = output_dir / '05_scatter_predicho_vs_real_REAL.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Guardada: {output_path.name}")
    plt.close()


def graficar_comparacion_unificada(predicciones, y_test, timestamps, output_dir):
    """Gráfica unificada con todas las predicciones reales"""
    print("\n📊 Generando gráfica comparativa unificada...")
    
    # Convertir timestamps a datetime
    if not isinstance(timestamps[0], pd.Timestamp):
        timestamps = pd.to_datetime(timestamps)
    
    # Tomar últimos 7 días
    n_points = min(168, len(y_test))
    timestamps_plot = timestamps[-n_points:]
    y_test_plot = y_test[-n_points:]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.plot(timestamps_plot, y_test_plot, 
           color='black', linewidth=2.5, label='Real', alpha=0.9, zorder=10)
    
    for modelo, y_pred in predicciones.items():
        y_pred_plot = y_pred[-n_points:]
        ax.plot(timestamps_plot, y_pred_plot, 
               color=COLORS[modelo], linewidth=1.5, 
               label=modelo, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Fecha y Hora', fontweight='bold', fontsize=13)
    ax.set_ylabel('Demanda de Agua Qout (m³/hr)', fontweight='bold', fontsize=13)
    ax.set_title('Comparación de Predicciones - Todos los Modelos\n(Últimos 7 días del Test Set: Sep 2025)', 
                fontweight='bold', fontsize=15, pad=15)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = output_dir / '04_comparacion_predicciones_unificada_REAL.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Guardada: {output_path.name}")
    plt.close()


def generar_reporte(metricas, output_dir):
    """Genera reporte con métricas reales"""
    print("\n📝 Generando reporte de métricas...")
    
    reporte = []
    reporte.append("=" * 80)
    reporte.append("REPORTE DE MÉTRICAS - COMPARACIÓN MODELOS ML (DATOS REALES)")
    reporte.append("Predicción de Demanda de Agua Potable - Gran Valparaíso")
    reporte.append("=" * 80)
    reporte.append(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    reporte.append(f"Test Set: Junio - Septiembre 2025 (2,256 registros)\n")
    
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
    reporte.append(f"🥇 RMSE más bajo: {mejor_rmse[0]} ({mejor_rmse[1]['rmse']:,.0f} m³/hr)")
    reporte.append(f"🥇 MAE más bajo: {mejor_mae[0]} ({mejor_mae[1]['mae']:,.0f} m³/hr)")
    reporte.append(f"🥇 MAPE más bajo: {mejor_mape[0]} ({mejor_mape[1]['mape']:.2f}%)")
    
    reporte.append("\n" + "=" * 80)
    
    output_path = output_dir / 'reporte_metricas_modelos_REAL.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(reporte))
    
    print(f"   ✅ Guardado: {output_path.name}")
    print("\n" + "\n".join(reporte))


def main():
    """Función principal"""
    print("\n" + "="*80)
    print("GENERADOR DE GRÁFICAS DE MÉTRICAS - DATOS REALES")
    print("="*80)
    
    output_dir = Path('outputs/metricas_ML')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Obtener métricas reales ejecutando modelos
        metricas, predicciones, y_test, timestamps = obtener_metricas_reales()
        
        # 2. Generar gráficas
        graficar_metricas_comparativas(metricas, output_dir)
        graficar_scatter(predicciones, y_test, output_dir)
        graficar_comparacion_unificada(predicciones, y_test, timestamps, output_dir)
        
        # 3. Generar reporte
        generar_reporte(metricas, output_dir)
        
        print("\n" + "="*80)
        print("✅ GENERACIÓN COMPLETADA CON DATOS REALES")
        print(f"📁 Gráficas guardadas en: {output_dir.absolute()}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
