"""
Comparación Antes/Después de Corrección Data Leakage
====================================================
Compara métricas y predicciones antes y después de corregir
el perfil de Qin (eliminar datos del test set).
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def cargar_predicciones_antiguas():
    """Carga predicciones con data leakage (si existen)"""
    path = Path('outputs/metricas_ML/predicciones_reales_CON_LEAKAGE.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        print("⚠️ No se encontraron predicciones antiguas guardadas")
        return None


def cargar_predicciones_nuevas():
    """Carga predicciones corregidas (sin leakage)"""
    path = Path('outputs/metricas_ML/predicciones_reales.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        print("❌ No se encontraron predicciones nuevas")
        return None


def calcular_metricas(y_true, y_pred):
    """Calcula métricas de evaluación"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def comparar_metricas(df_antiguas, df_nuevas):
    """Compara métricas antes/después para los 3 modelos"""
    print("\n" + "="*80)
    print("COMPARACIÓN DE MÉTRICAS: ANTES vs DESPUÉS DE CORRECCIÓN")
    print("="*80)
    
    # Buscar columnas de predicción
    cols_real = [c for c in df_antiguas.columns if 'demanda_real' in c.lower()][0]
    
    modelos = ['XGBoost', 'RandomForest', 'LightGBM']
    comparacion = []
    
    for modelo in modelos:
        # Buscar columna del modelo (case insensitive)
        cols_antiguas = [c for c in df_antiguas.columns if modelo.lower() in c.lower()]
        cols_nuevas = [c for c in df_nuevas.columns if modelo.lower() in c.lower()]
        
        if not cols_antiguas or not cols_nuevas:
            print(f"\n⚠️ {modelo}: Columnas no encontradas")
            continue
        
        col_antigua = cols_antiguas[0]
        col_nueva = cols_nuevas[0]
        
        # Calcular métricas
        y_true = df_antiguas[cols_real].values
        y_pred_antigua = df_antiguas[col_antigua].values
        y_pred_nueva = df_nuevas[col_nueva].values
        
        metricas_antigua = calcular_metricas(y_true, y_pred_antigua)
        metricas_nueva = calcular_metricas(y_true, y_pred_nueva)
        
        # Calcular diferencias
        diff_r2 = metricas_nueva['r2'] - metricas_antigua['r2']
        diff_mae = metricas_nueva['mae'] - metricas_antigua['mae']
        diff_rmse = metricas_nueva['rmse'] - metricas_antigua['rmse']
        diff_mape = metricas_nueva['mape'] - metricas_antigua['mape']
        
        print(f"\n📊 {modelo.upper()}")
        print("-" * 80)
        print(f"{'Métrica':<12} {'CON Leakage':>15} {'SIN Leakage':>15} {'Diferencia':>15} {'Cambio':>10}")
        print("-" * 80)
        print(f"{'R²':<12} {metricas_antigua['r2']:>15.4f} {metricas_nueva['r2']:>15.4f} "
              f"{diff_r2:>15.4f} {diff_r2*100:>9.2f}%")
        print(f"{'MAE':<12} {metricas_antigua['mae']:>15.0f} {metricas_nueva['mae']:>15.0f} "
              f"{diff_mae:>15.0f} {(diff_mae/metricas_antigua['mae'])*100:>9.2f}%")
        print(f"{'RMSE':<12} {metricas_antigua['rmse']:>15.0f} {metricas_nueva['rmse']:>15.0f} "
              f"{diff_rmse:>15.0f} {(diff_rmse/metricas_antigua['rmse'])*100:>9.2f}%")
        print(f"{'MAPE (%)':<12} {metricas_antigua['mape']:>15.2f} {metricas_nueva['mape']:>15.2f} "
              f"{diff_mape:>15.2f} {(diff_mape/metricas_antigua['mape'])*100:>9.2f}%")
        
        comparacion.append({
            'modelo': modelo,
            'r2_con_leakage': metricas_antigua['r2'],
            'r2_sin_leakage': metricas_nueva['r2'],
            'diff_r2': diff_r2,
            'mae_con_leakage': metricas_antigua['mae'],
            'mae_sin_leakage': metricas_nueva['mae'],
            'diff_mae': diff_mae,
            'rmse_con_leakage': metricas_antigua['rmse'],
            'rmse_sin_leakage': metricas_nueva['rmse'],
            'diff_rmse': diff_rmse
        })
    
    return pd.DataFrame(comparacion)


def analizar_peak_anomalo(df_antiguas, df_nuevas):
    """Analiza cómo cambió la predicción del peak anómalo (Sep 25, 2025 08:00)"""
    print("\n" + "="*80)
    print("ANÁLISIS DEL PEAK ANÓMALO (Sep 25, 2025 08:00)")
    print("="*80)
    
    fecha_peak = pd.Timestamp('2025-09-25 08:00:00', tz='UTC')
    
    # Buscar el peak en ambos dataframes
    idx_antigua = (df_antiguas['timestamp'] == fecha_peak)
    idx_nueva = (df_nuevas['timestamp'] == fecha_peak)
    
    if not idx_antigua.any() or not idx_nueva.any():
        print("⚠️ Peak no encontrado en los datos")
        return
    
    cols_real = [c for c in df_antiguas.columns if 'demanda_real' in c.lower()][0]
    demanda_real = df_antiguas.loc[idx_antigua, cols_real].values[0]
    
    print(f"\n📈 Demanda Real en Peak: {demanda_real:,.0f} m³/hr")
    print("-" * 80)
    print(f"{'Modelo':<15} {'CON Leakage':>15} {'Error':>12} {'SIN Leakage':>15} {'Error':>12} {'Δ Error':>12}")
    print("-" * 80)
    
    modelos = ['XGBoost', 'RandomForest', 'LightGBM']
    
    for modelo in modelos:
        cols_antiguas = [c for c in df_antiguas.columns if modelo.lower() in c.lower()]
        cols_nuevas = [c for c in df_nuevas.columns if modelo.lower() in c.lower()]
        
        if not cols_antiguas or not cols_nuevas:
            continue
        
        pred_antigua = df_antiguas.loc[idx_antigua, cols_antiguas[0]].values[0]
        pred_nueva = df_nuevas.loc[idx_nueva, cols_nuevas[0]].values[0]
        
        error_antigua = abs(pred_antigua - demanda_real)
        error_nueva = abs(pred_nueva - demanda_real)
        diff_error = error_nueva - error_antigua
        
        print(f"{modelo:<15} {pred_antigua:>15,.0f} {error_antigua:>12,.0f} "
              f"{pred_nueva:>15,.0f} {error_nueva:>12,.0f} {diff_error:>12,.0f}")
    
    print("\n💡 Interpretación:")
    print("   Si error aumentó → Modelo tenía información del futuro (leakage)")
    print("   Si error igual → Modelo realmente aprende el patrón")


def graficar_comparacion_7dias(df_antiguas, df_nuevas):
    """Genera gráfica comparando últimos 7 días antes/después"""
    # Últimos 7 días
    df_antigua_7d = df_antiguas.tail(168).copy()
    df_nueva_7d = df_nuevas.tail(168).copy()
    
    cols_real = [c for c in df_antiguas.columns if 'demanda_real' in c.lower()][0]
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Comparación Antes/Después Corrección Data Leakage\nÚltimos 7 Días Test Set',
                 fontsize=16, fontweight='bold', y=0.995)
    
    modelos = ['XGBoost', 'RandomForest', 'LightGBM']
    colores = ['#3498DB', '#E74C3C', '#27AE60']
    
    for idx, (modelo, color) in enumerate(zip(modelos, colores)):
        ax = axes[idx]
        
        # Buscar columnas
        cols_antiguas = [c for c in df_antiguas.columns if modelo.lower() in c.lower()]
        cols_nuevas = [c for c in df_nuevas.columns if modelo.lower() in c.lower()]
        
        if not cols_antiguas or not cols_nuevas:
            continue
        
        # Graficar
        ax.plot(df_antigua_7d['timestamp'], df_antigua_7d[cols_real],
                'k-', linewidth=2.5, label='Demanda Real', zorder=3)
        ax.plot(df_antigua_7d['timestamp'], df_antigua_7d[cols_antiguas[0]],
                '--', color=color, linewidth=2, alpha=0.7, 
                label=f'{modelo} CON leakage', zorder=2)
        ax.plot(df_nueva_7d['timestamp'], df_nueva_7d[cols_nuevas[0]],
                '-', color=color, linewidth=2,
                label=f'{modelo} SIN leakage', zorder=1)
        
        ax.set_ylabel('Demanda (m³/hr)', fontsize=11, fontweight='bold')
        ax.set_title(f'{modelo}', fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:00'))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_dir = Path('outputs/metricas_ML')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / 'comparacion_antes_despues_correccion.png',
                dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'comparacion_antes_despues_correccion.pdf',
                bbox_inches='tight')
    
    print(f"\n✅ Gráfica guardada en {output_dir}")
    
    plt.close()


def main():
    """Ejecuta análisis comparativo completo"""
    print("\n" + "="*80)
    print("ANÁLISIS COMPARATIVO: IMPACTO DE CORRECCIÓN DATA LEAKAGE")
    print("="*80)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Cargar datos
    print("\n📂 Cargando predicciones...")
    df_antiguas = cargar_predicciones_antiguas()
    df_nuevas = cargar_predicciones_nuevas()
    
    if df_antiguas is None:
        print("\n⚠️ NOTA: Ejecuta primero la interfaz con el modelo ANTIGUO (CON leakage)")
        print("   y guarda las predicciones como 'predicciones_reales_CON_LEAKAGE.csv'")
        print("   Luego reentrenar con el modelo CORREGIDO y ejecutar este script.")
        return
    
    if df_nuevas is None:
        print("\n❌ Ejecuta la interfaz con el modelo CORREGIDO primero")
        return
    
    print(f"   ✅ Antiguas (CON leakage): {len(df_antiguas)} registros")
    print(f"   ✅ Nuevas (SIN leakage): {len(df_nuevas)} registros")
    
    # Comparar métricas
    df_comp = comparar_metricas(df_antiguas, df_nuevas)
    
    # Analizar peak anómalo
    analizar_peak_anomalo(df_antiguas, df_nuevas)
    
    # Graficar comparación
    print("\n📊 Generando gráficas comparativas...")
    graficar_comparacion_7dias(df_antiguas, df_nuevas)
    
    # Guardar tabla de comparación
    output_dir = Path('outputs/metricas_ML')
    output_dir.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(output_dir / 'comparacion_metricas_antes_despues.csv', index=False)
    print(f"\n✅ Tabla de comparación guardada en {output_dir}")
    
    # Resumen ejecutivo
    print("\n" + "="*80)
    print("RESUMEN EJECUTIVO")
    print("="*80)
    print("\n📌 Data Leakage Detectado:")
    print("   - BD_Qin_m3_Local.csv incluía 2,328 registros del test set")
    print("   - Perfil horario de Qin contaminado con datos futuros")
    print("   - Afectaba a los 3 modelos (XGBoost, RF, LightGBM)")
    
    print("\n📊 Impacto en Métricas:")
    r2_promedio_caida = df_comp['diff_r2'].mean()
    mae_promedio_aumento = df_comp['diff_mae'].mean()
    
    if r2_promedio_caida < 0:
        print(f"   - R² disminuyó en promedio: {abs(r2_promedio_caida):.4f}")
        print(f"   - MAE aumentó en promedio: {mae_promedio_aumento:,.0f} m³/hr")
        print("   ✅ CONFIRMA que había data leakage artificial mejorando métricas")
    else:
        print(f"   - R² se mantuvo o mejoró: {r2_promedio_caida:.4f}")
        print("   ⚠️ Revisar si la corrección se aplicó correctamente")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
