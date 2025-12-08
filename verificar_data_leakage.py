"""
Verificación de Data Leakage - Train/Test Split
================================================
Verifica que no haya contaminación entre sets de entrenamiento y prueba.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def verificar_split():
    """Verifica el split train/val/test"""
    
    print("\n" + "="*70)
    print("VERIFICACIÓN DE DATA LEAKAGE - TRAIN/TEST SPLIT")
    print("="*70 + "\n")
    
    # Cargar dataset completo
    df_path = Path('data/processed/dataset_features_completo.csv')
    df = pd.read_csv(df_path)
    
    print(f"📊 Dataset completo: {len(df)} registros")
    
    # Calcular índices de split
    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)
    
    print(f"\n📍 Índices de split:")
    print(f"   Train: 0 → {train_end-1} ({train_end} registros)")
    print(f"   Val:   {train_end} → {val_end-1} ({val_end - train_end} registros)")
    print(f"   Test:  {val_end} → {len(df)-1} ({len(df) - val_end} registros)")
    
    # Convertir timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\n📅 Fechas por split:")
    print(f"   Train: {df.iloc[0]['timestamp']} → {df.iloc[train_end-1]['timestamp']}")
    print(f"   Val:   {df.iloc[train_end]['timestamp']} → {df.iloc[val_end-1]['timestamp']}")
    print(f"   Test:  {df.iloc[val_end]['timestamp']} → {df.iloc[-1]['timestamp']}")
    
    # Verificar predicciones exportadas
    pred_path = Path('outputs/metricas_ML/predicciones_reales.csv')
    if pred_path.exists():
        df_pred = pd.read_csv(pred_path)
        df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
        
        print(f"\n📈 Predicciones exportadas:")
        print(f"   N° registros: {len(df_pred)}")
        print(f"   Desde: {df_pred['timestamp'].iloc[0]}")
        print(f"   Hasta: {df_pred['timestamp'].iloc[-1]}")
        
        # CRÍTICO: Verificar si fechas de predicciones están en train
        fecha_inicio_test = pd.Timestamp(df.iloc[val_end]['timestamp'])
        fecha_fin_train = pd.Timestamp(df.iloc[train_end-1]['timestamp'])
        
        # Quitar timezone de predicciones para comparación
        df_pred['timestamp_simple'] = pd.to_datetime(df_pred['timestamp']).dt.tz_localize(None)
        fecha_inicio_test_simple = fecha_inicio_test.tz_localize(None) if hasattr(fecha_inicio_test, 'tz_localize') else fecha_inicio_test
        
        print(f"\n🔍 VERIFICACIÓN DE CONTAMINACIÓN:")
        print(f"   Fecha inicio test esperado: {fecha_inicio_test_simple}")
        print(f"   Fecha fin entrenamiento: {fecha_fin_train}")
        
        # Verificar overlap
        pred_en_train = df_pred[df_pred['timestamp_simple'] < fecha_inicio_test_simple]
        
        if len(pred_en_train) > 0:
            print(f"\n❌ ¡DATA LEAKAGE DETECTADO!")
            print(f"   {len(pred_en_train)} predicciones están en periodo de entrenamiento")
            print(f"\n   Primeros 10 registros contaminados:")
            print(pred_en_train[['timestamp', 'demanda_real_m3']].head(10))
        else:
            print(f"\n✅ No se detectó data leakage")
            print(f"   Todas las predicciones están en periodo test correcto")
        
        # Verificar últimos 7 días
        df_7dias = df_pred.tail(168)
        print(f"\n📊 Últimos 7 días graficados:")
        print(f"   Desde: {df_7dias['timestamp'].iloc[0]}")
        print(f"   Hasta: {df_7dias['timestamp'].iloc[-1]}")
        
        # Verificar si hay peaks anómalos
        demanda_7dias = df_7dias['demanda_real_m3']
        print(f"\n📈 Estadísticas últimos 7 días:")
        print(f"   Media: {demanda_7dias.mean():,.0f} m³/hr")
        print(f"   Máximo: {demanda_7dias.max():,.0f} m³/hr")
        print(f"   Mínimo: {demanda_7dias.min():,.0f} m³/hr")
        print(f"   Desv. Std: {demanda_7dias.std():,.0f} m³/hr")
        
        # Identificar peaks extremos (> 3 std de la media)
        threshold = demanda_7dias.mean() + 3 * demanda_7dias.std()
        peaks = df_7dias[df_7dias['demanda_real_m3'] > threshold]
        
        if len(peaks) > 0:
            print(f"\n⚠️ PEAKS ANÓMALOS DETECTADOS (> 3σ):")
            print(f"   {len(peaks)} registros con demanda > {threshold:,.0f} m³/hr")
            print(f"\n   Detalles:")
            for _, row in peaks.iterrows():
                print(f"   {row['timestamp']}: {row['demanda_real_m3']:,.0f} m³/hr")
        
    else:
        print(f"\n⚠️ No se encontró {pred_path}")
        print(f"   Ejecuta primero: python interfaz_planificacion_qin_v1.py")
    
    print("\n" + "="*70)
    
    # Verificar código de entrenamiento
    print("\n🔍 VERIFICANDO CÓDIGO DE ENTRENAMIENTO:")
    print("="*70 + "\n")
    
    interfaz_path = Path('interfaz_planificacion_qin_v1.py')
    if interfaz_path.exists():
        with open(interfaz_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar líneas de split
        if 'int(n * 0.85)' in content:
            print("✅ Split encontrado: int(n * 0.85)")
            print("   Código correcto para 70/15/15 split")
        
        # Verificar si usa .predict() en test
        if 'X_test' in content and 'modelo.predict(X_test)' in content:
            print("✅ Usa modelo.predict(X_test)")
            print("   Predicciones son sobre test set")
        
        # Buscar fit() en todo el dataset (ERROR)
        if 'fit(X, y)' in content and 'fit(X_train' not in content:
            print("\n❌ POSIBLE PROBLEMA:")
            print("   Código usa fit(X, y) en lugar de fit(X_train, y_train)")
            print("   Esto causaría data leakage!")
        
    print("\n" + "="*70)


if __name__ == '__main__':
    verificar_split()
