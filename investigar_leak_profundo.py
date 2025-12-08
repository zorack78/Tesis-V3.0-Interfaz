"""
Investigación Profunda de Data Leakage
======================================
Analiza si hay información del futuro en las features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def investigar_features():
    """Investiga si hay features con lookahead bias"""
    
    print("\n" + "="*70)
    print("INVESTIGACIÓN PROFUNDA - POSIBLE DATA LEAKAGE")
    print("="*70 + "\n")
    
    # 1. Cargar features usadas
    features_path = Path('models/forecasting/features.txt')
    if features_path.exists():
        with open(features_path, 'r') as f:
            features = [line.strip() for line in f.readlines()]
        
        print(f"📋 Features usadas en modelo: {len(features)}")
        
        # Buscar features sospechosas
        print(f"\n🔍 Features potencialmente problemáticas:")
        
        sospechosas = []
        for feat in features:
            # Features que podrían tener lookahead
            if any(x in feat.lower() for x in ['vol_total', 'volumen_total', 
                                                 'qin', 'q_in', 'produccion']):
                sospechosas.append(feat)
        
        if sospechosas:
            print(f"\n⚠️ {len(sospechosas)} features SOSPECHOSAS encontradas:")
            for feat in sospechosas[:10]:  # Mostrar primeras 10
                print(f"   • {feat}")
            if len(sospechosas) > 10:
                print(f"   ... y {len(sospechosas) - 10} más")
        else:
            print("   ✅ No se encontraron features obvias con lookahead")
    
    # 2. Cargar dataset y analizar
    df_path = Path('data/processed/dataset_features_completo.csv')
    df = pd.read_csv(df_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 3. Analizar el periodo del peak
    peak_fecha = pd.Timestamp('2025-09-25 08:00:00')
    
    # Obtener registro del peak
    df_peak = df[df['timestamp'] == peak_fecha]
    
    if len(df_peak) > 0:
        print(f"\n📍 ANÁLISIS DEL PEAK ANÓMALO:")
        print(f"   Fecha: {peak_fecha}")
        print(f"   Q_net: {df_peak.iloc[0]['Q_net_m3h']:,.0f} m³/hr")
        
        # Calcular demanda (Qout = Qin - Q_net)
        # Si hay feature 'Qin' o 'Vol_Total', eso sería el problema
        if 'Qin' in df.columns:
            qin_peak = df_peak.iloc[0]['Qin']
            qout_peak = qin_peak - df_peak.iloc[0]['Q_net_m3h']
            print(f"   Qin: {qin_peak:,.0f} m³/hr")
            print(f"   Qout (demanda): {qout_peak:,.0f} m³/hr")
            
            print(f"\n❌ PROBLEMA CRÍTICO DETECTADO:")
            print(f"   'Qin' está en las features!")
            print(f"   Qin = Qout + Q_net (contiene info de Qout)")
            print(f"   Esto es DATA LEAKAGE - el modelo ve la respuesta!")
    
    # 4. Verificar target vs features
    print(f"\n🎯 TARGET vs FEATURES:")
    print(f"   Target: Q_net_m3h")
    
    # Verificar si Qin está en features
    if features_path.exists():
        tiene_qin = any('qin' in f.lower() for f in features)
        tiene_vol = any('vol_total' in f.lower() for f in features)
        
        if tiene_qin:
            print(f"\n❌ LEAK CONFIRMADO: Qin está en features")
            print(f"   Problema: Qin = Demanda + Q_net")
            print(f"   El modelo predice Q_net conociendo Qin")
            print(f"   Puede calcular: Q_net = Qin - Demanda")
            print(f"   Por eso 'predice' bien los peaks!")
        
        if tiene_vol:
            print(f"\n⚠️ POSIBLE LEAK: Vol_Total está en features")
            print(f"   Vol_Total se calcula con Qin del mismo instante")
            print(f"   Podría contener información contemporánea")
    
    # 5. Comparar predicción vs real en el peak
    pred_path = Path('outputs/metricas_ML/predicciones_reales.csv')
    if pred_path.exists():
        df_pred = pd.read_csv(pred_path)
        df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
        
        df_pred_peak = df_pred[df_pred['timestamp'] == peak_fecha]
        
        if len(df_pred_peak) > 0:
            print(f"\n📊 PREDICCIONES EN EL PEAK:")
            demanda_real = df_pred_peak.iloc[0]['demanda_real_m3']
            
            cols = df_pred.columns.tolist()
            col_xgb = [c for c in cols if 'xgboost' in c.lower()][0]
            col_rf = [c for c in cols if 'randomforest' in c.lower()][0]
            col_lgb = [c for c in cols if 'lightgbm' in c.lower()][0]
            
            pred_xgb = df_pred_peak.iloc[0][col_xgb]
            pred_rf = df_pred_peak.iloc[0][col_rf]
            pred_lgb = df_pred_peak.iloc[0][col_lgb]
            
            print(f"   Demanda real:  {demanda_real:,.0f} m³/hr")
            print(f"   XGBoost:       {pred_xgb:,.0f} m³/hr (error: {abs(pred_xgb - demanda_real):,.0f})")
            print(f"   RandomForest:  {pred_rf:,.0f} m³/hr (error: {abs(pred_rf - demanda_real):,.0f})")
            print(f"   LightGBM:      {pred_lgb:,.0f} m³/hr (error: {abs(pred_lgb - demanda_real):,.0f})")
            
            # Si todos predicen cerca del peak, hay leak
            errores = [abs(pred_xgb - demanda_real), 
                      abs(pred_rf - demanda_real),
                      abs(pred_lgb - demanda_real)]
            error_promedio = np.mean(errores)
            
            if error_promedio < 2000:  # Error < 2000 en peak de 29k es sospechoso
                print(f"\n❌ MUY SOSPECHOSO:")
                print(f"   Error promedio: {error_promedio:,.0f} m³/hr")
                print(f"   En un peak anómalo de 29k, error < 2k es improbable")
                print(f"   sin información del futuro")
    
    # 6. Verificar cómo se calcula Q_net en el dataset
    print(f"\n🔬 ESTRUCTURA DE DATOS:")
    print(f"   Columnas principales:")
    for col in ['timestamp', 'Q_net_m3h', 'Qin', 'Demanda', 
                'Vol_Total_X_Hr_m3']:
        if col in df.columns:
            print(f"   ✓ {col}")
        else:
            print(f"   ✗ {col} (no existe)")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    investigar_features()
