"""
Análisis de Correlaciones con Features Completas
Target: Q_net_m3h (Q_flujo)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analizar_correlaciones():
    """Analiza correlaciones del dataset completo"""
    print("=" * 100)
    print("ANALISIS DE CORRELACIONES - DATASET COMPLETO")
    print("=" * 100)
    
    # Cargar dataset
    df = pd.read_csv('data/processed/dataset_features_completo.csv')
    print(f"\nDataset: {df.shape}")
    print(f"Registros: {df.shape[0]:,}")
    print(f"Features: {df.shape[1]:,}")
    
    # Identificar target
    target = 'Q_net_m3h'
    
    if target not in df.columns:
        print(f"\nERROR: Target '{target}' no encontrado!")
        return
    
    # Columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nFeatures numericas: {len(numeric_cols)}")
    
    # Calcular correlaciones con target
    print("\nCalculando correlaciones...")
    correlaciones = df[numeric_cols].corr()[target].sort_values(ascending=False)
    
    # Remover autocorrelación
    correlaciones = correlaciones[correlaciones.index != target]
    
    # Guardar todas las correlaciones
    output_all = 'outputs/correlaciones_completas_Q_net.csv'
    corr_df = pd.DataFrame({
        'feature': correlaciones.index,
        'correlacion': correlaciones.values,
        'correlacion_abs': abs(correlaciones.values)
    })
    corr_df = corr_df.sort_values('correlacion_abs', ascending=False)
    corr_df.to_csv(output_all, index=False)
    print(f"Guardado: {output_all}")
    
    # Análisis por categorías
    print("\n" + "=" * 100)
    print("TOP CORRELACIONES POR CATEGORIA")
    print("=" * 100)
    
    categorias = {
        'CLIMA_BASE': lambda c: c.startswith('clima_') and '__' not in c and 'lag' not in c,
        'CLIMA_LAGS': lambda c: 'lag_' in c and c.startswith('clima_'),
        'CLIMA_ROLLING': lambda c: ('win_' in c or 'accum_' in c) and c.startswith('clima_'),
        'CALENDARIO': lambda c: c.startswith('cal_'),
        'HIDRAULICA': lambda c: c.startswith('sist_'),
        'TARGET_DERIVADO': lambda c: c.startswith('Q_net_') and c != target,
        'INTERACCION': lambda c: c.startswith('mix_')
    }
    
    resultados_por_cat = {}
    
    for cat_name, filter_func in categorias.items():
        cat_features = [c for c in correlaciones.index if filter_func(c)]
        
        if len(cat_features) == 0:
            print(f"\n{cat_name}: Sin features")
            continue
        
        cat_corr = correlaciones[cat_features].sort_values(key=abs, ascending=False)
        resultados_por_cat[cat_name] = cat_corr
        
        print(f"\n{cat_name}: {len(cat_features)} features")
        print("-" * 100)
        
        # Top 10
        for i, (feature, corr) in enumerate(cat_corr.head(10).items(), 1):
            direccion = "RECUPERACION" if corr > 0 else "DEMANDA"
            print(f"  {i:2d}. {feature:50s} {corr:+.4f}  ({direccion})")
        
        # Guardar top de cada categoría
        output_cat = f'outputs/top_correlaciones_{cat_name.lower()}.csv'
        cat_df = pd.DataFrame({
            'feature': cat_corr.index,
            'correlacion': cat_corr.values,
            'correlacion_abs': abs(cat_corr.values)
        })
        cat_df.to_csv(output_cat, index=False)
    
    # Top global
    print("\n" + "=" * 100)
    print("TOP 50 CORRELACIONES GLOBALES (por valor absoluto)")
    print("=" * 100)
    
    top_50 = corr_df.head(50)
    
    for i, row in top_50.iterrows():
        feature = row['feature']
        corr = row['correlacion']
        direccion = "RECUPERACION" if corr > 0 else "DEMANDA"
        
        # Identificar categoría
        cat = "OTRA"
        for cat_name, filter_func in categorias.items():
            if filter_func(feature):
                cat = cat_name
                break
        
        print(f"{i+1:3d}. [{cat:18s}] {feature:45s} {corr:+.4f}  ({direccion})")
    
    # Guardar top 50
    output_top50 = 'outputs/top_50_features_Q_net.csv'
    top_50.to_csv(output_top50, index=False)
    print(f"\nGuardado: {output_top50}")
    
    # Estadísticas por categoría
    print("\n" + "=" * 100)
    print("ESTADISTICAS POR CATEGORIA")
    print("=" * 100)
    
    for cat_name, cat_corr in resultados_por_cat.items():
        if len(cat_corr) == 0:
            continue
        
        abs_corr = abs(cat_corr)
        print(f"\n{cat_name}:")
        print(f"  Features: {len(cat_corr)}")
        print(f"  Correlacion abs media:   {abs_corr.mean():.4f}")
        print(f"  Correlacion abs mediana: {abs_corr.median():.4f}")
        print(f"  Correlacion abs max:     {abs_corr.max():.4f}")
        print(f"  Features |corr| > 0.1:   {(abs_corr > 0.1).sum()}")
        print(f"  Features |corr| > 0.2:   {(abs_corr > 0.2).sum()}")
        print(f"  Features |corr| > 0.3:   {(abs_corr > 0.3).sum()}")
    
    # Identificar features más prometedoras
    print("\n" + "=" * 100)
    print("FEATURES MAS PROMETEDORAS (|corr| > 0.2)")
    print("=" * 100)
    
    features_prometedoras = corr_df[corr_df['correlacion_abs'] > 0.2]
    print(f"\nTotal: {len(features_prometedoras)} features")
    
    for cat_name, filter_func in categorias.items():
        cat_prom = [f for f in features_prometedoras['feature'] if filter_func(f)]
        if len(cat_prom) > 0:
            print(f"  {cat_name}: {len(cat_prom)}")
    
    output_prom = 'outputs/features_prometedoras_Q_net.csv'
    features_prometedoras.to_csv(output_prom, index=False)
    print(f"\nGuardado: {output_prom}")
    
    return df, correlaciones


if __name__ == '__main__':
    df, corr = analizar_correlaciones()
    
    print("\n" + "=" * 100)
    print("COMPLETADO!")
    print("=" * 100)
    print("\nArchivos generados:")
    print("  - outputs/correlaciones_completas_Q_net.csv")
    print("  - outputs/top_50_features_Q_net.csv")
    print("  - outputs/features_prometedoras_Q_net.csv")
    print("  - outputs/top_correlaciones_[categoria].csv (por cada categoría)")
