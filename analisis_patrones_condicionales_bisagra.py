"""
Análisis de Patrones Condicionales y Valores Bisagra
Busca combinaciones Calendar + Clima + Qin + Q_flujo
Identifica puntos de inflexión y umbrales críticos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def cargar_datos():
    """Carga dataset completo"""
    df = pd.read_csv('data/processed/dataset_features_completo.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def analisis_umbrales_temperatura(df):
    """Identifica umbrales de temperatura donde cambia el comportamiento"""
    print("=" * 100)
    print("ANALISIS DE UMBRALES DE TEMPERATURA")
    print("=" * 100)
    
    # Crear bins de temperatura
    temp_bins = np.arange(df['clima_temp_c'].min(), df['clima_temp_c'].max() + 2, 2)
    df['temp_bin'] = pd.cut(df['clima_temp_c'], bins=temp_bins)
    
    # Analizar Q_net por bin de temperatura
    temp_analysis = df.groupby('temp_bin').agg({
        'Q_net_m3h': ['mean', 'median', 'std', 'count'],
        'sist_Qin_m3h': ['mean', 'median']
    }).round(2)
    
    print("\nDemanda promedio por rango de temperatura:")
    print(temp_analysis)
    
    # Identificar umbrales críticos (cambios >20%)
    medias = df.groupby('temp_bin')['Q_net_m3h'].mean()
    cambios_pct = medias.pct_change() * 100
    
    print("\n" + "-" * 100)
    print("UMBRALES CRITICOS (cambios >20%):")
    print("-" * 100)
    umbrales_criticos = cambios_pct[abs(cambios_pct) > 20]
    if len(umbrales_criticos) > 0:
        for temp_range, cambio in umbrales_criticos.items():
            print(f"  {temp_range}: {cambio:+.1f}% cambio")
    else:
        print("  No se encontraron cambios >20% entre bins consecutivos")
    
    # Cuartiles
    q25 = df['clima_temp_c'].quantile(0.25)
    q50 = df['clima_temp_c'].quantile(0.50)
    q75 = df['clima_temp_c'].quantile(0.75)
    
    print("\n" + "-" * 100)
    print("CUARTILES DE TEMPERATURA Y DEMANDA:")
    print("-" * 100)
    
    cuartiles_temp = [
        ('Frío (Q1)', df['clima_temp_c'] <= q25),
        ('Templado bajo (Q2)', (df['clima_temp_c'] > q25) & (df['clima_temp_c'] <= q50)),
        ('Templado alto (Q3)', (df['clima_temp_c'] > q50) & (df['clima_temp_c'] <= q75)),
        ('Calor (Q4)', df['clima_temp_c'] > q75)
    ]
    
    for nombre, mask in cuartiles_temp:
        temp_media = df[mask]['clima_temp_c'].mean()
        q_net_media = df[mask]['Q_net_m3h'].mean()
        qin_media = df[mask]['sist_Qin_m3h'].mean()
        n_registros = mask.sum()
        print(f"  {nombre:20s} T={temp_media:5.1f}°C  Q_net={q_net_media:7.1f}  Qin={qin_media:7.1f}  n={n_registros:,}")
    
    return temp_bins, cuartiles_temp


def analisis_patrones_dia_semana_hora(df):
    """Analiza patrones por día de semana y hora"""
    print("\n" + "=" * 100)
    print("PATRONES DIA SEMANA × HORA")
    print("=" * 100)
    
    # Extraer día semana y hora desde timestamp
    df['dow'] = df['timestamp'].dt.dayofweek  # 0=Lunes, 6=Domingo
    df['hour'] = df['timestamp'].dt.hour
    
    # Pivot table: día × hora
    pivot_demanda = df.pivot_table(
        values='Q_net_m3h',
        index='hour',
        columns='dow',
        aggfunc='mean'
    )
    
    pivot_qin = df.pivot_table(
        values='sist_Qin_m3h',
        index='hour',
        columns='dow',
        aggfunc='mean'
    )
    
    dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    pivot_demanda.columns = dias
    pivot_qin.columns = dias
    
    print("\nDemanda promedio Q_net (m3/hr) por hora y día:")
    print(pivot_demanda.round(0))
    
    print("\n\nQin promedio (m3/hr) por hora y día:")
    print(pivot_qin.round(0))
    
    # Identificar horas bisagra (cambios >30%)
    print("\n" + "-" * 100)
    print("HORAS BISAGRA (cambios >30% entre horas consecutivas):")
    print("-" * 100)
    
    for dia, col in enumerate(dias):
        cambios = pivot_demanda[col].pct_change() * 100
        bisagras = cambios[abs(cambios) > 30]
        if len(bisagras) > 0:
            print(f"\n  {col}:")
            for hora, cambio in bisagras.items():
                hora_ant = hora - 1 if hora > 0 else 23
                print(f"    {hora_ant:02d}h → {hora:02d}h: {cambio:+.1f}%")
    
    return pivot_demanda, pivot_qin


def analisis_eventos_especiales_con_clima(df):
    """Analiza eventos especiales combinados con condiciones climáticas"""
    print("\n" + "=" * 100)
    print("EVENTOS ESPECIALES × CLIMA × DIA SEMANA")
    print("=" * 100)
    
    # Definir eventos
    eventos = {
        'cal_anio_nuevo': 'Año Nuevo',
        'cal_fiestas_patrias': 'Fiestas Patrias',
        'cal_festival_vina': 'Festival Viña',
        'cal_feriado': 'Feriado (cualquiera)',
        'cal_es_fin_de_semana': 'Fin de Semana',
        'cal_vacaciones_escolares': 'Vacaciones Escolares'
    }
    
    # Definir condiciones climáticas
    q25_temp = df['clima_temp_c'].quantile(0.25)
    q75_temp = df['clima_temp_c'].quantile(0.75)
    
    condiciones_clima = {
        'Frío': df['clima_temp_c'] <= q25_temp,
        'Normal': (df['clima_temp_c'] > q25_temp) & (df['clima_temp_c'] <= q75_temp),
        'Calor': df['clima_temp_c'] > q75_temp
    }
    
    resultados = []
    
    for evento_col, evento_nombre in eventos.items():
        if evento_col not in df.columns:
            continue
        
        evento_mask = df[evento_col] == 1
        n_eventos = evento_mask.sum()
        
        if n_eventos == 0:
            continue
        
        print(f"\n{evento_nombre} (n={n_eventos}):")
        print("-" * 100)
        
        for clima_nombre, clima_mask in condiciones_clima.items():
            combinado = evento_mask & clima_mask
            n_comb = combinado.sum()
            
            if n_comb < 5:  # Mínimo 5 observaciones
                continue
            
            q_net_media = df[combinado]['Q_net_m3h'].mean()
            qin_media = df[combinado]['sist_Qin_m3h'].mean()
            temp_media = df[combinado]['clima_temp_c'].mean()
            
            # Comparar con baseline (mismo clima sin evento)
            baseline = (~evento_mask) & clima_mask
            q_net_baseline = df[baseline]['Q_net_m3h'].mean()
            qin_baseline = df[baseline]['sist_Qin_m3h'].mean()
            
            delta_q_net = q_net_media - q_net_baseline
            delta_qin = qin_media - qin_baseline
            
            print(f"  {clima_nombre:10s} (n={n_comb:4d}): "
                  f"T={temp_media:5.1f}°C  "
                  f"Q_net={q_net_media:7.1f} (Δ{delta_q_net:+7.1f})  "
                  f"Qin={qin_media:7.1f} (Δ{delta_qin:+7.1f})")
            
            resultados.append({
                'evento': evento_nombre,
                'clima': clima_nombre,
                'n': n_comb,
                'temp_media': temp_media,
                'Q_net_media': q_net_media,
                'Qin_media': qin_media,
                'delta_Q_net': delta_q_net,
                'delta_Qin': delta_qin
            })
    
    # Guardar resultados
    resultados_df = pd.DataFrame(resultados)
    output_path = 'outputs/patrones_eventos_clima.csv'
    resultados_df.to_csv(output_path, index=False)
    print(f"\n\nGuardado: {output_path}")
    
    return resultados_df


def analisis_umbrales_qin_vs_temperatura(df):
    """Analiza relación Qin × Temperatura para identificar umbrales"""
    print("\n" + "=" * 100)
    print("UMBRALES Qin × TEMPERATURA")
    print("=" * 100)
    
    # Bins de temperatura
    temp_bins = pd.qcut(df['clima_temp_c'], q=5, labels=['Muy Frío', 'Frío', 'Normal', 'Cálido', 'Muy Cálido'])
    df['temp_categoria'] = temp_bins
    
    # Bins de Qin
    qin_bins = pd.qcut(df['sist_Qin_m3h'], q=5, labels=['Muy Bajo', 'Bajo', 'Normal', 'Alto', 'Muy Alto'])
    df['qin_categoria'] = qin_bins
    
    # Tabla cruzada
    pivot = df.pivot_table(
        values='Q_net_m3h',
        index='temp_categoria',
        columns='qin_categoria',
        aggfunc='mean'
    )
    
    print("\nDemanda Q_net promedio (m3/hr) por Temperatura × Qin:")
    print(pivot.round(0))
    
    # Conteo de casos
    print("\n\nNúmero de observaciones por Temperatura × Qin:")
    pivot_count = df.pivot_table(
        values='Q_net_m3h',
        index='temp_categoria',
        columns='qin_categoria',
        aggfunc='count'
    )
    print(pivot_count)
    
    # Identificar combinaciones extremas
    print("\n" + "-" * 100)
    print("COMBINACIONES EXTREMAS (demanda >1.5 std):")
    print("-" * 100)
    
    mean_q = df['Q_net_m3h'].mean()
    std_q = df['Q_net_m3h'].std()
    umbral_alto = mean_q + 1.5 * std_q
    umbral_bajo = mean_q - 1.5 * std_q
    
    for temp_cat in temp_bins.cat.categories:
        for qin_cat in qin_bins.cat.categories:
            mask = (df['temp_categoria'] == temp_cat) & (df['qin_categoria'] == qin_cat)
            if mask.sum() < 10:
                continue
            
            q_net_mean = df[mask]['Q_net_m3h'].mean()
            
            if q_net_mean > umbral_alto or q_net_mean < umbral_bajo:
                n_casos = mask.sum()
                extremo = "ALTA" if q_net_mean > umbral_alto else "BAJA"
                print(f"  Temp={temp_cat:12s} × Qin={qin_cat:12s}: Q_net={q_net_mean:7.1f} ({extremo}, n={n_casos})")


def analisis_cambios_temperatura_vs_demanda(df):
    """Analiza cómo cambios de temperatura afectan demanda"""
    print("\n" + "=" * 100)
    print("CAMBIOS DE TEMPERATURA × CAMBIOS DE DEMANDA")
    print("=" * 100)
    
    # Usar deltas ya calculados
    delta_cols = ['clima_temp_delta_1h', 'clima_temp_delta_3h', 'clima_temp_delta_6h', 'clima_temp_delta_12h', 'clima_temp_delta_24h']
    
    for delta_col in delta_cols:
        if delta_col not in df.columns:
            continue
        
        horizon = delta_col.split('_')[-1]
        
        # Categorizar cambios de temperatura
        q25 = df[delta_col].quantile(0.25)
        q75 = df[delta_col].quantile(0.75)
        
        categorias = [
            ('Enfriamiento fuerte', df[delta_col] <= q25),
            ('Estable', (df[delta_col] > q25) & (df[delta_col] <= q75)),
            ('Calentamiento fuerte', df[delta_col] > q75)
        ]
        
        print(f"\n{horizon.upper()}:")
        print("-" * 100)
        
        for nombre, mask in categorias:
            if mask.sum() < 10:
                continue
            
            delta_temp_mean = df[mask][delta_col].mean()
            q_net_mean = df[mask]['Q_net_m3h'].mean()
            qin_mean = df[mask]['sist_Qin_m3h'].mean()
            n = mask.sum()
            
            print(f"  {nombre:25s} (ΔT={delta_temp_mean:+5.1f}°C): "
                  f"Q_net={q_net_mean:7.1f}  Qin={qin_mean:7.1f}  n={n:,}")


def generar_reporte_final(df):
    """Genera reporte resumen con hallazgos clave"""
    print("\n" + "=" * 100)
    print("REPORTE FINAL - VALORES BISAGRA Y UMBRALES CRITICOS")
    print("=" * 100)
    
    reporte = []
    
    # 1. Umbrales de temperatura
    q25_temp = df['clima_temp_c'].quantile(0.25)
    q75_temp = df['clima_temp_c'].quantile(0.75)
    
    reporte.append({
        'tipo': 'Umbral Temperatura',
        'variable': 'clima_temp_c',
        'valor': f'{q25_temp:.1f}°C (Q25)',
        'descripcion': 'Umbral frío - Demanda típica en temperaturas bajas'
    })
    
    reporte.append({
        'tipo': 'Umbral Temperatura',
        'variable': 'clima_temp_c',
        'valor': f'{q75_temp:.1f}°C (Q75)',
        'descripcion': 'Umbral calor - Demanda típica en temperaturas altas'
    })
    
    # 2. Horas críticas
    hora_demanda = df.groupby(df['timestamp'].dt.hour)['Q_net_m3h'].mean()
    hora_max_demanda = hora_demanda.idxmin()  # Más negativo = más demanda
    hora_max_recup = hora_demanda.idxmax()  # Más positivo = más recuperación
    
    reporte.append({
        'tipo': 'Hora Crítica',
        'variable': 'hora',
        'valor': f'{hora_max_demanda}h',
        'descripcion': f'Hora de máxima demanda: {hora_demanda[hora_max_demanda]:.0f} m3/hr'
    })
    
    reporte.append({
        'tipo': 'Hora Crítica',
        'variable': 'hora',
        'valor': f'{hora_max_recup}h',
        'descripcion': f'Hora de máxima recuperación: {hora_demanda[hora_max_recup]:.0f} m3/hr'
    })
    
    # 3. Umbrales Qin
    q25_qin = df['sist_Qin_m3h'].quantile(0.25)
    q75_qin = df['sist_Qin_m3h'].quantile(0.75)
    
    reporte.append({
        'tipo': 'Umbral Qin',
        'variable': 'sist_Qin_m3h',
        'valor': f'{q25_qin:.0f} m3/hr (Q25)',
        'descripcion': 'Qin bajo - Producción reducida'
    })
    
    reporte.append({
        'tipo': 'Umbral Qin',
        'variable': 'sist_Qin_m3h',
        'valor': f'{q75_qin:.0f} m3/hr (Q75)',
        'descripcion': 'Qin alto - Producción elevada'
    })
    
    # Guardar reporte
    reporte_df = pd.DataFrame(reporte)
    output_path = 'outputs/valores_bisagra_umbrales.csv'
    reporte_df.to_csv(output_path, index=False)
    
    print("\nVALORES BISAGRA IDENTIFICADOS:")
    print("-" * 100)
    for item in reporte:
        print(f"  [{item['tipo']:20s}] {item['variable']:20s} = {item['valor']:20s}")
        print(f"    → {item['descripcion']}")
    
    print(f"\n\nGuardado: {output_path}")


def main():
    """Pipeline principal"""
    print("=" * 100)
    print("ANALISIS DE PATRONES CONDICIONALES Y VALORES BISAGRA")
    print("=" * 100)
    print("\nObjetivo: Identificar umbrales críticos y combinaciones")
    print("Variables: Calendar + Clima + Qin + Q_flujo (las 4 fuentes)")
    
    # Cargar datos
    df = cargar_datos()
    print(f"\nDataset: {df.shape}")
    
    # Análisis
    analisis_umbrales_temperatura(df)
    analisis_patrones_dia_semana_hora(df)
    analisis_eventos_especiales_con_clima(df)
    analisis_umbrales_qin_vs_temperatura(df)
    analisis_cambios_temperatura_vs_demanda(df)
    generar_reporte_final(df)
    
    print("\n" + "=" * 100)
    print("COMPLETADO!")
    print("=" * 100)
    print("\nArchivos generados:")
    print("  - outputs/patrones_eventos_clima.csv")
    print("  - outputs/valores_bisagra_umbrales.csv")


if __name__ == '__main__':
    main()
