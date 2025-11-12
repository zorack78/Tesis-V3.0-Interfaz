"""
Visualización del Impacto Climático en Demanda de Agua
Gráficos interactivos que muestran cómo eventos extremos afectan el consumo
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración estética
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def cargar_datos():
    """Carga dataset con features climáticas avanzadas"""
    data_path = Path("data/processed/data_processed_complete_with_climate_advanced.csv")
    print(f"📂 Cargando: {data_path}")
    
    df = pd.read_csv(data_path)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    
    print(f"✅ Datos cargados: {len(df):,} registros")
    print(f"   Features: {len(df.columns)}")
    
    return df

def grafico_temperatura_vs_demanda(df):
    """Muestra relación entre temperatura y demanda"""
    print("\n📊 Creando: Temperatura vs Demanda...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Impacto de Temperatura en Demanda de Agua', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Scatter plot temperatura vs volumen
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['temp'], df[' Volumen_Total_m3'], 
                          c=df['HR'], cmap='viridis', alpha=0.3, s=1)
    ax1.set_xlabel('Temperatura (°C)')
    ax1.set_ylabel('Volumen (m³/hr)')
    ax1.set_title('Temperatura vs Demanda (color = Humedad)')
    plt.colorbar(scatter, ax=ax1, label='Humedad Relativa (%)')
    
    # 2. Boxplot por rangos de temperatura
    ax2 = axes[0, 1]
    temp_bins = pd.cut(df['temp'], bins=[-10, 8, 12, 15, 22, 26, 35], 
                       labels=['<8°C\nMuy Fría', '8-12°C\nFría', 
                               '12-15°C\nFresca', '15-22°C\nNormal',
                               '22-26°C\nCálida', '>26°C\nMuy Cálida'])
    df['temp_rango'] = temp_bins
    df.boxplot(column=' Volumen_Total_m3', by='temp_rango', ax=ax2)
    ax2.set_xlabel('Rango de Temperatura')
    ax2.set_ylabel('Volumen (m³/hr)')
    ax2.set_title('Distribución de Demanda por Temperatura')
    plt.sca(ax2)
    plt.xticks(rotation=45)
    
    # 3. Serie temporal con olas de calor
    ax3 = axes[1, 0]
    # Filtrar un mes de verano
    verano = df[(df['timestamp_utc'].dt.month.isin([1, 2])) & 
                (df['timestamp_utc'].dt.year == 2024)].copy()
    ax3.plot(verano['timestamp_utc'], verano[' Volumen_Total_m3'], 
             label='Demanda', linewidth=1, color='blue')
    ax3_temp = ax3.twinx()
    ax3_temp.plot(verano['timestamp_utc'], verano['temp'], 
                  label='Temperatura', linewidth=1, color='red', alpha=0.7)
    
    # Marcar olas de calor
    olas = verano[verano['en_ola_calor'] == 1]
    if len(olas) > 0:
        ax3.scatter(olas['timestamp_utc'], olas[' Volumen_Total_m3'],
                   color='orange', s=20, zorder=5, label='Ola de calor')
    
    ax3.set_xlabel('Fecha')
    ax3.set_ylabel('Volumen (m³/hr)', color='blue')
    ax3_temp.set_ylabel('Temperatura (°C)', color='red')
    ax3.set_title('Verano 2024: Demanda vs Temperatura')
    ax3.legend(loc='upper left')
    ax3_temp.legend(loc='upper right')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Impacto de cambios bruscos de temperatura
    ax4 = axes[1, 1]
    frentes = df[df['frente_frio_6h'] == 1].copy()
    normales = df[df['frente_frio_6h'] == 0].sample(min(5000, len(frentes))).copy()
    
    ax4.hist(normales[' Volumen_Total_m3'], bins=50, alpha=0.5, 
             label=f'Normal (n={len(normales):,})', color='blue', density=True)
    ax4.hist(frentes[' Volumen_Total_m3'], bins=50, alpha=0.5,
             label=f'Frente Frío (n={len(frentes):,})', color='red', density=True)
    ax4.axvline(normales[' Volumen_Total_m3'].mean(), color='blue', 
                linestyle='--', linewidth=2, label='Media Normal')
    ax4.axvline(frentes[' Volumen_Total_m3'].mean(), color='red',
                linestyle='--', linewidth=2, label='Media Frente Frío')
    ax4.set_xlabel('Volumen (m³/hr)')
    ax4.set_ylabel('Densidad')
    ax4.set_title(f'Impacto de Frentes Fríos en Demanda\n' + 
                  f'Diferencia: {frentes[" Volumen_Total_m3"].mean() - normales[" Volumen_Total_m3"].mean():+,.0f} m³/hr')
    ax4.legend()
    
    plt.tight_layout()
    output_file = Path("outputs/analisis_climatico/impacto_temperatura.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    plt.close()

def grafico_precipitacion_vs_demanda(df):
    """Muestra impacto de precipitación"""
    print("\n📊 Creando: Precipitación vs Demanda...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Impacto de Precipitación en Demanda de Agua',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Comparación con/sin lluvia
    ax1 = axes[0, 0]
    sin_lluvia = df[df['sin_lluvia'] == 1][' Volumen_Total_m3']
    con_lluvia = df[df['sin_lluvia'] == 0][' Volumen_Total_m3']
    
    data_plot = [sin_lluvia, con_lluvia]
    labels_plot = [f'Sin lluvia\n(n={len(sin_lluvia):,})',
                   f'Con lluvia\n(n={len(con_lluvia):,})']
    ax1.boxplot(data_plot, labels=labels_plot)
    ax1.set_ylabel('Volumen (m³/hr)')
    ax1.set_title('Demanda: Sin Lluvia vs Con Lluvia')
    ax1.grid(True, alpha=0.3)
    
    # 2. Por intensidad de lluvia
    ax2 = axes[0, 1]
    intensidades = {
        'Sin lluvia': df[df['sin_lluvia'] == 1],
        'Llovizna': df[df['lluvia_ligera'] == 1],
        'Lluvia\nmoderada': df[df['lluvia_moderada'] == 1],
        'Lluvia\nfuerte': df[df['lluvia_fuerte'] == 1]
    }
    
    means = [grupo[' Volumen_Total_m3'].mean() for grupo in intensidades.values()]
    stds = [grupo[' Volumen_Total_m3'].std() for grupo in intensidades.values()]
    colors = ['green', 'yellow', 'orange', 'red']
    
    bars = ax2.bar(range(len(intensidades)), means, yerr=stds, 
                   color=colors, alpha=0.7, capsize=5)
    ax2.set_xticks(range(len(intensidades)))
    ax2.set_xticklabels(intensidades.keys())
    ax2.set_ylabel('Volumen Promedio (m³/hr)')
    ax2.set_title('Demanda por Intensidad de Lluvia')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores sobre barras
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax2.text(i, mean + std + 1000, f'{mean:,.0f}', 
                ha='center', fontsize=9)
    
    # 3. Serie temporal con tormentas
    ax3 = axes[1, 0]
    # Filtrar invierno (más lluvia)
    invierno = df[(df['timestamp_utc'].dt.month.isin([6, 7, 8])) & 
                  (df['timestamp_utc'].dt.year == 2024)].copy()
    
    ax3.plot(invierno['timestamp_utc'], invierno[' Volumen_Total_m3'],
             label='Demanda', linewidth=1, color='blue')
    ax3_precip = ax3.twinx()
    ax3_precip.bar(invierno['timestamp_utc'], invierno['mmhr'],
                   label='Precipitación', color='cyan', alpha=0.4, width=0.03)
    
    # Marcar tormentas
    tormentas = invierno[invierno['en_tormenta'] == 1]
    if len(tormentas) > 0:
        ax3.scatter(tormentas['timestamp_utc'], tormentas[' Volumen_Total_m3'],
                   color='red', s=30, zorder=5, label='Tormenta', marker='*')
    
    ax3.set_xlabel('Fecha')
    ax3.set_ylabel('Volumen (m³/hr)', color='blue')
    ax3_precip.set_ylabel('Precipitación (mm/hr)', color='cyan')
    ax3.set_title('Invierno 2024: Demanda vs Precipitación')
    ax3.legend(loc='upper left')
    ax3_precip.legend(loc='upper right')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Acumulado de lluvia vs demanda
    ax4 = axes[1, 1]
    # Filtrar solo casos con lluvia
    con_lluvia_df = df[df['sin_lluvia'] == 0].copy()
    scatter = ax4.scatter(con_lluvia_df['precip_acum_6h'], 
                          con_lluvia_df[' Volumen_Total_m3'],
                          c=con_lluvia_df['temp'], cmap='coolwarm',
                          alpha=0.5, s=10)
    ax4.set_xlabel('Precipitación Acumulada 6h (mm)')
    ax4.set_ylabel('Volumen (m³/hr)')
    ax4.set_title('Acumulado Lluvia 6h vs Demanda (color = Temp)')
    plt.colorbar(scatter, ax=ax4, label='Temperatura (°C)')
    
    plt.tight_layout()
    output_file = Path("outputs/analisis_climatico/impacto_precipitacion.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    plt.close()

def grafico_combinaciones_extremas(df):
    """Muestra impacto de condiciones compuestas"""
    print("\n📊 Creando: Condiciones Climáticas Compuestas...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Impacto de Condiciones Climáticas Compuestas',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Comparación de escenarios
    ax1 = axes[0, 0]
    escenarios = {
        'Calor\nseco': df[df['calor_seco'] == 1],
        'Calor\nhúmedo': df[df['calor_humedo'] == 1],
        'Frío\nseco': df[df['frio_seco'] == 1],
        'Frío\nhúmedo': df[df['frio_humedo'] == 1],
        'Niebla': df[df['niebla'] == 1],
        'Ideales': df[df['condiciones_ideales'] == 1]
    }
    
    means = [grupo[' Volumen_Total_m3'].mean() for grupo in escenarios.values()]
    counts = [len(grupo) for grupo in escenarios.values()]
    colors = ['red', 'orange', 'lightblue', 'blue', 'gray', 'green']
    
    bars = ax1.barh(range(len(escenarios)), means, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(escenarios)))
    ax1.set_yticklabels([f'{k}\n({c:,} casos)' 
                         for k, c in zip(escenarios.keys(), counts)])
    ax1.set_xlabel('Volumen Promedio (m³/hr)')
    ax1.set_title('Demanda por Condición Climática Compuesta')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores
    for i, mean in enumerate(means):
        ax1.text(mean + 1000, i, f'{mean:,.0f}', 
                va='center', fontsize=9)
    
    # 2. Sensación térmica vs demanda
    ax2 = axes[0, 1]
    # Binear sensación térmica
    bins_sensacion = pd.cut(df['sensacion_termica'], bins=10)
    grouped = df.groupby(bins_sensacion)[' Volumen_Total_m3'].agg(['mean', 'count'])
    
    x_vals = range(len(grouped))
    ax2.plot(x_vals, grouped['mean'], marker='o', linewidth=2, markersize=8)
    ax2.set_xticks(x_vals)
    ax2.set_xticklabels([f'{b.left:.0f}-{b.right:.0f}' 
                         for b in grouped.index], rotation=45, ha='right')
    ax2.set_xlabel('Sensación Térmica (°C)')
    ax2.set_ylabel('Volumen Promedio (m³/hr)')
    ax2.set_title('Sensación Térmica vs Demanda\n(Correlación: -0.451)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Mapa de calor temp vs humedad
    ax3 = axes[1, 0]
    # Crear bins
    temp_bins = pd.cut(df['temp'], bins=15)
    hr_bins = pd.cut(df['HR'], bins=15)
    
    # Calcular demanda promedio por combinación
    pivot_data = df.groupby([temp_bins, hr_bins])[' Volumen_Total_m3'].mean().unstack()
    
    sns.heatmap(pivot_data, cmap='RdYlGn_r', ax=ax3, cbar_kws={'label': 'Volumen (m³/hr)'})
    ax3.set_xlabel('Humedad Relativa (%)')
    ax3.set_ylabel('Temperatura (°C)')
    ax3.set_title('Mapa de Calor: Demanda por Temp × Humedad')
    ax3.tick_params(axis='both', labelsize=8)
    
    # 4. Índice de condiciones adversas
    ax4 = axes[1, 1]
    adversas_data = df.groupby('condiciones_adversas')[' Volumen_Total_m3'].agg(['mean', 'count'])
    
    x_adv = adversas_data.index
    y_mean = adversas_data['mean']
    y_count = adversas_data['count']
    
    ax4_count = ax4.twinx()
    
    line1 = ax4.plot(x_adv, y_mean, marker='o', linewidth=2, markersize=10,
                     color='red', label='Demanda promedio')
    bars = ax4_count.bar(x_adv, y_count, alpha=0.3, color='gray', 
                         label='Frecuencia')
    
    ax4.set_xlabel('Índice de Condiciones Adversas')
    ax4.set_ylabel('Volumen Promedio (m³/hr)', color='red')
    ax4_count.set_ylabel('Frecuencia', color='gray')
    ax4.set_title('Impacto del Índice de Condiciones Adversas')
    ax4.tick_params(axis='y', labelcolor='red')
    ax4_count.tick_params(axis='y', labelcolor='gray')
    ax4.grid(True, alpha=0.3)
    
    # Combinar leyendas
    lines = line1
    labels = [l.get_label() for l in lines]
    lines.append(bars)
    labels.append('Frecuencia')
    ax4.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    output_file = Path("outputs/analisis_climatico/condiciones_compuestas.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    plt.close()

def grafico_correlaciones_top(df):
    """Muestra top features por correlación"""
    print("\n📊 Creando: Top Correlaciones...")
    
    # Seleccionar features climáticas
    features_clima = [col for col in df.columns if any(x in col.lower() 
                      for x in ['temp', 'hr', 'precip', 'lluvia', 'frente',
                                'ola', 'calor', 'frio', 'niebla', 'tormenta',
                                'sensacion', 'adversas'])]
    
    # Calcular correlaciones
    correlaciones = df[features_clima + [' Volumen_Total_m3']].corr()[' Volumen_Total_m3'].drop(' Volumen_Total_m3')
    correlaciones_abs = correlaciones.abs().sort_values(ascending=False)
    
    # Top 20
    top_20 = correlaciones[correlaciones_abs.head(20).index]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['red' if x < 0 else 'green' for x in top_20.values]
    bars = ax.barh(range(len(top_20)), top_20.values, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20.index, fontsize=9)
    ax.set_xlabel('Correlación con Demanda de Agua')
    ax.set_title('Top 20 Features Climáticas por Correlación', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores
    for i, val in enumerate(top_20.values):
        ax.text(val + (0.01 if val > 0 else -0.01), i, f'{val:+.3f}',
                va='center', ha='left' if val > 0 else 'right', fontsize=8)
    
    plt.tight_layout()
    output_file = Path("outputs/analisis_climatico/top_correlaciones.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    plt.close()

def main():
    """Ejecuta todas las visualizaciones"""
    print("="*80)
    print("VISUALIZACIÓN IMPACTO CLIMÁTICO EN DEMANDA DE AGUA")
    print("="*80)
    
    # Crear directorio de salida
    Path("outputs/analisis_climatico").mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    df = cargar_datos()
    
    # Generar gráficos
    grafico_temperatura_vs_demanda(df)
    grafico_precipitacion_vs_demanda(df)
    grafico_combinaciones_extremas(df)
    grafico_correlaciones_top(df)
    
    print("\n" + "="*80)
    print("✅ VISUALIZACIÓN COMPLETA")
    print("="*80)
    print("\n📁 Gráficos guardados en: outputs/analisis_climatico/")
    print("   • impacto_temperatura.png")
    print("   • impacto_precipitacion.png")
    print("   • condiciones_compuestas.png")
    print("   • top_correlaciones.png")
    print("\n🎯 Próximo paso: Revisar gráficos y reentrenar modelo")

if __name__ == "__main__":
    main()
