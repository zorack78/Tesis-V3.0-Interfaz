"""
Gráfico de Torta - Top 5 Estanques por Capacidad
================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuración
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

BASE_DIR = 'c:/Users/socce/Downloads/rafa/Tesis3.0-Interfaz'
DATA_DIR = f'{BASE_DIR}/data/raw'
OUTPUT_DIR = f'{BASE_DIR}/outputs'

print("="*80)
print("GRÁFICO DE TORTA - TOP 5 ESTANQUES POR CAPACIDAD")
print("="*80)
print()

# Cargar datos
df = pd.read_csv(f'{DATA_DIR}/BD_Capacidad_89Tks_m3.csv')
df.columns = df.columns.str.strip()
print(f"✓ Datos cargados: {len(df)} estanques")
print(f"  Capacidad total: {df['VOL(m3)'].sum():,.0f} m³")
print()

# Ordenar por capacidad
df_sorted = df.sort_values('VOL(m3)', ascending=False)

# Top 5
top5 = df_sorted.head(5).copy()
otros = df_sorted.iloc[5:]['VOL(m3)'].sum()

print("📊 TOP 5 ESTANQUES:")
for idx, row in top5.iterrows():
    pct = row['VOL(m3)'] / df['VOL(m3)'].sum() * 100
    print(f"   {row['TK']:.<40} {row['VOL(m3)']:>6,.0f} m³ ({pct:>5.2f}%)")
print(f"   {'Otros (84 estanques)':.<40} {otros:>6,.0f} m³ ({otros/df['VOL(m3)'].sum()*100:>5.2f}%)")
print()

# Crear gráfico con diseño moderno
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#F8F9FA')

# Grid layout para composición
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4], 
                       hspace=0.1, wspace=0.3)

# Título superior
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis('off')
ax_title.text(0.5, 0.6, 'Top 5 Estanques por Capacidad de Almacenamiento', 
             ha='center', va='center', fontsize=20, fontweight='bold', 
             color='#2C3E50', family='sans-serif')
ax_title.text(0.5, 0.1, f'Sistema de Distribución de Agua Potable | Total: {df["VOL(m3)"].sum():,.0f} m³ en {len(df)} estanques', 
             ha='center', va='center', fontsize=11, style='italic', color='#7F8C8D')

# Gráfico de torta principal
ax = fig.add_subplot(gs[1, 0])

# Datos para el gráfico
labels = list(top5['TK']) + ['Otros\n(84 estanques)']
sizes = list(top5['VOL(m3)']) + [otros]

# Paleta de colores moderna y elegante
colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#95A5A6']

# Explotar todos los slices ligeramente para efecto moderno
explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.02]

# Crear torta con estilo donut
wedges, texts, autotexts = ax.pie(sizes, 
                                    labels=None,  # Sin labels en el gráfico
                                    autopct='%1.1f%%',
                                    startangle=45,
                                    colors=colors,
                                    explode=explode,
                                    wedgeprops=dict(width=0.6, edgecolor='white', linewidth=3),
                                    textprops=dict(color="white", fontweight='bold', fontsize=12))

# Mejorar apariencia de porcentajes
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

# Círculo central para efecto donut
centre_circle = plt.Circle((0, 0), 0.40, fc='#F8F9FA', linewidth=2, edgecolor='#BDC3C7')
ax.add_artist(centre_circle)

# Texto central
ax.text(0, 0, f'{df["VOL(m3)"].sum():,.0f}\nm³', 
        ha='center', va='center', fontsize=18, fontweight='bold', color='#2C3E50')

ax.set_aspect('equal')

# Panel lateral con información detallada
ax_info = fig.add_subplot(gs[1, 1])
ax_info.axis('off')

y_position = 0.95
for idx, (label, size, color) in enumerate(zip(labels, sizes, colors)):
    # Cuadrado de color
    ax_info.add_patch(plt.Rectangle((0.05, y_position - 0.04), 0.08, 0.08, 
                                     facecolor=color, edgecolor='white', linewidth=2))
    
    # Nombre del estanque (acortar si es muy largo)
    name = label.replace('_', ' ').title()
    if len(name) > 25:
        name = name[:22] + '...'
    
    ax_info.text(0.18, y_position, name, 
                va='center', fontsize=10, fontweight='bold', color='#2C3E50')
    
    # Capacidad y porcentaje
    pct = size / df['VOL(m3)'].sum() * 100
    ax_info.text(0.18, y_position - 0.04, f'{size:,.0f} m³  •  {pct:.1f}%', 
                va='center', fontsize=9, color='#7F8C8D')
    
    y_position -= 0.15

# Línea decorativa
ax_info.plot([0.05, 0.95], [0.08, 0.08], color='#BDC3C7', linewidth=1.5, alpha=0.5)

# Información adicional en la parte inferior
ax_info.text(0.5, 0.02, f'Total: {len(df)} estanques\n{df["VOL(m3)"].sum():,.0f} m³', 
            ha='center', va='bottom', fontsize=9, style='italic', color='#95A5A6')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/grafico_torta_top5_estanques.png', bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/grafico_torta_top5_estanques.pdf', bbox_inches='tight')
plt.close()

print("✅ Gráfico guardado:")
print(f"   • outputs/grafico_torta_top5_estanques.png")
print(f"   • outputs/grafico_torta_top5_estanques.pdf")
print()
print("="*80)
