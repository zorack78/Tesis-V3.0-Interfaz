# -*- coding: utf-8 -*-
"""
Analisis 6: Patrones Temporales de Demanda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 80)
print("ANALISIS 6: PATRONES TEMPORALES DE DEMANDA")
print("=" * 80)

plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path('outputs/descriptivo')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cargar datos
print("\n[1] Cargando datos...")
df = pd.read_csv('../data/processed/dataset_features_completo.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"   OK - Cargado: {len(df):,} registros")

# Calcular demanda (Qout = Qin - Q_net)
df['Demanda_m3h'] = df['sist_Qin_m3h'] - df['Q_net_m3h']
print(f"   Demanda calculada: media={df['Demanda_m3h'].mean():.1f} m3/hr")

# 2. Extraer componentes temporales
print("\n[2] Extrayendo componentes temporales...")
df['hora'] = df['timestamp'].dt.hour
df['dia_semana'] = df['timestamp'].dt.dayofweek
df['dia_nombre'] = df['timestamp'].dt.day_name()
df['mes'] = df['timestamp'].dt.month
df['mes_nombre'] = df['timestamp'].dt.month_name()
df['fecha'] = df['timestamp'].dt.date

# Crear figura con 4 subplots
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

# ============================================================================
# Subplot 1: Demanda promedio por hora del dia
# ============================================================================
print("\n[3] Analizando patrones horarios...")
ax1 = fig.add_subplot(gs[0, 0])

demanda_hora = df.groupby('hora')['Demanda_m3h'].agg(['mean', 'std']).reset_index()

ax1.plot(demanda_hora['hora'], demanda_hora['mean'], 
         marker='o', linewidth=2, markersize=6, color='steelblue', label='Media')
ax1.fill_between(demanda_hora['hora'], 
                  demanda_hora['mean'] - demanda_hora['std'],
                  demanda_hora['mean'] + demanda_hora['std'],
                  alpha=0.3, color='steelblue', label='+/- 1 std')

ax1.set_xlabel('Hora del Día', fontsize=11, fontweight='bold')
ax1.set_ylabel('Demanda (m3/hr)', fontsize=11, fontweight='bold')
ax1.set_title('Patron de Demanda por Hora del Dia', fontsize=13, fontweight='bold')
ax1.set_xticks(range(0, 24, 2))
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Marcar horas pico
hora_max = demanda_hora.loc[demanda_hora['mean'].idxmax(), 'hora']
hora_min = demanda_hora.loc[demanda_hora['mean'].idxmin(), 'hora']
ax1.axvline(hora_max, color='red', linestyle='--', alpha=0.5, label=f'Pico: {int(hora_max)}:00')
ax1.axvline(hora_min, color='green', linestyle='--', alpha=0.5, label=f'Valle: {int(hora_min)}:00')

print(f"   Hora pico: {int(hora_max)}:00 hrs ({demanda_hora['mean'].max():.1f} m³/hr)")
print(f"   Hora valle: {int(hora_min)}:00 hrs ({demanda_hora['mean'].min():.1f} m³/hr)")

# ============================================================================
# Subplot 2: Demanda por dia de la semana
# ============================================================================
print("\n[4] Analizando patrones semanales...")
ax2 = fig.add_subplot(gs[0, 1])

dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

demanda_dia = df.groupby('dia_nombre')['Demanda_m3h'].agg(['mean', 'std']).reindex(dias_orden).reset_index()
demanda_dia['dia_es'] = dias_es

colores = ['steelblue'] * 5 + ['orange'] * 2  # Semana azul, fin de semana naranja

bars = ax2.bar(demanda_dia['dia_es'], demanda_dia['mean'], 
               color=colores, alpha=0.7, edgecolor='black')
ax2.errorbar(demanda_dia['dia_es'], demanda_dia['mean'], 
             yerr=demanda_dia['std'], fmt='none', color='black', capsize=5)

ax2.set_xlabel('Día de la Semana', fontsize=11, fontweight='bold')
ax2.set_ylabel('Demanda (m3/hr)', fontsize=11, fontweight='bold')
ax2.set_title('Demanda Promedio por Dia de la Semana', fontsize=13, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# Calcular diferencia semana vs fin de semana
demanda_semana = df[df['dia_semana'] < 5]['Demanda_m3h'].mean()
demanda_finde = df[df['dia_semana'] >= 5]['Demanda_m3h'].mean()
print(f"   Promedio semana: {demanda_semana:.1f} m³/hr")
print(f"   Promedio fin de semana: {demanda_finde:.1f} m³/hr")
print(f"   Diferencia: {((demanda_finde - demanda_semana) / demanda_semana * 100):.1f}%")

# ============================================================================
# Subplot 3: Demanda por mes (boxplot)
# ============================================================================
print("\n[5] Analizando patrones mensuales...")
ax3 = fig.add_subplot(gs[1, 0])

meses_orden = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
meses_es = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
            'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

# Preparar datos para boxplot
df_meses = df[df['mes_nombre'].isin(meses_orden)].copy()
df_meses['mes_es'] = df_meses['mes_nombre'].map(dict(zip(meses_orden, meses_es)))

# Ordenar por mes
mes_order = [m for m in meses_es if m in df_meses['mes_es'].unique()]

bp = ax3.boxplot([df_meses[df_meses['mes_es'] == mes]['Demanda_m3h'].values 
                   for mes in mes_order],
                  labels=mes_order, patch_artist=True)

# Colorear boxplots (verano=rojo, invierno=azul)
colores_meses = ['steelblue', 'steelblue', 'lightblue', 'lightgreen', 'lightgreen', 'yellow',
                 'orange', 'orange', 'lightgreen', 'lightgreen', 'lightblue', 'steelblue']
for patch, color in zip(bp['boxes'], colores_meses[:len(bp['boxes'])]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.set_xlabel('Mes', fontsize=11, fontweight='bold')
ax3.set_ylabel('Demanda (m3/hr)', fontsize=11, fontweight='bold')
ax3.set_title('Distribucion de Demanda por Mes', fontsize=13, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Subplot 4: Tendencia temporal (promedio movil 7 dias)
# ============================================================================
print("\n[6] Calculando tendencia temporal...")
ax4 = fig.add_subplot(gs[1, 1])

# Demanda diaria promedio
demanda_diaria = df.groupby('fecha')['Demanda_m3h'].mean().reset_index()
demanda_diaria['fecha'] = pd.to_datetime(demanda_diaria['fecha'])

# Promedio movil 7 dias
demanda_diaria['ma_7d'] = demanda_diaria['Demanda_m3h'].rolling(window=7, center=True).mean()

ax4.plot(demanda_diaria['fecha'], demanda_diaria['Demanda_m3h'], 
         alpha=0.3, color='gray', linewidth=0.5, label='Diario')
ax4.plot(demanda_diaria['fecha'], demanda_diaria['ma_7d'], 
         color='darkblue', linewidth=2, label='Media Móvil 7 días')

ax4.set_xlabel('Fecha', fontsize=11, fontweight='bold')
ax4.set_ylabel('Demanda (m3/hr)', fontsize=11, fontweight='bold')
ax4.set_title('Tendencia Temporal de Demanda', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

# Rotar etiquetas de fecha
ax4.tick_params(axis='x', rotation=45)

# Titulo principal
fig.suptitle('Analisis de Patrones Temporales - Demanda de Agua Potable',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(OUTPUT_DIR / '02_patrones_temporales.png', dpi=300, bbox_inches='tight')
print(f"\n   OK - Guardado: {OUTPUT_DIR / '02_patrones_temporales.png'}")
plt.close()

# 7. Guardar datos agregados
print("\n[7] Guardando datos agregados...")
demanda_hora.to_csv(OUTPUT_DIR / '02_demanda_por_hora.csv', index=False, float_format='%.2f')
demanda_dia.to_csv(OUTPUT_DIR / '02_demanda_por_dia.csv', index=False, float_format='%.2f')
print(f"   OK - Datos guardados")

print("\n" + "=" * 80)
print("ANALISIS COMPLETADO")
print("=" * 80)
