"""
Informe Detallado de Outliers en Demanda

Analiza los 339 registros filtrados por estar fuera del rango 0-30,000 m³/hr
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("="*80)
print("INFORME DETALLADO: OUTLIERS EN DEMANDA DE AGUA")
print("="*80)

# ==================== 1. CARGAR DATOS ====================
print("\n[1/6] Cargando datos...")

# Cargar datos con demanda
data_path = Path('data/processed/data_processed_demanda_valid.csv')
df = pd.read_csv(data_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])

print(f"✅ Datos cargados: {len(df)} registros")
print(f"   Periodo: {df['timestamp_utc'].min()} a {df['timestamp_utc'].max()}")

# Cargar clima para enriquecer el análisis
clima_path = Path('data/raw/BD_Clima2024a202509_UTC.csv')
df_clima = pd.read_csv(clima_path)
df_clima['timestamp'] = pd.to_datetime(df_clima['timestamp'])
df_clima.set_index('timestamp', inplace=True)

# Merge con clima
df = df.join(df_clima[['temp', 'HR', 'mmhr']], on='timestamp_utc', how='left')

print(f"✅ Datos enriquecidos con clima")

# ==================== 2. IDENTIFICAR OUTLIERS ====================
print("\n[2/6] Identificando outliers...")

# Criterio: fuera del rango 0 a 30,000 m³/hr
rango_min = 0
rango_max = 30000

# Registros normales
df_normal = df[
    (df['Demanda_m3_hr'] >= rango_min) &
    (df['Demanda_m3_hr'] <= rango_max)
].copy()

# Outliers
df_outliers = df[
    (df['Demanda_m3_hr'] < rango_min) |
    (df['Demanda_m3_hr'] > rango_max)
].copy()

print(f"\n📊 Clasificación:")
print(f"   Registros normales: {len(df_normal):,} ({len(df_normal)/len(df)*100:.2f}%)")
print(f"   Outliers:           {len(df_outliers):,} ({len(df_outliers)/len(df)*100:.2f}%)")

# ==================== 3. ANÁLISIS ESTADÍSTICO ====================
print("\n[3/6] Análisis estadístico de outliers...")

print("\n" + "="*80)
print("ESTADÍSTICAS DE OUTLIERS")
print("="*80)

print(f"\n📈 Demanda en outliers:")
print(f"   Mínimo:    {df_outliers['Demanda_m3_hr'].min():>15,.2f} m³/hr")
print(f"   Máximo:    {df_outliers['Demanda_m3_hr'].max():>15,.2f} m³/hr")
print(f"   Promedio:  {df_outliers['Demanda_m3_hr'].mean():>15,.2f} m³/hr")
print(f"   Mediana:   {df_outliers['Demanda_m3_hr'].median():>15,.2f} m³/hr")
print(f"   Std Dev:   {df_outliers['Demanda_m3_hr'].std():>15,.2f} m³/hr")

# Clasificar outliers
outliers_negativos = df_outliers[df_outliers['Demanda_m3_hr'] < 0]
outliers_positivos = df_outliers[df_outliers['Demanda_m3_hr'] > rango_max]

print(f"\n🔴 Outliers NEGATIVOS (Demanda < 0):")
print(f"   Cantidad:  {len(outliers_negativos):,}")
if len(outliers_negativos) > 0:
    print(f"   Mínimo:    {outliers_negativos['Demanda_m3_hr'].min():,.2f} m³/hr")
    print(f"   Promedio:  {outliers_negativos['Demanda_m3_hr'].mean():,.2f} m³/hr")

print(f"\n🔴 Outliers POSITIVOS (Demanda > {rango_max:,}):")
print(f"   Cantidad:  {len(outliers_positivos):,}")
if len(outliers_positivos) > 0:
    print(f"   Máximo:    {outliers_positivos['Demanda_m3_hr'].max():,.2f} m³/hr")
    print(f"   Promedio:  {outliers_positivos['Demanda_m3_hr'].mean():,.2f} m³/hr")

# ==================== 4. ANÁLISIS TEMPORAL ====================
print("\n[4/6] Análisis temporal de outliers...")

print("\n" + "="*80)
print("DISTRIBUCIÓN TEMPORAL DE OUTLIERS")
print("="*80)

# Agregar columnas temporales
df_outliers['fecha'] = df_outliers['timestamp_utc'].dt.date
df_outliers['hora'] = df_outliers['timestamp_utc'].dt.hour
df_outliers['mes'] = df_outliers['timestamp_utc'].dt.month
df_outliers['dia_semana'] = df_outliers['timestamp_utc'].dt.day_name()

# Por mes
print(f"\n📅 Outliers por mes:")
outliers_por_mes = df_outliers.groupby('mes').size().sort_index()
for mes, count in outliers_por_mes.items():
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
             'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    print(f"   {meses[mes-1]:3} : {count:>4} outliers ({count/len(df_outliers)*100:>5.1f}%)")

# Por hora
print(f"\n🕐 Outliers por hora del día:")
outliers_por_hora = df_outliers.groupby('hora').size().sort_values(ascending=False)
for hora, count in outliers_por_hora.head(10).items():
    print(f"   Hora {hora:>2}:00 : {count:>3} outliers ({count/len(df_outliers)*100:>5.1f}%)")

# Por día de la semana
print(f"\n📆 Outliers por día de la semana:")
outliers_por_dia = df_outliers.groupby('dia_semana').size()
dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
for dia_en, dia_es in zip(dias_orden, dias_es):
    if dia_en in outliers_por_dia.index:
        count = outliers_por_dia[dia_en]
        print(f"   {dia_es:10} : {count:>3} outliers ({count/len(df_outliers)*100:>5.1f}%)")

# Fechas con más outliers
print(f"\n📅 Fechas con más outliers:")
outliers_por_fecha = df_outliers.groupby('fecha').size().sort_values(ascending=False)
for fecha, count in outliers_por_fecha.head(10).items():
    print(f"   {fecha} : {count:>2} outliers")

# ==================== 5. ANÁLISIS DE CAUSAS ====================
print("\n[5/6] Análisis de posibles causas...")

print("\n" + "="*80)
print("ANÁLISIS DE VARIABLES RELACIONADAS")
print("="*80)

# Qin (caudal de entrada)
print(f"\n🚰 Caudal de Entrada (Qin):")
print(f"   Normal - Promedio:  {df_normal['Qin'].mean():>10,.2f} m³/hr")
print(f"   Outliers - Promedio: {df_outliers['Qin'].mean():>10,.2f} m³/hr")
print(f"   Outliers - Mínimo:   {df_outliers['Qin'].min():>10,.2f} m³/hr")
print(f"   Outliers - Máximo:   {df_outliers['Qin'].max():>10,.2f} m³/hr")

# Delta Volumen
print(f"\n📊 Variación de Volumen (ΔV):")
print(f"   Normal - Promedio:  {df_normal['delta_volumen_m3_hr'].mean():>10,.2f} m³/hr")
print(f"   Outliers - Promedio: {df_outliers['delta_volumen_m3_hr'].mean():>10,.2f} m³/hr")
print(f"   Outliers - Mínimo:   {df_outliers['delta_volumen_m3_hr'].min():>10,.2f} m³/hr")
print(f"   Outliers - Máximo:   {df_outliers['delta_volumen_m3_hr'].max():>10,.2f} m³/hr")

# Temperatura
if 'temp' in df_outliers.columns:
    print(f"\n🌡️  Temperatura:")
    print(f"   Normal - Promedio:  {df_normal['temp'].mean():>6.1f}°C")
    print(f"   Outliers - Promedio: {df_outliers['temp'].mean():>6.1f}°C")
    
    # ¿Hay correlación con eventos climáticos?
    outliers_con_lluvia = df_outliers[df_outliers['mmhr'] > 0]
    print(f"\n🌧️  Precipitación durante outliers:")
    print(f"   Con lluvia: {len(outliers_con_lluvia)} ({len(outliers_con_lluvia)/len(df_outliers)*100:.1f}%)")
    if len(outliers_con_lluvia) > 0:
        print(f"   Lluvia promedio: {outliers_con_lluvia['mmhr'].mean():.2f} mm/hr")

# ==================== 6. LISTADO DETALLADO ====================
print("\n[6/6] Generando listado detallado...")

print("\n" + "="*80)
print("LISTADO COMPLETO DE OUTLIERS")
print("="*80)

# Ordenar por valor absoluto de demanda (más extremos primero)
df_outliers_sorted = df_outliers.copy()
df_outliers_sorted['abs_demanda'] = df_outliers_sorted['Demanda_m3_hr'].abs()
df_outliers_sorted = df_outliers_sorted.sort_values('abs_demanda', ascending=False)

print(f"\n🔝 Top 20 outliers más extremos:\n")
print(f"{'#':<4} {'Fecha/Hora':<20} {'Demanda (m³/hr)':>20} {'Qin':>12} {'ΔV':>12} {'Temp':>7}")
print("-" * 80)

for i, (idx, row) in enumerate(df_outliers_sorted.head(20).iterrows(), 1):
    fecha_hora = row['timestamp_utc'].strftime('%Y-%m-%d %H:%M')
    demanda = row['Demanda_m3_hr']
    qin = row['Qin']
    delta_v = row['delta_volumen_m3_hr']
    temp = row['temp'] if pd.notna(row['temp']) else 0
    
    print(f"{i:<4} {fecha_hora:<20} {demanda:>20,.0f} {qin:>12,.0f} {delta_v:>12,.0f} {temp:>6.1f}°C")

# ==================== 7. GUARDAR REPORTE ====================
print("\n" + "="*80)
print("GUARDANDO REPORTES")
print("="*80)

# Crear directorio de outputs
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

# Guardar CSV completo de outliers
outliers_csv_path = output_dir / 'outliers_demanda_completo.csv'
df_outliers_sorted.to_csv(outliers_csv_path, index=False)
print(f"\n✅ CSV guardado: {outliers_csv_path}")
print(f"   Registros: {len(df_outliers_sorted)}")

# Guardar reporte de texto
report_path = output_dir / 'reporte_outliers_demanda.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("REPORTE: OUTLIERS EN DEMANDA DE AGUA POTABLE\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total registros: {len(df):,}\n")
    f.write(f"Outliers identificados: {len(df_outliers):,} ({len(df_outliers)/len(df)*100:.2f}%)\n")
    f.write(f"Criterio: Demanda fuera del rango [0, 30,000] m³/hr\n\n")
    
    f.write("ESTADÍSTICAS DE OUTLIERS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Demanda mínima:   {df_outliers['Demanda_m3_hr'].min():>15,.2f} m³/hr\n")
    f.write(f"Demanda máxima:   {df_outliers['Demanda_m3_hr'].max():>15,.2f} m³/hr\n")
    f.write(f"Demanda promedio: {df_outliers['Demanda_m3_hr'].mean():>15,.2f} m³/hr\n")
    f.write(f"Demanda mediana:  {df_outliers['Demanda_m3_hr'].median():>15,.2f} m³/hr\n\n")
    
    f.write("CLASIFICACIÓN\n")
    f.write("-" * 80 + "\n")
    f.write(f"Outliers negativos (< 0):        {len(outliers_negativos):>5} ({len(outliers_negativos)/len(df_outliers)*100:.1f}%)\n")
    f.write(f"Outliers positivos (> 30,000):   {len(outliers_positivos):>5} ({len(outliers_positivos)/len(df_outliers)*100:.1f}%)\n\n")
    
    f.write("TOP 50 OUTLIERS MÁS EXTREMOS\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'#':<4} {'Fecha/Hora':<20} {'Demanda (m³/hr)':>20} {'Qin':>12} {'ΔV':>12} {'Temp':>7}\n")
    f.write("-" * 80 + "\n")
    
    for i, (idx, row) in enumerate(df_outliers_sorted.head(50).iterrows(), 1):
        fecha_hora = row['timestamp_utc'].strftime('%Y-%m-%d %H:%M')
        demanda = row['Demanda_m3_hr']
        qin = row['Qin']
        delta_v = row['delta_volumen_m3_hr']
        temp = row['temp'] if pd.notna(row['temp']) else 0
        
        f.write(f"{i:<4} {fecha_hora:<20} {demanda:>20,.0f} {qin:>12,.0f} {delta_v:>12,.0f} {temp:>6.1f}°C\n")

print(f"✅ Reporte guardado: {report_path}")

# ==================== 8. VISUALIZACIONES ====================
print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Análisis de Outliers en Demanda de Agua', fontsize=16, fontweight='bold')

# 1. Distribución de outliers
ax1 = axes[0, 0]
ax1.hist(df_outliers['Demanda_m3_hr'], bins=50, color='red', alpha=0.7, edgecolor='black')
ax1.axvline(0, color='black', linestyle='--', linewidth=2, label='Límite inferior (0)')
ax1.axvline(rango_max, color='blue', linestyle='--', linewidth=2, label=f'Límite superior ({rango_max:,})')
ax1.set_xlabel('Demanda (m³/hr)', fontsize=11)
ax1.set_ylabel('Frecuencia', fontsize=11)
ax1.set_title(f'Distribución de {len(df_outliers)} Outliers', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Outliers por mes
ax2 = axes[0, 1]
outliers_por_mes.plot(kind='bar', ax=ax2, color='orange', edgecolor='black')
ax2.set_xlabel('Mes', fontsize=11)
ax2.set_ylabel('Cantidad de Outliers', fontsize=11)
ax2.set_title('Outliers por Mes', fontsize=12, fontweight='bold')
ax2.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'][:len(outliers_por_mes)], rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Outliers por hora
ax3 = axes[0, 2]
outliers_por_hora_sorted = df_outliers.groupby('hora').size().sort_index()
ax3.plot(outliers_por_hora_sorted.index, outliers_por_hora_sorted.values, 
         marker='o', linewidth=2, markersize=8, color='red')
ax3.set_xlabel('Hora del día', fontsize=11)
ax3.set_ylabel('Cantidad de Outliers', fontsize=11)
ax3.set_title('Outliers por Hora del Día', fontsize=12, fontweight='bold')
ax3.set_xticks(range(0, 24, 2))
ax3.grid(True, alpha=0.3)

# 4. Serie temporal de outliers
ax4 = axes[1, 0]
df_outliers_sorted_time = df_outliers.sort_values('timestamp_utc')
ax4.scatter(df_outliers_sorted_time['timestamp_utc'], 
           df_outliers_sorted_time['Demanda_m3_hr'],
           c=df_outliers_sorted_time['Demanda_m3_hr'],
           cmap='RdYlBu_r', s=30, alpha=0.6, edgecolors='black', linewidths=0.5)
ax4.axhline(0, color='black', linestyle='-', linewidth=1)
ax4.axhline(rango_max, color='blue', linestyle='--', linewidth=1)
ax4.set_xlabel('Fecha', fontsize=11)
ax4.set_ylabel('Demanda (m³/hr)', fontsize=11)
ax4.set_title('Serie Temporal de Outliers', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

# 5. Qin vs Demanda en outliers
ax5 = axes[1, 1]
scatter = ax5.scatter(df_outliers['Qin'], df_outliers['Demanda_m3_hr'],
                     c=df_outliers['delta_volumen_m3_hr'], cmap='viridis',
                     s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
ax5.set_xlabel('Qin (m³/hr)', fontsize=11)
ax5.set_ylabel('Demanda (m³/hr)', fontsize=11)
ax5.set_title('Qin vs Demanda en Outliers', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax5, label='ΔVolumen (m³/hr)')

# 6. Temperatura vs Demanda en outliers
ax6 = axes[1, 2]
if 'temp' in df_outliers.columns and df_outliers['temp'].notna().sum() > 0:
    scatter2 = ax6.scatter(df_outliers['temp'], df_outliers['Demanda_m3_hr'],
                          c=df_outliers['mmhr'], cmap='Blues',
                          s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    ax6.set_xlabel('Temperatura (°C)', fontsize=11)
    ax6.set_ylabel('Demanda (m³/hr)', fontsize=11)
    ax6.set_title('Temperatura vs Demanda en Outliers', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax6, label='Precipitación (mm/hr)')
else:
    ax6.text(0.5, 0.5, 'Datos de temperatura no disponibles',
            ha='center', va='center', fontsize=12)
    ax6.set_title('Temperatura vs Demanda', fontsize=12, fontweight='bold')

plt.tight_layout()

# Guardar visualización
viz_path = output_dir / 'analisis_outliers_demanda.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✅ Visualización guardada: {viz_path}")

print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)

print(f"""
📊 OUTLIERS IDENTIFICADOS: {len(df_outliers):,} de {len(df):,} registros ({len(df_outliers)/len(df)*100:.2f}%)

🔴 Más extremos:
   • Demanda más baja:  {df_outliers['Demanda_m3_hr'].min():,.0f} m³/hr
   • Demanda más alta:  {df_outliers['Demanda_m3_hr'].max():,.0f} m³/hr

📅 Distribución temporal:
   • Mes con más outliers: {outliers_por_mes.idxmax()} ({outliers_por_mes.max()} outliers)
   • Hora con más outliers: {outliers_por_hora.idxmax()}:00 ({outliers_por_hora.max()} outliers)

💾 Archivos generados:
   • {outliers_csv_path}
   • {report_path}
   • {viz_path}

💡 RECOMENDACIÓN:
   Estos outliers fueron correctamente filtrados en el entrenamiento del modelo
   para evitar que distorsionen las predicciones. Representan eventos atípicos
   del sistema (posibles errores de medición, eventos extraordinarios, etc.)
""")

print("="*80)
