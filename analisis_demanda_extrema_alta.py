"""
ANÁLISIS DE DEMANDA EXTREMA ALTA - CORRELACIÓN CON OLAS DE CALOR
==================================================================

HIPÓTESIS DEL USUARIO (OPERADOR):
Los outliers de demanda extrema alta (>30,000 m³/hr) NO son errores.
Son eventos REALES que ocurren durante:
   • Olas de calor (alta temperatura + baja humedad)
   • Condiciones climáticas extremas sostenidas
   • Situaciones que estresan la infraestructura

OBJETIVO:
Validar si los 78 outliers positivos corresponden a:
   1. Eventos climáticos extremos (olas de calor, sequía)
   2. Patrones de consumo real extremo
   3. Errores de medición/telemetría

Analiza:
   - Distribución horaria y temporal
   - Correlación con temperatura extrema (>25°C, >28°C)
   - Baja humedad (<40%, <30%)
   - Eventos sostenidos (múltiples horas/días consecutivos)
   - Estacionalidad (verano vs otras estaciones)
   - Comparación con demanda típica por rango térmico
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Configuración visual
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 14)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("ANÁLISIS: DEMANDA EXTREMA ALTA = ¿OLAS DE CALOR REALES?")
print("=" * 80)
print(f"\nFecha análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================

print("\n" + "=" * 80)
print("[1/8] CARGANDO DATOS")
print("=" * 80)

# Outliers clasificados
df_outliers = pd.read_csv('outputs/outliers_clasificados.csv')
df_outliers['timestamp_utc'] = pd.to_datetime(df_outliers['timestamp_utc'], 
                                               utc=True)
df_outliers['fecha_hora_local'] = pd.to_datetime(
    df_outliers['fecha_hora_local'], utc=True)

# Filtrar solo positivos >30k
df_extremos = df_outliers[df_outliers['Demanda_m3_hr'] > 30000].copy()

print(f"✅ Outliers de demanda extrema alta: {len(df_extremos)}")
print(f"   Rango: {df_extremos['Demanda_m3_hr'].min():.0f} a "
      f"{df_extremos['Demanda_m3_hr'].max():.0f} m³/hr")
print(f"   Media: {df_extremos['Demanda_m3_hr'].mean():.0f} m³/hr")

# Datos completos para comparación
df_all = pd.read_csv('data/processed/data_processed_complete.csv')
df_all['timestamp_utc'] = pd.to_datetime(df_all['timestamp_utc'], utc=True)
df_all['fecha_hora_local'] = pd.to_datetime(df_all['fecha_hora_local'], 
                                             utc=True)

print(f"✅ Datos completos: {len(df_all)} registros")

# ============================================================================
# 2. ANÁLISIS TEMPORAL
# ============================================================================

print("\n" + "=" * 80)
print("[2/8] ANÁLISIS TEMPORAL")
print("=" * 80)

# Hora del día
df_extremos['hora_del_dia'] = df_extremos['fecha_hora_local'].dt.hour
dist_horaria = df_extremos['hora_del_dia'].value_counts().sort_index()

print("\n📊 DISTRIBUCIÓN HORARIA:")
print("\nHora    N     %")
print("-" * 30)
for hora, count in dist_horaria.items():
    pct = count / len(df_extremos) * 100
    bar = "█" * int(pct / 2)
    marca = "☀️" if 10 <= hora < 20 else "🌙"
    print(f"{marca} {hora:02d}:00  {count:3d}  {pct:5.1f}%  {bar}")

# Clasificar por período
def clasificar_periodo(hora):
    if 0 <= hora < 6:
        return "Madrugada (00-06)"
    elif 6 <= hora < 12:
        return "Mañana (06-12)"
    elif 12 <= hora < 18:
        return "Tarde (12-18)"
    else:
        return "Noche (18-24)"


df_extremos['periodo'] = df_extremos['hora_del_dia'].apply(clasificar_periodo)

print("\n📊 DISTRIBUCIÓN POR PERÍODO:")
for periodo in ["Madrugada (00-06)", "Mañana (06-12)", 
                "Tarde (12-18)", "Noche (18-24)"]:
    count = (df_extremos['periodo'] == periodo).sum()
    pct = count / len(df_extremos) * 100
    print(f"   {periodo:<20}: {count:3d} ({pct:5.1f}%)")

# ============================================================================
# 3. ANÁLISIS CLIMÁTICO - TEMPERATURA
# ============================================================================

print("\n" + "=" * 80)
print("[3/8] ANÁLISIS DE TEMPERATURA")
print("=" * 80)

temp_stats = df_extremos['temp'].describe()
print(f"\n📊 ESTADÍSTICAS DE TEMPERATURA:")
print(f"   • Media:    {temp_stats['mean']:.1f}°C")
print(f"   • Mediana:  {temp_stats['50%']:.1f}°C")
print(f"   • Mín:      {temp_stats['min']:.1f}°C")
print(f"   • Máx:      {temp_stats['max']:.1f}°C")
print(f"   • Q75:      {temp_stats['75%']:.1f}°C")

# Clasificar por temperatura
temp_muy_alta = (df_extremos['temp'] >= 25).sum()
temp_extrema = (df_extremos['temp'] >= 28).sum()
temp_ola_calor = (df_extremos['temp'] >= 30).sum()
temp_normal = (df_extremos['temp'] < 25).sum()

print(f"\n🌡️  CLASIFICACIÓN TÉRMICA:")
print(f"   • Normal (<25°C):         {temp_normal:3d} "
      f"({temp_normal/len(df_extremos)*100:.1f}%)")
print(f"   • Muy alta (25-28°C):     {temp_muy_alta-temp_extrema:3d} "
      f"({(temp_muy_alta-temp_extrema)/len(df_extremos)*100:.1f}%)")
print(f"   • Extrema (28-30°C):      {temp_extrema-temp_ola_calor:3d} "
      f"({(temp_extrema-temp_ola_calor)/len(df_extremos)*100:.1f}%)")
print(f"   • Ola de calor (≥30°C):   {temp_ola_calor:3d} "
      f"({temp_ola_calor/len(df_extremos)*100:.1f}%)")

# Comparar con temperatura general
print(f"\n📊 COMPARACIÓN CON DATASET COMPLETO:")
# Cargar clima completo
df_clima = pd.read_csv('data/processed/clima_chile_v3.csv')
df_clima['timestamp'] = pd.to_datetime(df_clima['timestamp'], utc=True)

temp_general_media = df_clima['temp'].mean()
temp_general_p75 = df_clima['temp'].quantile(0.75)
temp_general_p95 = df_clima['temp'].quantile(0.95)

print(f"   Temp media general:        {temp_general_media:.1f}°C")
print(f"   Temp P75 general:          {temp_general_p75:.1f}°C")
print(f"   Temp P95 general:          {temp_general_p95:.1f}°C")
print(f"   Temp media extremos:       {temp_stats['mean']:.1f}°C")
print(f"   Diferencia:                "
      f"{temp_stats['mean'] - temp_general_media:+.1f}°C")

if temp_stats['mean'] > temp_general_p95:
    print(f"   ✅ Temperatura extremos > P95 → CORRELACIÓN FUERTE")
elif temp_stats['mean'] > temp_general_p75:
    print(f"   ⚠️  Temperatura extremos > P75 → Correlación moderada")
else:
    print(f"   ❌ Temperatura extremos NO elevada")

# ============================================================================
# 4. ANÁLISIS DE HUMEDAD
# ============================================================================

print("\n" + "=" * 80)
print("[4/8] ANÁLISIS DE HUMEDAD RELATIVA")
print("=" * 80)

hr_stats = df_extremos['HR'].describe()
print(f"\n📊 ESTADÍSTICAS DE HUMEDAD:")
print(f"   • Media:    {hr_stats['mean']:.1f}%")
print(f"   • Mediana:  {hr_stats['50%']:.1f}%")
print(f"   • Mín:      {hr_stats['min']:.1f}%")
print(f"   • Máx:      {hr_stats['max']:.1f}%")
print(f"   • Q25:      {hr_stats['25%']:.1f}%")

# Clasificar por humedad
hr_muy_baja = (df_extremos['HR'] < 30).sum()
hr_baja = ((df_extremos['HR'] >= 30) & (df_extremos['HR'] < 50)).sum()
hr_normal = ((df_extremos['HR'] >= 50) & (df_extremos['HR'] < 70)).sum()
hr_alta = (df_extremos['HR'] >= 70).sum()

print(f"\n💧 CLASIFICACIÓN DE HUMEDAD:")
print(f"   • Muy baja (<30%):    {hr_muy_baja:3d} "
      f"({hr_muy_baja/len(df_extremos)*100:.1f}%)")
print(f"   • Baja (30-50%):      {hr_baja:3d} "
      f"({hr_baja/len(df_extremos)*100:.1f}%)")
print(f"   • Normal (50-70%):    {hr_normal:3d} "
      f"({hr_normal/len(df_extremos)*100:.1f}%)")
print(f"   • Alta (≥70%):        {hr_alta:3d} "
      f"({hr_alta/len(df_extremos)*100:.1f}%)")

# Condiciones de ola de calor (temp alta + HR baja)
ola_calor = ((df_extremos['temp'] >= 28) & 
             (df_extremos['HR'] < 40)).sum()
print(f"\n🔥 CONDICIONES DE OLA DE CALOR (Temp≥28°C + HR<40%):")
print(f"   {ola_calor}/{len(df_extremos)} casos "
      f"({ola_calor/len(df_extremos)*100:.1f}%)")

# ============================================================================
# 5. EVENTOS SOSTENIDOS - ANÁLISIS TEMPORAL
# ============================================================================

print("\n" + "=" * 80)
print("[5/8] ANÁLISIS DE EVENTOS SOSTENIDOS")
print("=" * 80)

# Ordenar por fecha
df_extremos_sorted = df_extremos.sort_values('fecha_hora_local')

# Detectar eventos consecutivos (diferencia < 2 horas)
df_extremos_sorted['diff_horas'] = df_extremos_sorted[
    'fecha_hora_local'].diff().dt.total_seconds() / 3600

# Crear grupos de eventos consecutivos
df_extremos_sorted['es_consecutivo'] = df_extremos_sorted[
    'diff_horas'] <= 2
df_extremos_sorted['grupo_evento'] = (
    ~df_extremos_sorted['es_consecutivo']).cumsum()

# Analizar grupos
eventos = df_extremos_sorted.groupby('grupo_evento').agg({
    'fecha_hora_local': ['min', 'max', 'count'],
    'temp': 'mean',
    'HR': 'mean',
    'Demanda_m3_hr': 'mean'
})

eventos.columns = ['inicio', 'fin', 'duracion_horas', 
                   'temp_media', 'hr_media', 'demanda_media']
eventos['duracion_dias'] = (eventos['fin'] - eventos[
    'inicio']).dt.total_seconds() / 86400

print(f"\n📊 EVENTOS IDENTIFICADOS: {len(eventos)}")
print(f"\n   Eventos por duración:")
eventos_1h = (eventos['duracion_horas'] == 1).sum()
eventos_2_5h = ((eventos['duracion_horas'] > 1) & 
                (eventos['duracion_horas'] <= 5)).sum()
eventos_6_12h = ((eventos['duracion_horas'] > 5) & 
                 (eventos['duracion_horas'] <= 12)).sum()
eventos_mas_12h = (eventos['duracion_horas'] > 12).sum()

print(f"      • 1 hora (aislados):     {eventos_1h:3d} "
      f"({eventos_1h/len(eventos)*100:.1f}%)")
print(f"      • 2-5 horas:             {eventos_2_5h:3d} "
      f"({eventos_2_5h/len(eventos)*100:.1f}%)")
print(f"      • 6-12 horas:            {eventos_6_12h:3d} "
      f"({eventos_6_12h/len(eventos)*100:.1f}%)")
print(f"      • >12 horas (sostenido): {eventos_mas_12h:3d} "
      f"({eventos_mas_12h/len(eventos)*100:.1f}%)")

# Mostrar eventos más largos
print(f"\n📋 TOP 5 EVENTOS MÁS LARGOS:")
eventos_top = eventos.nlargest(5, 'duracion_horas')
print(f"\n{'Inicio':<20} {'Duración':<15} {'Temp':<8} {'HR':<6} "
      f"{'Demanda Media':<15}")
print("-" * 75)
for idx, row in eventos_top.iterrows():
    print(f"{str(row['inicio'])[:19]:<20} "
          f"{row['duracion_horas']:.0f}h ({row['duracion_dias']:.1f}d)    "
          f"{row['temp_media']:5.1f}°C  {row['hr_media']:4.0f}%  "
          f"{row['demanda_media']:10,.0f} m³/hr")

# ============================================================================
# 6. ANÁLISIS POR ESTACIÓN
# ============================================================================

print("\n" + "=" * 80)
print("[6/8] ANÁLISIS POR ESTACIÓN DEL AÑO")
print("=" * 80)

df_extremos['mes'] = df_extremos['fecha_hora_local'].dt.month


def clasificar_estacion(mes):
    if mes in [12, 1, 2]:
        return "Verano"
    elif mes in [3, 4, 5]:
        return "Otoño"
    elif mes in [6, 7, 8]:
        return "Invierno"
    else:
        return "Primavera"


df_extremos['estacion'] = df_extremos['mes'].apply(clasificar_estacion)
estacion_dist = df_extremos['estacion'].value_counts()

print(f"\n📊 DISTRIBUCIÓN POR ESTACIÓN:")
for estacion in ["Verano", "Otoño", "Invierno", "Primavera"]:
    count = estacion_dist.get(estacion, 0)
    pct = count / len(df_extremos) * 100
    print(f"   {estacion:<12}: {count:3d} ({pct:5.1f}%)")

# Estadísticas por estación
print(f"\n📊 CLIMA Y DEMANDA POR ESTACIÓN:")
print(f"\n{'Estación':<12} {'N':>4} {'Temp':>8} {'HR':>6} "
      f"{'Demanda Media':>15}")
print("-" * 55)
for estacion in ["Verano", "Otoño", "Invierno", "Primavera"]:
    df_est = df_extremos[df_extremos['estacion'] == estacion]
    if len(df_est) > 0:
        print(f"{estacion:<12} {len(df_est):4d} "
              f"{df_est['temp'].mean():6.1f}°C "
              f"{df_est['HR'].mean():4.0f}%  "
              f"{df_est['Demanda_m3_hr'].mean():13,.0f} m³/hr")

# ============================================================================
# 7. ANÁLISIS DE TELEMETRÍA
# ============================================================================

print("\n" + "=" * 80)
print("[7/8] ANÁLISIS DE PROBLEMAS DE TELEMETRÍA")
print("=" * 80)

# Cargar análisis de estanques
try:
    df_estanques_problemas = pd.read_csv(
        'outputs/outliers_con_problemas_estanques.csv')
    
    # Filtrar solo extremos positivos
    extremos_con_problemas = df_estanques_problemas[
        df_estanques_problemas['Demanda_m3_hr'] > 30000]
    
    print(f"\n📊 OUTLIERS EXTREMOS CON PROBLEMAS DE TELEMETRÍA:")
    print(f"   {len(extremos_con_problemas)}/{len(df_extremos)} "
          f"({len(extremos_con_problemas)/len(df_extremos)*100:.1f}%)")
    
    if len(extremos_con_problemas) > 0:
        print(f"\n   ⚠️  Algunos extremos pueden ser errores de telemetría")
        print(f"   ✅ {len(df_extremos) - len(extremos_con_problemas)} "
              f"extremos sin problemas de sensores")
    else:
        print(f"\n   ✅ NINGÚN extremo tiene problemas de telemetría")
        print(f"   → Todos son eventos REALES de demanda alta")
        
except FileNotFoundError:
    print(f"\n   ℹ️  Archivo de problemas de telemetría no encontrado")
    print(f"   → Asumiendo que todos los extremos son válidos")

# ============================================================================
# 8. VISUALIZACIÓN
# ============================================================================

print("\n" + "=" * 80)
print("[8/8] GENERANDO VISUALIZACIONES")
print("=" * 80)

fig, axes = plt.subplots(4, 2, figsize=(16, 18))
fig.suptitle('ANÁLISIS: Demanda Extrema Alta - Correlación con Olas de Calor', 
             fontsize=16, fontweight='bold', y=0.995)

# 1. Distribución horaria
ax = axes[0, 0]
dist_horaria.plot(kind='bar', ax=ax, color='orangered', alpha=0.7)
ax.axvspan(9.5, 19.5, alpha=0.2, color='orange', 
           label='Período diurno (10-20h)')
ax.set_xlabel('Hora del día')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución Horaria de Demandas Extremas')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Temperatura vs Demanda
ax = axes[0, 1]
scatter = ax.scatter(df_extremos['temp'], df_extremos['Demanda_m3_hr'],
                     c=df_extremos['HR'], cmap='RdYlBu_r', 
                     alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
ax.axvline(x=25, color='orange', linestyle='--', linewidth=2, 
           alpha=0.5, label='Calor (25°C)')
ax.axvline(x=28, color='red', linestyle='--', linewidth=2, 
           alpha=0.5, label='Extremo (28°C)')
ax.axvline(x=30, color='darkred', linestyle='--', linewidth=2, 
           alpha=0.5, label='Ola calor (30°C)')
ax.set_xlabel('Temperatura (°C)')
ax.set_ylabel('Demanda (m³/hr)')
ax.set_title('Temperatura vs Demanda Extrema')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Humedad Relativa (%)')

# 3. Humedad vs Demanda
ax = axes[1, 0]
scatter2 = ax.scatter(df_extremos['HR'], df_extremos['Demanda_m3_hr'],
                      c=df_extremos['temp'], cmap='YlOrRd', 
                      alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
ax.axvline(x=30, color='red', linestyle='--', linewidth=2, 
           alpha=0.5, label='HR muy baja (<30%)')
ax.axvline(x=40, color='orange', linestyle='--', linewidth=2, 
           alpha=0.5, label='HR baja (<40%)')
ax.set_xlabel('Humedad Relativa (%)')
ax.set_ylabel('Demanda (m³/hr)')
ax.set_title('Humedad vs Demanda Extrema')
ax.legend()
ax.grid(True, alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=ax)
cbar2.set_label('Temperatura (°C)')

# 4. Distribución por estación
ax = axes[1, 1]
estaciones_orden = ["Verano", "Otoño", "Invierno", "Primavera"]
estacion_counts = [estacion_dist.get(e, 0) for e in estaciones_orden]
colores_estacion = ['#FF6B35', '#FF8C42', '#4A90E2', '#7FBA00']
bars = ax.bar(estaciones_orden, estacion_counts, 
              color=colores_estacion, alpha=0.7, 
              edgecolor='black', linewidth=1.5)
ax.set_xlabel('Estación del año')
ax.set_ylabel('Frecuencia')
ax.set_title('Demandas Extremas por Estación')
ax.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, estacion_counts):
    height = bar.get_height()
    if count > 0:
        pct = count / len(df_extremos) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

# 5. Temp + HR combinados (ola de calor)
ax = axes[2, 0]
colors = []
for _, row in df_extremos.iterrows():
    if row['temp'] >= 28 and row['HR'] < 40:
        colors.append('darkred')  # Ola de calor
    elif row['temp'] >= 25 and row['HR'] < 50:
        colors.append('orange')  # Calor moderado
    else:
        colors.append('steelblue')  # Normal

ax.scatter(df_extremos['temp'], df_extremos['HR'], 
           c=colors, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
ax.axvline(x=28, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.axhline(y=40, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.fill_between([28, 35], [0, 0], [40, 40], 
                alpha=0.2, color='red', label='Zona ola de calor')
ax.set_xlabel('Temperatura (°C)')
ax.set_ylabel('Humedad Relativa (%)')
ax.set_title('Condiciones de Ola de Calor (Temp≥28°C + HR<40%)')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Duración de eventos
ax = axes[2, 1]
duraciones = [eventos_1h, eventos_2_5h, eventos_6_12h, eventos_mas_12h]
labels = ['1h\n(aislado)', '2-5h', '6-12h', '>12h\n(sostenido)']
colors_dur = ['lightblue', 'skyblue', 'orange', 'darkred']
bars2 = ax.bar(labels, duraciones, color=colors_dur, alpha=0.7,
               edgecolor='black', linewidth=1.5)
ax.set_ylabel('Número de eventos')
ax.set_title('Duración de Eventos Extremos')
ax.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars2, duraciones):
    height = bar.get_height()
    if count > 0:
        pct = count / len(eventos) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.0f}%)',
                ha='center', va='bottom', fontweight='bold')

# 7. Serie temporal
ax = axes[3, 0]
df_extremos_sorted_plot = df_extremos.sort_values('fecha_hora_local')
colors_temp = df_extremos_sorted_plot['temp']
scatter3 = ax.scatter(df_extremos_sorted_plot['fecha_hora_local'],
                      df_extremos_sorted_plot['Demanda_m3_hr'],
                      c=colors_temp, cmap='YlOrRd', 
                      alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Fecha')
ax.set_ylabel('Demanda (m³/hr)')
ax.set_title('Serie Temporal de Demandas Extremas')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)
cbar3 = plt.colorbar(scatter3, ax=ax)
cbar3.set_label('Temperatura (°C)')

# 8. Boxplot por clasificación térmica
ax = axes[3, 1]
df_extremos['clase_termica'] = pd.cut(df_extremos['temp'], 
                                       bins=[0, 15, 20, 25, 28, 50],
                                       labels=['<15°C', '15-20°C', 
                                              '20-25°C', '25-28°C', '≥28°C'])
sns.boxplot(data=df_extremos, x='clase_termica', y='Demanda_m3_hr', 
            ax=ax, palette='YlOrRd')
ax.set_xlabel('Rango de Temperatura')
ax.set_ylabel('Demanda (m³/hr)')
ax.set_title('Demanda Extrema por Rango Térmico')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/analisis_demanda_extrema_alta.png', 
            dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado: outputs/analisis_demanda_extrema_alta.png")

# ============================================================================
# 9. CONCLUSIONES
# ============================================================================

print("\n" + "=" * 80)
print("📊 CONCLUSIONES")
print("=" * 80)

# Calcular porcentajes clave
pct_temp_alta = (df_extremos['temp'] >= 25).sum() / len(df_extremos) * 100
pct_ola_calor = ola_calor / len(df_extremos) * 100
pct_verano = estacion_dist.get('Verano', 0) / len(df_extremos) * 100
pct_diurno = ((df_extremos['periodo'] == 'Mañana (06-12)').sum() + 
              (df_extremos['periodo'] == 'Tarde (12-18)').sum()) / len(df_extremos) * 100

print(f"\n1️⃣ CORRELACIÓN CON TEMPERATURA:")
print(f"   • Temperatura media extremos: {temp_stats['mean']:.1f}°C")
print(f"   • Temperatura media general:  {temp_general_media:.1f}°C")
print(f"   • Diferencia: {temp_stats['mean'] - temp_general_media:+.1f}°C")
print(f"   • Con temp ≥25°C: {pct_temp_alta:.1f}%")

if pct_temp_alta > 70:
    print(f"   ✅ CORRELACIÓN MUY FUERTE con calor")
elif pct_temp_alta > 50:
    print(f"   ⚠️  Correlación MODERADA con calor")
else:
    print(f"   ❌ Correlación DÉBIL")

print(f"\n2️⃣ CONDICIONES DE OLA DE CALOR:")
print(f"   • Casos con Temp≥28°C + HR<40%: {pct_ola_calor:.1f}%")

if pct_ola_calor > 40:
    print(f"   ✅ ALTA incidencia de olas de calor")
elif pct_ola_calor > 20:
    print(f"   ⚠️  Incidencia MODERADA de olas de calor")
else:
    print(f"   ❌ Baja incidencia de olas de calor")

print(f"\n3️⃣ EVENTOS SOSTENIDOS:")
print(f"   • Eventos >12 horas: {eventos_mas_12h} de {len(eventos)} "
      f"({eventos_mas_12h/len(eventos)*100:.1f}%)")

if eventos_mas_12h > len(eventos) * 0.3:
    print(f"   ✅ Muchos eventos SOSTENIDOS (estrés prolongado)")
elif eventos_mas_12h > 0:
    print(f"   ⚠️  Algunos eventos sostenidos")
else:
    print(f"   ℹ️  Principalmente eventos aislados")

print(f"\n4️⃣ ESTACIONALIDAD:")
print(f"   • En verano: {pct_verano:.1f}%")

if pct_verano > 60:
    print(f"   ✅ FUERTE concentración en verano")
elif pct_verano > 40:
    print(f"   ⚠️  Moderada concentración en verano")
else:
    print(f"   ❌ NO concentrado en verano")

print(f"\n5️⃣ PATRÓN HORARIO:")
print(f"   • En horario diurno (06-18): {pct_diurno:.1f}%")

if pct_diurno > 70:
    print(f"   ✅ Fuertemente concentrado en horas de calor")
elif pct_diurno > 50:
    print(f"   ⚠️  Moderadamente concentrado en horas diurnas")
else:
    print(f"   ❌ NO concentrado en horas de calor")

# Conclusión final
print(f"\n💡 CONCLUSIÓN FINAL:")

if (pct_temp_alta > 70 and pct_ola_calor > 30 and pct_verano > 50):
    print(f"""
   ✅ HIPÓTESIS VALIDADA: Las demandas extremas SON eventos reales correlacionados con olas de calor.
   
   EVIDENCIAS:
   • {pct_temp_alta:.0f}% ocurren con temperaturas ≥25°C
   • {pct_ola_calor:.0f}% en condiciones de ola de calor (Temp≥28°C + HR<40%)
   • {pct_verano:.0f}% en verano
   • {pct_diurno:.0f}% en horario diurno
   • {eventos_mas_12h} eventos sostenidos >12 horas
   
   RECOMENDACIÓN:
   1. NO eliminar estos {len(df_extremos)} registros como outliers
   2. SON eventos REALES de estrés del sistema por calor
   3. Crear feature 'ola_de_calor' para el modelo
   4. Modelo debe APRENDER estos patrones extremos
   5. Sistema de alertas para Temp≥28°C + HR<40%
   
   IMPACTO EN MODELO:
   • Recuperar {len(df_extremos)} eventos extremos reales
   • Mejorar predicción en condiciones de estrés climático
   • Capturar no-linealidad: calor → demanda extrema
    """)
elif (pct_temp_alta > 50 or pct_ola_calor > 20):
    print(f"""
   ⚠️  HIPÓTESIS PARCIALMENTE VALIDADA: Algunos extremos correlacionan con calor, otros no.
   
   ACCIONES:
   1. Separar extremos por temperatura: ≥25°C vs <25°C
   2. Extremos con calor → VÁLIDOS (recuperar para modelo)
   3. Extremos sin calor → Investigar causas alternativas
   4. Puede haber otros factores: eventos sociales, fallas, etc.
   
   IMPACTO:
   • Recuperar ~{int(len(df_extremos) * pct_temp_alta/100)} extremos con calor
   • Mantener filtrado cauteloso para extremos sin explicación climática
    """)
else:
    print(f"""
   ❌ HIPÓTESIS NO VALIDADA: Demandas extremas NO correlacionan fuertemente con olas de calor.
   
   POSIBLES CAUSAS:
   1. Errores de medición (Qin sobreestimado, ΔVol subestimado)
   2. Eventos especiales no climáticos
   3. Fugas masivas o pérdidas no registradas
   4. Problemas de telemetría distribuidos
   
   ACCIONES:
   1. Auditar casos específicos con equipo operacional
   2. Verificar cálculo de Demanda = Qin - ΔVolumen
   3. Correlacionar con eventos sociales (festivales, etc.)
   4. Mantener filtrado de extremos actual
    """)

print("\n" + "=" * 80)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 80)
