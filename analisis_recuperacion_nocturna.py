"""
ANÁLISIS DE RECUPERACIÓN NOCTURNA DEL SISTEMA
==============================================

HIPÓTESIS DEL USUARIO (OPERADOR):
Los valores negativos de demanda NO son necesariamente errores.
Son normales en condiciones de BAJO CONSUMO:

1. MADRUGADA (00-06):
   • Población durmiendo
   • Qin constante > consumo
   • Sistema en recuperación

2. DÍAS DE BAJO CONSUMO (resto del día):
   • Bajas temperaturas → menos consumo
   • Días lluviosos → población no sale
   • Invierno → menor demanda general
   • Fines de semana/feriados → rutinas diferentes

OBJETIVO:
Validar si los 253 outliers negativos corresponden a:
   1. Recuperación nocturna (madrugada)
   2. Bajo consumo climático (frío, lluvia, invierno)
   3. Errores de medición reales

Analiza:
   - Distribución horaria y por período del día
   - Relación Qin vs ΔVolumen
   - Correlación con temperatura, precipitación, estación
   - Días de semana vs fin de semana
   - Comparación con demanda típica
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuración visual
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("ANÁLISIS: VALORES NEGATIVOS = ¿RECUPERACIÓN NOCTURNA LEGÍTIMA?")
print("=" * 80)
print(f"\nFecha análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================

print("\n" + "=" * 80)
print("[1/6] CARGANDO DATOS")
print("=" * 80)

# Outliers clasificados
df_outliers = pd.read_csv('outputs/outliers_clasificados.csv')
df_outliers['timestamp_utc'] = pd.to_datetime(df_outliers['timestamp_utc'], utc=True)
df_outliers['fecha_hora_local'] = pd.to_datetime(df_outliers['fecha_hora_local'], utc=True)

# Filtrar solo negativos
df_negativos = df_outliers[df_outliers['Demanda_m3_hr'] < 0].copy()

print(f"✅ Outliers negativos: {len(df_negativos)}")
print(f"   Rango demanda: {df_negativos['Demanda_m3_hr'].min():.0f} a {df_negativos['Demanda_m3_hr'].max():.0f} m³/hr")

# Datos completos para comparar con demanda típica
df_all = pd.read_csv('data/processed/data_processed_complete.csv')
df_all['timestamp_utc'] = pd.to_datetime(df_all['timestamp_utc'], utc=True)
df_all['fecha_hora_local'] = pd.to_datetime(df_all['fecha_hora_local'], utc=True)

print(f"✅ Datos completos: {len(df_all)} registros")

# ============================================================================
# 2. ANÁLISIS HORARIO
# ============================================================================

print("\n" + "=" * 80)
print("[2/6] ANÁLISIS HORARIO - ¿Ocurren en madrugada?")
print("=" * 80)

# Extraer hora
df_negativos['hora_del_dia'] = df_negativos['fecha_hora_local'].dt.hour
df_all['hora_del_dia'] = df_all['fecha_hora_local'].dt.hour

# Distribución horaria de negativos
dist_horaria = df_negativos['hora_del_dia'].value_counts().sort_index()

print("\n📊 DISTRIBUCIÓN HORARIA DE VALORES NEGATIVOS:")
print("\nHora    N     %")
print("-" * 30)
for hora, count in dist_horaria.items():
    pct = count / len(df_negativos) * 100
    bar = "█" * int(pct / 2)
    print(f"{hora:02d}:00  {count:3d}  {pct:5.1f}%  {bar}")

# Clasificar por período del día
def clasificar_periodo(hora):
    if 0 <= hora < 6:
        return "Madrugada (00-06)"
    elif 6 <= hora < 12:
        return "Mañana (06-12)"
    elif 12 <= hora < 18:
        return "Tarde (12-18)"
    else:
        return "Noche (18-24)"

df_negativos['periodo'] = df_negativos['hora_del_dia'].apply(clasificar_periodo)

print("\n📊 DISTRIBUCIÓN POR PERÍODO DEL DÍA:")
periodo_dist = df_negativos['periodo'].value_counts()
for periodo in ["Madrugada (00-06)", "Mañana (06-12)", "Tarde (12-18)", "Noche (18-24)"]:
    count = periodo_dist.get(periodo, 0)
    pct = count / len(df_negativos) * 100
    print(f"   {periodo:<20}: {count:3d} ({pct:5.1f}%)")

# ============================================================================
# 3. ANÁLISIS DE RECUPERACIÓN: Qin vs ΔVolumen
# ============================================================================

print("\n" + "=" * 80)
print("[3/6] ANÁLISIS DE RECUPERACIÓN: Qin vs ΔVolumen")
print("=" * 80)

# Calcular ΔVolumen
df_negativos['Delta_Volumen'] = df_negativos['Qin'] - df_negativos['Demanda_m3_hr']

print("\n📊 ESTADÍSTICAS DE RECUPERACIÓN:")
print(f"\n   Qin (caudal fuentes):")
print(f"      • Media:    {df_negativos['Qin'].mean():,.0f} m³/hr")
print(f"      • Rango:    {df_negativos['Qin'].min():,.0f} - {df_negativos['Qin'].max():,.0f} m³/hr")

print(f"\n   ΔVolumen (aumento del sistema):")
print(f"      • Media:    {df_negativos['Delta_Volumen'].mean():,.0f} m³/hr")
print(f"      • Rango:    {df_negativos['Delta_Volumen'].min():,.0f} - {df_negativos['Delta_Volumen'].max():,.0f} m³/hr")

print(f"\n   Demanda aparente (negativa):")
print(f"      • Media:    {df_negativos['Demanda_m3_hr'].mean():,.0f} m³/hr")
print(f"      • Rango:    {df_negativos['Demanda_m3_hr'].min():,.0f} - {df_negativos['Demanda_m3_hr'].max():,.0f} m³/hr")

# Verificar condición: ΔVolumen > Qin
df_negativos['cumple_recuperacion'] = df_negativos['Delta_Volumen'] > df_negativos['Qin']
recuperacion_valida = df_negativos['cumple_recuperacion'].sum()

print(f"\n💡 VALIDACIÓN HIPÓTESIS:")
print(f"   Casos donde ΔVolumen > Qin: {recuperacion_valida} de {len(df_negativos)} ({recuperacion_valida/len(df_negativos)*100:.1f}%)")

# ============================================================================
# 4. ANÁLISIS DE DÍA DE LA SEMANA
# ============================================================================

print("\n" + "=" * 80)
print("[4/6] ANÁLISIS POR DÍA DE LA SEMANA")
print("=" * 80)

df_negativos['dia_semana'] = df_negativos['fecha_hora_local'].dt.day_name()
df_negativos['es_fin_semana'] = df_negativos['fecha_hora_local'].dt.dayofweek >= 5

dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

print("\n📊 DISTRIBUCIÓN POR DÍA DE LA SEMANA:")
print("\nDía          N     %")
print("-" * 35)
dia_dist = df_negativos['dia_semana'].value_counts()
for dia_en, dia_es in zip(dias_orden, dias_es):
    count = dia_dist.get(dia_en, 0)
    pct = count / len(df_negativos) * 100
    marca = "🔵" if dia_en in ['Saturday', 'Sunday'] else "⚪"
    print(f"{marca} {dia_es:<12} {count:3d}  {pct:5.1f}%")

fin_semana = df_negativos['es_fin_semana'].sum()
print(f"\n   Fin de semana: {fin_semana} ({fin_semana/len(df_negativos)*100:.1f}%)")
print(f"   Entre semana:  {len(df_negativos)-fin_semana} ({(len(df_negativos)-fin_semana)/len(df_negativos)*100:.1f}%)")

# ============================================================================
# 5. ANÁLISIS CLIMÁTICO - Temperatura, Precipitación, Estación
# ============================================================================

print("\n" + "=" * 80)
print("[5/7] ANÁLISIS CLIMÁTICO - ¿Bajo consumo por frío/lluvia/invierno?")
print("=" * 80)

# Separar madrugada vs resto del día
df_negativos['es_madrugada'] = df_negativos['hora_del_dia'] < 6
negativos_madrugada = df_negativos[df_negativos['es_madrugada']]
negativos_diurnos = df_negativos[~df_negativos['es_madrugada']]

print(f"\n📊 SEPARACIÓN:")
print(f"   Madrugada (00-06):  {len(negativos_madrugada):3d} ({len(negativos_madrugada)/len(df_negativos)*100:.1f}%)")
print(f"   Resto del día:      {len(negativos_diurnos):3d} ({len(negativos_diurnos)/len(df_negativos)*100:.1f}%)")

# Analizar temperatura (solo outliers tienen datos climáticos)
print(f"\n🌡️  ANÁLISIS DE TEMPERATURA:")
temp_negativos_diurnos = negativos_diurnos['temp'].describe()
temp_todos_negativos = df_negativos['temp'].describe()

print(f"\n   Negativos diurnos:")
print(f"      • Media:  {temp_negativos_diurnos['mean']:.1f}°C")
print(f"      • Mín:    {temp_negativos_diurnos['min']:.1f}°C")
print(f"      • Máx:    {temp_negativos_diurnos['max']:.1f}°C")
print(f"      • Q25:    {temp_negativos_diurnos['25%']:.1f}°C")

print(f"\n   Todos los negativos:")
print(f"      • Media:  {temp_todos_negativos['mean']:.1f}°C")
print(f"      • Mín:    {temp_todos_negativos['min']:.1f}°C")
print(f"      • Máx:    {temp_todos_negativos['max']:.1f}°C")
print(f"      • Q25:    {temp_todos_negativos['25%']:.1f}°C")

# Clasificar temperatura
temp_frio = (negativos_diurnos['temp'] < 12).sum()
temp_media = ((negativos_diurnos['temp'] >= 12) & (negativos_diurnos['temp'] < 18)).sum()
temp_alta = (negativos_diurnos['temp'] >= 18).sum()

print(f"\n   Distribución térmica (negativos diurnos):")
print(f"      • Frío (<12°C):      {temp_frio:3d} ({temp_frio/len(negativos_diurnos)*100:.1f}%)")
print(f"      • Templado (12-18°C): {temp_media:3d} ({temp_media/len(negativos_diurnos)*100:.1f}%)")
print(f"      • Cálido (>18°C):     {temp_alta:3d} ({temp_alta/len(negativos_diurnos)*100:.1f}%)")

# Analizar precipitación
print(f"\n🌧️  ANÁLISIS DE PRECIPITACIÓN:")
hay_lluvia_diurnos = (negativos_diurnos['mmhr'] > 0).sum()
lluvia_intensa_diurnos = (negativos_diurnos['mmhr'] > 2).sum()

print(f"   Negativos diurnos con lluvia: {hay_lluvia_diurnos}/{len(negativos_diurnos)} ({hay_lluvia_diurnos/len(negativos_diurnos)*100:.1f}%)")
print(f"   Lluvia intensa (>2mm/hr):     {lluvia_intensa_diurnos}/{len(negativos_diurnos)} ({lluvia_intensa_diurnos/len(negativos_diurnos)*100:.1f}%)")

precip_media_diurnos = negativos_diurnos['mmhr'].mean()
precip_max_diurnos = negativos_diurnos['mmhr'].max()
print(f"   Precipitación media:          {precip_media_diurnos:.2f} mm/hr")
print(f"   Precipitación máxima:         {precip_max_diurnos:.2f} mm/hr")

# Analizar por mes/estación
print(f"\n🗓️  ANÁLISIS POR ESTACIÓN:")
negativos_diurnos['mes'] = negativos_diurnos['fecha_hora_local'].dt.month

def clasificar_estacion(mes):
    if mes in [12, 1, 2]:
        return "Verano"
    elif mes in [3, 4, 5]:
        return "Otoño"
    elif mes in [6, 7, 8]:
        return "Invierno"
    else:
        return "Primavera"

negativos_diurnos['estacion'] = negativos_diurnos['mes'].apply(clasificar_estacion)
estacion_dist = negativos_diurnos['estacion'].value_counts()

for estacion in ["Verano", "Otoño", "Invierno", "Primavera"]:
    count = estacion_dist.get(estacion, 0)
    pct = count / len(negativos_diurnos) * 100
    print(f"   {estacion:<12}: {count:3d} ({pct:5.1f}%)")

# Estadísticas por estación
print(f"\n📊 TEMPERATURA Y DEMANDA POR ESTACIÓN (negativos diurnos):")
print(f"\n{'Estación':<12} {'N':>4} {'Temp Media':>12} {'Demanda Media':>15}")
print("-" * 50)
for estacion in ["Verano", "Otoño", "Invierno", "Primavera"]:
    df_est = negativos_diurnos[negativos_diurnos['estacion'] == estacion]
    if len(df_est) > 0:
        temp_media = df_est['temp'].mean()
        demanda_media = df_est['Demanda_m3_hr'].mean()
        print(f"{estacion:<12} {len(df_est):4d} {temp_media:10.1f}°C {demanda_media:13,.0f} m³/hr")

# ============================================================================
# 6. COMPARACIÓN CON DEMANDA TÍPICA NOCTURNA
# ============================================================================

print("\n" + "=" * 80)
print("[6/7] COMPARACIÓN CON DEMANDA TÍPICA NOCTURNA")
print("=" * 80)

# Filtrar demanda normal (entre 0 y 30,000)
# Usar hour en lugar de hora_del_dia
df_all['hora_del_dia'] = df_all['fecha_hora_local'].dt.hour
df_normal = df_all[(df_all['Demanda_m3_hr'] >= 0) & (df_all['Demanda_m3_hr'] <= 30000)].copy()

# Calcular demanda típica por hora
demanda_tipica_hora = df_normal.groupby('hora_del_dia')['Demanda_m3_hr'].agg(['mean', 'std', 'min', 'max'])

print("\n📊 DEMANDA TÍPICA POR HORA (solo valores normales):")
print("\nHora    Media      Std      Mín      Máx")
print("-" * 50)
for hora in range(24):
    if hora in demanda_tipica_hora.index:
        stats = demanda_tipica_hora.loc[hora]
        marca = "🌙" if 0 <= hora < 6 else "☀️"
        print(f"{marca} {hora:02d}:00  {stats['mean']:7,.0f}  {stats['std']:6,.0f}  {stats['min']:6,.0f}  {stats['max']:7,.0f}")

# Comparar Qin de negativos con demanda típica nocturna
demanda_madrugada_media = demanda_tipica_hora.loc[0:5, 'mean'].mean()
qin_negativos_madrugada = df_negativos[df_negativos['hora_del_dia'] < 6]['Qin'].mean()

print(f"\n💡 COMPARACIÓN MADRUGADA (00-06):")
print(f"   Demanda típica:        {demanda_madrugada_media:,.0f} m³/hr")
print(f"   Qin en negativos:      {qin_negativos_madrugada:,.0f} m³/hr")
print(f"   Diferencia:            {qin_negativos_madrugada - demanda_madrugada_media:+,.0f} m³/hr")

if qin_negativos_madrugada > demanda_madrugada_media:
    print(f"   ✅ Qin SUPERA demanda típica nocturna → Recuperación esperada")
else:
    print(f"   ⚠️  Qin NO supera demanda típica nocturna → Revisar")

# ============================================================================
# 7. VISUALIZACIÓN
# ============================================================================

print("\n" + "=" * 80)
print("[7/7] GENERANDO VISUALIZACIONES")
print("=" * 80)

fig, axes = plt.subplots(4, 2, figsize=(16, 18))
fig.suptitle('ANÁLISIS: Valores Negativos = Recuperación por Bajo Consumo (Madrugada + Clima)', 
             fontsize=16, fontweight='bold', y=0.995)

# 1. Distribución horaria
ax = axes[0, 0]
dist_horaria.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
ax.axvspan(-0.5, 5.5, alpha=0.2, color='navy', label='Madrugada (00-06)')
ax.set_xlabel('Hora del día')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución Horaria de Valores Negativos')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Qin vs ΔVolumen
ax = axes[0, 1]
ax.scatter(df_negativos['Qin'], df_negativos['Delta_Volumen'], 
           alpha=0.6, s=50, c=df_negativos['hora_del_dia'], cmap='twilight')
ax.plot([0, df_negativos['Qin'].max()], [0, df_negativos['Qin'].max()], 
        'r--', linewidth=2, label='ΔVol = Qin (equilibrio)')
ax.set_xlabel('Qin (m³/hr)')
ax.set_ylabel('ΔVolumen (m³/hr)')
ax.set_title('Qin vs ΔVolumen en Valores Negativos')
ax.legend()
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Hora del día')

# 3. Boxplot por período
ax = axes[1, 0]
periodos_orden = ["Madrugada (00-06)", "Mañana (06-12)", "Tarde (12-18)", "Noche (18-24)"]
df_negativos_sorted = df_negativos.copy()
df_negativos_sorted['periodo'] = pd.Categorical(df_negativos_sorted['periodo'], 
                                                 categories=periodos_orden, ordered=True)
sns.boxplot(data=df_negativos_sorted, x='periodo', y='Demanda_m3_hr', ax=ax, palette='Set2')
ax.set_xlabel('Período del día')
ax.set_ylabel('Demanda (m³/hr)')
ax.set_title('Distribución de Valores Negativos por Período')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

# 4. Día de la semana
ax = axes[1, 1]
dia_counts = []
for dia_en in dias_orden:
    dia_counts.append(dia_dist.get(dia_en, 0))
colores = ['lightcoral' if dia in ['Saturday', 'Sunday'] else 'steelblue' 
           for dia in dias_orden]
ax.bar(dias_es, dia_counts, color=colores, alpha=0.7)
ax.set_xlabel('Día de la semana')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución por Día de la Semana')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# 5. Comparación demanda típica vs Qin negativos por hora
ax = axes[2, 0]
horas = range(24)
demanda_media = [demanda_tipica_hora.loc[h, 'mean'] if h in demanda_tipica_hora.index else 0 
                 for h in horas]
qin_por_hora = df_negativos.groupby('hora_del_dia')['Qin'].mean()
qin_media = [qin_por_hora.get(h, 0) for h in horas]

ax.plot(horas, demanda_media, 'o-', linewidth=2, markersize=6, label='Demanda típica', color='green')
ax.plot(horas, qin_media, 's-', linewidth=2, markersize=6, label='Qin (en negativos)', color='orange')
ax.axvspan(-0.5, 5.5, alpha=0.2, color='navy')
ax.set_xlabel('Hora del día')
ax.set_ylabel('m³/hr')
ax.set_title('Comparación: Demanda Típica vs Qin en Valores Negativos')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 24, 2))

# 6. Serie temporal de negativos
ax = axes[2, 1]
df_negativos_sorted_time = df_negativos.sort_values('fecha_hora_local')
colores_hora = ['navy' if h < 6 else 'steelblue' for h in df_negativos_sorted_time['hora_del_dia']]
ax.scatter(df_negativos_sorted_time['fecha_hora_local'], 
           df_negativos_sorted_time['Demanda_m3_hr'],
           c=colores_hora, alpha=0.6, s=30)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Fecha')
ax.set_ylabel('Demanda (m³/hr)')
ax.set_title('Serie Temporal de Valores Negativos')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

# Leyenda manual
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='navy', alpha=0.6, label='Madrugada (00-06)'),
                   Patch(facecolor='steelblue', alpha=0.6, label='Otros períodos')]
ax.legend(handles=legend_elements, loc='lower right')

# 7. Análisis climático - Temperatura
ax = axes[3, 0]
# Scatter temperatura vs demanda negativos diurnos
scatter = ax.scatter(negativos_diurnos['temp'], negativos_diurnos['Demanda_m3_hr'],
                     c=negativos_diurnos['mmhr'], cmap='Blues', alpha=0.6, s=60,
                     edgecolors='black', linewidth=0.5)
ax.axvline(x=12, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Frío (<12°C)')
ax.axvline(x=18, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Cálido (>18°C)')
ax.set_xlabel('Temperatura (°C)')
ax.set_ylabel('Demanda Negativa (m³/hr)')
ax.set_title('Temperatura vs Demanda Negativa (Resto del día)')
ax.legend()
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Precipitación (mm/hr)')

# 8. Distribución por estación
ax = axes[3, 1]
estaciones_orden = ["Verano", "Otoño", "Invierno", "Primavera"]
estacion_counts = [estacion_dist.get(e, 0) for e in estaciones_orden]
colores_estacion = ['#FFD700', '#FF8C00', '#4169E1', '#32CD32']
bars = ax.bar(estaciones_orden, estacion_counts, color=colores_estacion, alpha=0.7, 
              edgecolor='black', linewidth=1.5)
ax.set_xlabel('Estación del año')
ax.set_ylabel('Frecuencia')
ax.set_title('Negativos Diurnos por Estación')
ax.grid(True, alpha=0.3, axis='y')

# Anotar porcentajes
for bar, count in zip(bars, estacion_counts):
    height = bar.get_height()
    if count > 0:
        pct = count / len(negativos_diurnos) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/analisis_recuperacion_nocturna.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado: outputs/analisis_recuperacion_nocturna.png")

# ============================================================================
# 7. CONCLUSIONES Y RECOMENDACIONES
# ============================================================================

print("\n" + "=" * 80)
print("📊 CONCLUSIONES")
print("=" * 80)

madrugada_pct = (df_negativos['periodo'] == 'Madrugada (00-06)').sum() / len(df_negativos) * 100
recuperacion_pct = df_negativos['cumple_recuperacion'].sum() / len(df_negativos) * 100

print(f"\n1️⃣ DISTRIBUCIÓN TEMPORAL:")
print(f"   • Madrugada (00-06):  {madrugada_pct:.1f}% de los negativos")
if madrugada_pct > 60:
    print(f"   ✅ ALTA concentración nocturna → Consistente con recuperación")
elif madrugada_pct > 40:
    print(f"   ⚠️  MODERADA concentración nocturna → Recuperación parcial")
else:
    print(f"   ❌ BAJA concentración nocturna → NO es principalmente recuperación")

print(f"\n2️⃣ VALIDACIÓN FÍSICA:")
print(f"   • Casos con ΔVolumen > Qin: {recuperacion_pct:.1f}%")
if recuperacion_pct > 80:
    print(f"   ✅ MAYORÍA cumple física de recuperación")
elif recuperacion_pct > 50:
    print(f"   ⚠️  PARCIALMENTE válido como recuperación")
else:
    print(f"   ❌ NO cumple física de recuperación")

print(f"\n3️⃣ COMPARACIÓN CON OPERACIÓN TÍPICA:")
print(f"   • Demanda típica madrugada: {demanda_madrugada_media:,.0f} m³/hr")
print(f"   • Qin promedio (negativos): {qin_negativos_madrugada:,.0f} m³/hr")
if qin_negativos_madrugada > demanda_madrugada_media:
    print(f"   ✅ Qin EXCEDE demanda típica → Recuperación esperada")
else:
    print(f"   ❌ Qin NO excede demanda típica → Revisar anomalía")

print(f"\n4️⃣ ANÁLISIS CLIMÁTICO (negativos diurnos):")
print(f"   • Ocurren fuera de madrugada: {len(negativos_diurnos)} ({len(negativos_diurnos)/len(df_negativos)*100:.1f}%)")
print(f"   • Temperatura media: {negativos_diurnos['temp'].mean():.1f}°C")
print(f"   • Con temperaturas frías (<12°C): {temp_frio}/{len(negativos_diurnos)} ({temp_frio/len(negativos_diurnos)*100:.1f}%)")
print(f"   • Con lluvia: {hay_lluvia_diurnos}/{len(negativos_diurnos)} ({hay_lluvia_diurnos/len(negativos_diurnos)*100:.1f}%)")
print(f"   • En invierno: {estacion_dist.get('Invierno', 0)}/{len(negativos_diurnos)} ({estacion_dist.get('Invierno', 0)/len(negativos_diurnos)*100:.1f}%)")

if temp_frio/len(negativos_diurnos) > 0.5 or estacion_dist.get('Invierno', 0)/len(negativos_diurnos) > 0.4:
    print(f"   ✅ ALTA correlación con condiciones de bajo consumo (frío/invierno)")
elif temp_frio/len(negativos_diurnos) > 0.3:
    print(f"   ⚠️  MODERADA correlación con bajo consumo")
else:
    print(f"   ❌ BAJA correlación con bajo consumo climático")

print(f"\n💡 RECOMENDACIÓN FINAL:")

if madrugada_pct > 60 and recuperacion_pct > 80 and qin_negativos_madrugada > demanda_madrugada_media:
    print(f"""
   ✅ HIPÓTESIS VALIDADA: Los valores negativos SON recuperación legítima (nocturna + bajo consumo climático).
   
   ACCIONES:
   1. NO eliminar estos registros como outliers
   2. Tratar valores negativos como demanda = 0 o pequeña
   3. Considerar variable binaria 'en_recuperacion' como feature
   4. Ajustar umbral de outliers: solo extremos fuera de [-20k, +50k]
   5. Crear feature 'capacidad_recuperacion' = ΔVolumen - demanda_típica
   
   IMPACTO EN MODELO:
   • Recuperar {len(df_negativos)} registros válidos ({len(df_negativos)/len(df_all)*100:.2f}% del dataset)
   • Mejorar aprendizaje de patrones nocturnos
   • Capturar dinámica de acumulación/consumo del sistema
    """)
elif madrugada_pct > 40 or recuperacion_pct > 50:
    print(f"""
   ⚠️  HIPÓTESIS PARCIALMENTE VÁLIDA: Algunos negativos son recuperación, otros posibles errores.
   
   ACCIONES:
   1. Separar negativos por hora: madrugada vs resto del día
   2. Negativos madrugada con Qin>demanda_típica → VÁLIDOS
   3. Negativos diurnos o con Qin bajo → Revisar individualmente
   4. Crear filtro condicional por hora + Qin
   
   IMPACTO EN MODELO:
   • Recuperar ~{int(len(df_negativos)*madrugada_pct/100)} registros de madrugada
   • Mantener filtrado para casos diurnos sospechosos
    """)
else:
    print(f"""
   ❌ HIPÓTESIS NO VALIDADA: Valores negativos NO parecen ser recuperación normal.
   
   POSIBLES CAUSAS:
   1. Errores de medición de Qin (subestimado)
   2. Errores en cálculo de ΔVolumen (sensor volumen)
   3. Pérdidas no contabilizadas
   4. Recirculaciones internas no modeladas
   
   ACCIONES:
   1. Mantener filtrado actual de negativos
   2. Auditar sensores de Qin y volumen total
   3. Revisar casos específicos con equipo operacional
    """)

print("\n" + "=" * 80)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 80)
