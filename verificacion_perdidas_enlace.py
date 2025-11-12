"""
VERIFICACIÓN: PÉRDIDAS DE ENLACE ANTES/DESPUÉS DE OUTLIERS EXTREMOS
====================================================================

HIPÓTESIS DEL USUARIO (OPERADOR):
Los outliers extremos con ΔVolumen muy negativo pueden estar causados por:
   • Pérdida de enlace ANTES del outlier → Volumen aparece como 0
   • Recuperación de enlace EN el outlier → Volumen "salta" a valor real
   • Esto causa un ΔVolumen negativo artificial muy grande

CRITERIO:
   Pérdida de enlace masiva = >15% de los 89 estanques con volumen=0 o null

ANÁLISIS:
   Para cada uno de los 86 outliers extremos, verificar:
   1. Hora del outlier (t)
   2. Hora anterior (t-1)
   3. Hora posterior (t+1)
   
   Detectar si hay pérdida masiva en t-1, t, o t+1
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
print("VERIFICACIÓN: PÉRDIDAS DE ENLACE EN OUTLIERS EXTREMOS")
print("=" * 80)
print(f"\nFecha análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================

print("\n" + "=" * 80)
print("[1/5] CARGANDO DATOS")
print("=" * 80)

# Cargar outliers extremos
df_outliers = pd.read_csv('outputs/outliers_clasificados.csv')
df_outliers['timestamp_utc'] = pd.to_datetime(df_outliers['timestamp_utc'], 
                                               utc=True)
df_extremos = df_outliers[df_outliers['Demanda_m3_hr'] > 30000].copy()

print(f"✅ Outliers extremos cargados: {len(df_extremos)}")

# Cargar volúmenes individuales por estanque
df_vol_tanks = pd.read_csv('data/raw/Vol_X_TK_Hr_m3_UTC.csv')

# Identificar columna de timestamp
timestamp_col = None
for col in df_vol_tanks.columns:
    if 'timestamp' in col.lower() or 'fecha' in col.lower():
        timestamp_col = col
        break

if timestamp_col:
    df_vol_tanks[timestamp_col] = pd.to_datetime(df_vol_tanks[timestamp_col], 
                                                  utc=True)
    print(f"✅ Volúmenes por estanque cargados: {len(df_vol_tanks)} registros")
    print(f"   Columna timestamp: {timestamp_col}")
else:
    print("❌ No se encontró columna de timestamp")
    exit()

# Identificar columnas de estanques (todas excepto timestamp)
tank_cols = [col for col in df_vol_tanks.columns if col != timestamp_col]
print(f"   Estanques detectados: {len(tank_cols)}")
if len(tank_cols) > 0:
    print(f"   Ejemplos: {tank_cols[:3]}")

# ============================================================================
# 2. FUNCIÓN PARA DETECTAR PÉRDIDA DE ENLACE
# ============================================================================

print("\n" + "=" * 80)
print("[2/5] DEFINIENDO CRITERIOS DE PÉRDIDA DE ENLACE")
print("=" * 80)

THRESHOLD_PERDIDA = 0.15  # 15% de estanques
print(f"\n🔍 CRITERIO: Pérdida masiva si >{THRESHOLD_PERDIDA*100:.0f}% "
      f"estanques con problemas")

def analizar_perdida_enlace(timestamp, df_vol):
    """
    Analiza si hay pérdida de enlace en un timestamp dado
    
    Returns:
        dict con: n_ceros, n_nulls, pct_problemas, hay_perdida
    """
    # Buscar registro en ese timestamp
    registro = df_vol[df_vol[timestamp_col] == timestamp]
    
    if len(registro) == 0:
        return {
            'n_ceros': 0,
            'n_nulls': len(tank_cols),
            'pct_problemas': 100.0,
            'hay_perdida': True,
            'existe_registro': False
        }
    
    registro = registro.iloc[0]
    
    # Contar problemas
    n_ceros = 0
    n_nulls = 0
    
    for col in tank_cols:
        valor = registro[col]
        if pd.isna(valor):
            n_nulls += 1
        elif valor == 0:
            n_ceros += 1
    
    n_problemas = n_ceros + n_nulls
    pct_problemas = (n_problemas / len(tank_cols)) * 100
    hay_perdida = pct_problemas > (THRESHOLD_PERDIDA * 100)
    
    return {
        'n_ceros': n_ceros,
        'n_nulls': n_nulls,
        'pct_problemas': pct_problemas,
        'hay_perdida': hay_perdida,
        'existe_registro': True
    }

# ============================================================================
# 3. ANALIZAR VENTANA TEMPORAL PARA CADA OUTLIER
# ============================================================================

print("\n" + "=" * 80)
print("[3/5] ANALIZANDO VENTANA TEMPORAL (t-1, t, t+1) PARA CADA OUTLIER")
print("=" * 80)

resultados = []

print(f"\n⏳ Procesando {len(df_extremos)} outliers...")

for idx, row in df_extremos.iterrows():
    timestamp = row['timestamp_utc']
    demanda = row['Demanda_m3_hr']
    
    # Ventana temporal
    t_anterior = timestamp - pd.Timedelta(hours=1)
    t_actual = timestamp
    t_posterior = timestamp + pd.Timedelta(hours=1)
    
    # Analizar cada momento
    analisis_t_1 = analizar_perdida_enlace(t_anterior, df_vol_tanks)
    analisis_t = analizar_perdida_enlace(t_actual, df_vol_tanks)
    analisis_t_plus = analizar_perdida_enlace(t_posterior, df_vol_tanks)
    
    # Guardar resultado
    resultado = {
        'timestamp': timestamp,
        'fecha_hora_local': row['fecha_hora_local'],
        'Demanda_m3_hr': demanda,
        'Qin': row['Qin'],
        'Delta_Volumen': row['Qin'] - demanda,
        
        # t-1 (hora anterior)
        't_1_existe': analisis_t_1['existe_registro'],
        't_1_ceros': analisis_t_1['n_ceros'],
        't_1_nulls': analisis_t_1['n_nulls'],
        't_1_pct_problemas': analisis_t_1['pct_problemas'],
        't_1_perdida': analisis_t_1['hay_perdida'],
        
        # t (hora del outlier)
        't_existe': analisis_t['existe_registro'],
        't_ceros': analisis_t['n_ceros'],
        't_nulls': analisis_t['n_nulls'],
        't_pct_problemas': analisis_t['pct_problemas'],
        't_perdida': analisis_t['hay_perdida'],
        
        # t+1 (hora posterior)
        't_plus_existe': analisis_t_plus['existe_registro'],
        't_plus_ceros': analisis_t_plus['n_ceros'],
        't_plus_nulls': analisis_t_plus['n_nulls'],
        't_plus_pct_problemas': analisis_t_plus['pct_problemas'],
        't_plus_perdida': analisis_t_plus['hay_perdida'],
    }
    
    # Detectar patrón de recuperación
    resultado['patron_recuperacion'] = (
        analisis_t_1['hay_perdida'] and not analisis_t['hay_perdida']
    )
    
    # Detectar patrón de pérdida
    resultado['patron_perdida'] = (
        not analisis_t['hay_perdida'] and analisis_t_plus['hay_perdida']
    )
    
    # Cualquier pérdida en ventana
    resultado['tiene_perdida_ventana'] = (
        analisis_t_1['hay_perdida'] or 
        analisis_t['hay_perdida'] or 
        analisis_t_plus['hay_perdida']
    )
    
    resultados.append(resultado)

df_resultados = pd.DataFrame(resultados)

print(f"✅ Análisis completado")

# ============================================================================
# 4. ESTADÍSTICAS Y HALLAZGOS
# ============================================================================

print("\n" + "=" * 80)
print("[4/5] ESTADÍSTICAS Y HALLAZGOS")
print("=" * 80)

# Contar patrones
n_con_perdida = df_resultados['tiene_perdida_ventana'].sum()
n_patron_recuperacion = df_resultados['patron_recuperacion'].sum()
n_patron_perdida = df_resultados['patron_perdida'].sum()

print(f"\n📊 RESUMEN GENERAL:")
print(f"   Total outliers analizados:           {len(df_resultados)}")
print(f"   Con pérdida de enlace en ventana:    {n_con_perdida} "
      f"({n_con_perdida/len(df_resultados)*100:.1f}%)")
print(f"   Sin pérdida de enlace:               "
      f"{len(df_resultados) - n_con_perdida} "
      f"({(len(df_resultados) - n_con_perdida)/len(df_resultados)*100:.1f}%)")

print(f"\n🔍 PATRONES DETECTADOS:")
print(f"   Patrón RECUPERACIÓN (t-1 perdida → t OK):  "
      f"{n_patron_recuperacion} "
      f"({n_patron_recuperacion/len(df_resultados)*100:.1f}%)")
print(f"   Patrón PÉRDIDA (t OK → t+1 perdida):       "
      f"{n_patron_perdida} "
      f"({n_patron_perdida/len(df_resultados)*100:.1f}%)")

# Detalle por momento
print(f"\n📊 PÉRDIDAS POR MOMENTO:")
perdida_t_1 = df_resultados['t_1_perdida'].sum()
perdida_t = df_resultados['t_perdida'].sum()
perdida_t_plus = df_resultados['t_plus_perdida'].sum()

print(f"   En t-1 (hora anterior):   {perdida_t_1} "
      f"({perdida_t_1/len(df_resultados)*100:.1f}%)")
print(f"   En t (hora del outlier):  {perdida_t} "
      f"({perdida_t/len(df_resultados)*100:.1f}%)")
print(f"   En t+1 (hora posterior):  {perdida_t_plus} "
      f"({perdida_t_plus/len(df_resultados)*100:.1f}%)")

# Promedio de % de problemas
print(f"\n📊 PORCENTAJE PROMEDIO DE ESTANQUES CON PROBLEMAS:")
print(f"   En t-1:  {df_resultados['t_1_pct_problemas'].mean():.1f}%")
print(f"   En t:    {df_resultados['t_pct_problemas'].mean():.1f}%")
print(f"   En t+1:  {df_resultados['t_plus_pct_problemas'].mean():.1f}%")

# Top 10 casos con mayor ΔVolumen negativo
print(f"\n📋 TOP 10 CASOS CON MAYOR ΔVol NEGATIVO:")
print()
df_top10 = df_resultados.nsmallest(10, 'Delta_Volumen')

print(f"{'Fecha':<20} {'ΔVol':>12} {'t-1':>6} {'t':>6} {'t+1':>6} {'Patrón':<15}")
print("-" * 80)
for _, row in df_top10.iterrows():
    fecha = str(row['fecha_hora_local'])[:19]
    delta = row['Delta_Volumen']
    
    t_1_mark = "❌" if row['t_1_perdida'] else "✅"
    t_mark = "❌" if row['t_perdida'] else "✅"
    t_plus_mark = "❌" if row['t_plus_perdida'] else "✅"
    
    if row['patron_recuperacion']:
        patron = "RECUPERACIÓN"
    elif row['patron_perdida']:
        patron = "PÉRDIDA"
    elif row['tiene_perdida_ventana']:
        patron = "Con pérdida"
    else:
        patron = "Sin pérdida"
    
    print(f"{fecha:<20} {delta:>12,.0f} {t_1_mark:>6} {t_mark:>6} "
          f"{t_plus_mark:>6} {patron:<15}")

# Correlación entre pérdida y magnitud de ΔVolumen
print(f"\n📊 ANÁLISIS DE CORRELACIÓN:")
delta_con_perdida = df_resultados[
    df_resultados['tiene_perdida_ventana']]['Delta_Volumen'].mean()
delta_sin_perdida = df_resultados[
    ~df_resultados['tiene_perdida_ventana']]['Delta_Volumen'].mean()

print(f"   ΔVol promedio CON pérdida:    {delta_con_perdida:,.0f} m³/hr")
print(f"   ΔVol promedio SIN pérdida:    {delta_sin_perdida:,.0f} m³/hr")
print(f"   Diferencia:                   "
      f"{abs(delta_con_perdida - delta_sin_perdida):,.0f} m³/hr")

if abs(delta_con_perdida) > abs(delta_sin_perdida) * 1.5:
    print(f"   ✅ FUERTE correlación: Pérdidas causan ΔVol más extremos")
elif abs(delta_con_perdida) > abs(delta_sin_perdida) * 1.2:
    print(f"   ⚠️  Correlación MODERADA")
else:
    print(f"   ❌ NO hay correlación clara")

# ============================================================================
# 5. VISUALIZACIÓN
# ============================================================================

print("\n" + "=" * 80)
print("[5/5] GENERANDO VISUALIZACIONES")
print("=" * 80)

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Análisis: Pérdidas de Enlace y Outliers Extremos', 
             fontsize=16, fontweight='bold', y=0.995)

# 1. Distribución de pérdidas por momento
ax = axes[0, 0]
momentos = ['t-1\n(anterior)', 't\n(outlier)', 't+1\n(posterior)']
perdidas = [perdida_t_1, perdida_t, perdida_t_plus]
colors = ['orange', 'red', 'purple']
bars = ax.bar(momentos, perdidas, color=colors, alpha=0.7, 
              edgecolor='black', linewidth=1.5)
ax.set_ylabel('Número de casos')
ax.set_title('Pérdidas de Enlace por Momento')
ax.axhline(y=len(df_resultados)*0.15, color='red', linestyle='--', 
           linewidth=2, alpha=0.5, label='15% del total')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, perdidas):
    height = bar.get_height()
    pct = count / len(df_resultados) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({pct:.1f}%)',
            ha='center', va='bottom', fontweight='bold')

# 2. Patrones detectados
ax = axes[0, 1]
patron_counts = {
    'Recuperación\n(t-1❌→t✅)': n_patron_recuperacion,
    'Pérdida\n(t✅→t+1❌)': n_patron_perdida,
    'Otras con\npérdida': n_con_perdida - n_patron_recuperacion - n_patron_perdida,
    'Sin pérdida': len(df_resultados) - n_con_perdida
}
colors_patron = ['green', 'red', 'orange', 'blue']
bars2 = ax.bar(patron_counts.keys(), patron_counts.values(), 
               color=colors_patron, alpha=0.7, 
               edgecolor='black', linewidth=1.5)
ax.set_ylabel('Número de casos')
ax.set_title('Patrones de Pérdida de Enlace')
ax.tick_params(axis='x', rotation=0)
ax.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars2, patron_counts.values()):
    height = bar.get_height()
    pct = count / len(df_resultados) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({pct:.0f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

# 3. ΔVolumen: Con vs Sin pérdida
ax = axes[1, 0]
data_boxplot = []
labels_boxplot = []

con_perdida = df_resultados[df_resultados['tiene_perdida_ventana']]
sin_perdida = df_resultados[~df_resultados['tiene_perdida_ventana']]

if len(con_perdida) > 0:
    data_boxplot.append(con_perdida['Delta_Volumen'])
    labels_boxplot.append('Con pérdida')
if len(sin_perdida) > 0:
    data_boxplot.append(sin_perdida['Delta_Volumen'])
    labels_boxplot.append('Sin pérdida')

if len(data_boxplot) > 0:
    bp = ax.boxplot(data_boxplot, labels=labels_boxplot, patch_artist=True,
                    showmeans=True)
    colors_box = ['lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color='blue', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_ylabel('ΔVolumen (m³/hr)')
    ax.set_title('ΔVolumen: Con vs Sin Pérdida de Enlace')
    ax.grid(True, alpha=0.3, axis='y')

# 4. % de estanques con problemas (promedio)
ax = axes[1, 1]
pct_promedios = {
    't-1': df_resultados['t_1_pct_problemas'].mean(),
    't': df_resultados['t_pct_problemas'].mean(),
    't+1': df_resultados['t_plus_pct_problemas'].mean()
}
bars3 = ax.bar(pct_promedios.keys(), pct_promedios.values(),
               color=['orange', 'red', 'purple'], alpha=0.7,
               edgecolor='black', linewidth=1.5)
ax.axhline(y=15, color='red', linestyle='--', linewidth=2, alpha=0.7,
           label='Umbral pérdida (15%)')
ax.set_ylabel('% Promedio de estanques con problemas')
ax.set_title('Promedio de Estanques Afectados por Momento')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bar, pct in zip(bars3, pct_promedios.values()):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{pct:.1f}%',
            ha='center', va='bottom', fontweight='bold')

# 5. Scatter: ΔVolumen vs % problemas en t
ax = axes[2, 0]
colors_scatter = ['red' if row['tiene_perdida_ventana'] else 'blue' 
                  for _, row in df_resultados.iterrows()]
ax.scatter(df_resultados['t_pct_problemas'], 
           df_resultados['Delta_Volumen'],
           c=colors_scatter, alpha=0.6, s=80, 
           edgecolors='black', linewidth=0.5)
ax.axvline(x=15, color='red', linestyle='--', linewidth=2, alpha=0.5,
           label='Umbral pérdida (15%)')
ax.set_xlabel('% Estanques con problemas en t')
ax.set_ylabel('ΔVolumen (m³/hr)')
ax.set_title('Relación: % Problemas vs ΔVolumen')
ax.legend(['Con pérdida', 'Sin pérdida', 'Umbral'], loc='lower left')
ax.grid(True, alpha=0.3)

# 6. Serie temporal
ax = axes[2, 1]
df_plot = df_resultados.sort_values('timestamp')
colors_time = ['red' if tiene else 'blue' 
               for tiene in df_plot['tiene_perdida_ventana']]
ax.scatter(df_plot['timestamp'], df_plot['Delta_Volumen'],
           c=colors_time, alpha=0.6, s=60, 
           edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='green', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Fecha')
ax.set_ylabel('ΔVolumen (m³/hr)')
ax.set_title('Serie Temporal: ΔVolumen y Pérdidas de Enlace')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

# Leyenda manual
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', alpha=0.6, label='Con pérdida enlace'),
                   Patch(facecolor='blue', alpha=0.6, label='Sin pérdida enlace')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('outputs/analisis_perdidas_enlace_outliers.png', 
            dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado: outputs/analisis_perdidas_enlace_outliers.png")

# Guardar resultados detallados
df_resultados_export = df_resultados.sort_values('Delta_Volumen')
df_resultados_export.to_csv('outputs/outliers_con_analisis_perdidas_enlace.csv', 
                            index=False)
print("✅ Resultados guardados: "
      "outputs/outliers_con_analisis_perdidas_enlace.csv")

# ============================================================================
# 6. CONCLUSIONES
# ============================================================================

print("\n" + "=" * 80)
print("🔍 CONCLUSIONES")
print("=" * 80)

print(f"\n1️⃣ PREVALENCIA DE PÉRDIDAS DE ENLACE:")
print(f"   {n_con_perdida}/{len(df_resultados)} outliers "
      f"({n_con_perdida/len(df_resultados)*100:.1f}%) tienen pérdida en ventana")

if n_con_perdida / len(df_resultados) > 0.7:
    print(f"   ✅ MAYORÍA de outliers asociados a pérdidas de enlace")
elif n_con_perdida / len(df_resultados) > 0.4:
    print(f"   ⚠️  MUCHOS outliers asociados a pérdidas")
else:
    print(f"   ❌ POCOS outliers asociados a pérdidas")

print(f"\n2️⃣ PATRÓN DOMINANTE:")
if n_patron_recuperacion > n_patron_perdida and n_patron_recuperacion > 10:
    print(f"   ✅ PATRÓN RECUPERACIÓN dominante ({n_patron_recuperacion} casos)")
    print(f"   → Pérdida en t-1, recuperación en t")
    print(f"   → ΔVolumen negativo artificial por salto de 0 a valor real")
elif n_patron_perdida > 10:
    print(f"   ⚠️  PATRÓN PÉRDIDA significativo ({n_patron_perdida} casos)")
else:
    print(f"   ℹ️  Sin patrón dominante claro")

print(f"\n3️⃣ MAGNITUD DEL EFECTO:")
if abs(delta_con_perdida) > abs(delta_sin_perdida) * 1.5:
    print(f"   ✅ Pérdidas causan ΔVol MÁS extremos")
    print(f"   → Con pérdida: {delta_con_perdida:,.0f} m³/hr")
    print(f"   → Sin pérdida: {delta_sin_perdida:,.0f} m³/hr")
else:
    print(f"   ⚠️  Efecto menos claro")

print(f"\n💡 RECOMENDACIÓN FINAL:")

if n_con_perdida / len(df_resultados) > 0.6:
    print(f"""
   ✅ HIPÓTESIS VALIDADA: Pérdidas de enlace causan outliers extremos
   
   MECANISMO IDENTIFICADO:
   1. Pérdida de enlace → Estanques reportan 0 o null
   2. Volumen total aparenta ser bajo
   3. Recuperación de enlace → Volumen "salta" a valor real
   4. ΔVolumen calculado es muy negativo (artificial)
   5. Demanda = Qin - ΔVolumen aparece extremadamente alta
   
   ACCIONES:
   1. ✅ Estos {n_con_perdida} outliers SON errores de telemetría
   2. ✅ MANTENER filtrado de outliers >30,000 m³/hr
   3. ✅ Implementar detección de pérdida de enlace en tiempo real
   4. ✅ Cuando >15% estanques = 0/null → marcar como dato inválido
   5. ✅ NO usar para entrenamiento del modelo
   
   IMPACTO:
   • Confirma que outliers extremos NO son demanda real
   • Son artefactos de problemas de comunicación
   • Filtrado actual es correcto y necesario
    """)
elif n_con_perdida > 0:
    print(f"""
   ⚠️  PÉRDIDAS PARCIALMENTE RESPONSABLES
   
   • {n_con_perdida} casos con pérdidas
   • {len(df_resultados) - n_con_perdida} casos sin pérdidas claras
   
   ACCIONES:
   1. Separar outliers: con vs sin pérdida enlace
   2. Filtrar los {n_con_perdida} con pérdidas (errores)
   3. Investigar {len(df_resultados) - n_con_perdida} restantes
    """)
else:
    print(f"""
   ❌ PÉRDIDAS NO EXPLICAN LOS OUTLIERS
   
   • Mayoría de outliers sin pérdida de enlace detectada
   • Buscar otras causas (errores Qin, ΔVol, etc.)
    """)

print("\n" + "=" * 80)
print("✅ VERIFICACIÓN COMPLETADA")
print("=" * 80)
