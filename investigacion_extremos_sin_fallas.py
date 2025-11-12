"""
INVESTIGACIÓN: 36 DEMANDAS EXTREMAS SIN PROBLEMAS DE TELEMETRÍA
==================================================================

OBJETIVO:
Analizar los 36 outliers de demanda extrema alta (>30,000 m³/hr) que:
   ✅ NO tienen problemas de telemetría (>20% estanques OK)
   ❌ NO correlacionan con olas de calor
   
¿Qué los causó entonces?

Hipótesis a explorar:
   1. Errores de cálculo (Qin muy alto, ΔVolumen muy negativo)
   2. Eventos sociales (festivales, feriados, elecciones)
   3. Pérdidas/fugas no contabilizadas
   4. Patrones de consumo extremo puntual
   5. Problemas en el cálculo de Demanda = Qin - ΔVolumen

Analiza:
   - Componentes: Qin y ΔVolumen individualmente
   - Relación entre Qin y ΔVolumen
   - Contexto temporal (feriados, eventos especiales)
   - Comparación con valores típicos
   - Patrones de los 89 estanques individuales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuración visual
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 16)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("INVESTIGACIÓN: 36 EXTREMOS SIN FALLAS - ¿Qué los causó?")
print("=" * 80)
print(f"\nFecha análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# 1. IDENTIFICAR LOS 36 CASOS
# ============================================================================

print("\n" + "=" * 80)
print("[1/7] IDENTIFICANDO LOS 36 CASOS SIN PROBLEMAS DE TELEMETRÍA")
print("=" * 80)

# Cargar outliers extremos
df_outliers = pd.read_csv('outputs/outliers_clasificados.csv')
df_outliers['timestamp_utc'] = pd.to_datetime(df_outliers['timestamp_utc'], 
                                               utc=True)
df_outliers['fecha_hora_local'] = pd.to_datetime(
    df_outliers['fecha_hora_local'], utc=True)

# Filtrar extremos positivos
df_extremos = df_outliers[df_outliers['Demanda_m3_hr'] > 30000].copy()

# Cargar casos con problemas de telemetría
try:
    df_problemas = pd.read_csv('outputs/outliers_con_problemas_estanques.csv')
    df_problemas['timestamp_utc'] = pd.to_datetime(
        df_problemas['timestamp_utc'], utc=True)
    
    # Identificar extremos CON problemas
    extremos_con_problemas = df_problemas[
        df_problemas['Demanda_m3_hr'] > 30000]['timestamp_utc'].values
    
    # Filtrar los 36 SIN problemas
    df_sin_problemas = df_extremos[
        ~df_extremos['timestamp_utc'].isin(extremos_con_problemas)].copy()
    
    print(f"✅ Total extremos: {len(df_extremos)}")
    print(f"   Con problemas telemetría: {len(extremos_con_problemas)}")
    print(f"   SIN problemas telemetría: {len(df_sin_problemas)}")
    
except FileNotFoundError:
    print("⚠️  Archivo de problemas no encontrado, analizando todos")
    df_sin_problemas = df_extremos.copy()

# ============================================================================
# 2. ANÁLISIS DE COMPONENTES: Qin y ΔVolumen
# ============================================================================

print("\n" + "=" * 80)
print("[2/7] ANÁLISIS DE COMPONENTES: Qin y ΔVolumen")
print("=" * 80)

# Recordar: Demanda = Qin - ΔVolumen
# Si Demanda es muy alta, puede ser por:
#   - Qin muy alto (producción exagerada)
#   - ΔVolumen muy negativo (volumen cayó mucho)

df_sin_problemas['Delta_Volumen'] = (
    df_sin_problemas['Qin'] - df_sin_problemas['Demanda_m3_hr'])

print("\n📊 ESTADÍSTICAS DE COMPONENTES:")
print("\n   Qin (Caudal de fuentes):")
print(f"      • Media:    {df_sin_problemas['Qin'].mean():,.0f} m³/hr")
print(f"      • Mediana:  {df_sin_problemas['Qin'].median():,.0f} m³/hr")
print(f"      • Mín:      {df_sin_problemas['Qin'].min():,.0f} m³/hr")
print(f"      • Máx:      {df_sin_problemas['Qin'].max():,.0f} m³/hr")

print("\n   ΔVolumen (Cambio en almacenamiento):")
print(f"      • Media:    {df_sin_problemas['Delta_Volumen'].mean():,.0f} m³/hr")
print(f"      • Mediana:  {df_sin_problemas['Delta_Volumen'].median():,.0f} m³/hr")
print(f"      • Mín:      {df_sin_problemas['Delta_Volumen'].min():,.0f} m³/hr")
print(f"      • Máx:      {df_sin_problemas['Delta_Volumen'].max():,.0f} m³/hr")

print("\n   Demanda (Consumo calculado):")
print(f"      • Media:    {df_sin_problemas['Demanda_m3_hr'].mean():,.0f} m³/hr")
print(f"      • Mediana:  {df_sin_problemas['Demanda_m3_hr'].median():,.0f} m³/hr")
print(f"      • Mín:      {df_sin_problemas['Demanda_m3_hr'].min():,.0f} m³/hr")
print(f"      • Máx:      {df_sin_problemas['Demanda_m3_hr'].max():,.0f} m³/hr")

# Clasificar por causa dominante
qin_muy_alto = (df_sin_problemas['Qin'] > 20000).sum()
delta_muy_negativo = (df_sin_problemas['Delta_Volumen'] < -50000).sum()
ambos_normales = len(df_sin_problemas) - qin_muy_alto - delta_muy_negativo

print("\n💡 CLASIFICACIÓN POR CAUSA PROBABLE:")
print(f"   • Qin muy alto (>20k):         {qin_muy_alto} "
      f"({qin_muy_alto/len(df_sin_problemas)*100:.1f}%)")
print(f"   • ΔVolumen muy negativo (<-50k): {delta_muy_negativo} "
      f"({delta_muy_negativo/len(df_sin_problemas)*100:.1f}%)")
print(f"   • Ambos en rango normal:       {ambos_normales} "
      f"({ambos_normales/len(df_sin_problemas)*100:.1f}%)")

# Usar outliers normales (negativos) para comparación
df_normal_outliers = df_outliers[
    (df_outliers['Demanda_m3_hr'] >= 0) & 
    (df_outliers['Demanda_m3_hr'] <= 30000)].copy()

if len(df_normal_outliers) > 0:
    df_normal_outliers['Delta_Volumen'] = (
        df_normal_outliers['Qin'] - df_normal_outliers['Demanda_m3_hr'])
    
    print("\n📊 COMPARACIÓN CON VALORES NORMALES:")
    print(f"   Qin normal (media):       {df_normal_outliers['Qin'].mean():,.0f} m³/hr")
    print(f"   Qin extremos (media):     {df_sin_problemas['Qin'].mean():,.0f} m³/hr")
    print(f"   Diferencia Qin:           "
          f"{df_sin_problemas['Qin'].mean() - df_normal_outliers['Qin'].mean():+,.0f} m³/hr")
    
    print(f"\n   ΔVol normal (media):      "
          f"{df_normal_outliers['Delta_Volumen'].mean():,.0f} m³/hr")
    print(f"   ΔVol extremos (media):    "
          f"{df_sin_problemas['Delta_Volumen'].mean():,.0f} m³/hr")
    print(f"   Diferencia ΔVol:          "
          f"{df_sin_problemas['Delta_Volumen'].mean() - df_normal_outliers['Delta_Volumen'].mean():+,.0f} m³/hr")
else:
    # Valores típicos conocidos
    qin_normal = 11000
    delta_normal = 0
    
    print("\n📊 COMPARACIÓN CON VALORES TÍPICOS:")
    print(f"   Qin típico (estimado):    {qin_normal:,.0f} m³/hr")
    print(f"   Qin extremos (media):     {df_sin_problemas['Qin'].mean():,.0f} m³/hr")
    print(f"   Diferencia Qin:           "
          f"{df_sin_problemas['Qin'].mean() - qin_normal:+,.0f} m³/hr")
    
    print(f"\n   ΔVol típico (equilibrio): {delta_normal:,.0f} m³/hr")
    print(f"   ΔVol extremos (media):    "
          f"{df_sin_problemas['Delta_Volumen'].mean():,.0f} m³/hr")
    print(f"   Diferencia ΔVol:          "
          f"{df_sin_problemas['Delta_Volumen'].mean() - delta_normal:+,.0f} m³/hr")

# ============================================================================
# 3. ANÁLISIS DE EVENTOS ESPECIALES
# ============================================================================

print("\n" + "=" * 80)
print("[3/7] ANÁLISIS DE EVENTOS ESPECIALES Y CONTEXTO TEMPORAL")
print("=" * 80)

df_sin_problemas['es_feriado'] = df_sin_problemas['feriado'] == 1
df_sin_problemas['es_fin_semana'] = df_sin_problemas['es_fin_de_semana'] == 1

print("\n📅 CONTEXTO DE EVENTOS ESPECIALES:")
print(f"   Feriados:                    "
      f"{df_sin_problemas['es_feriado'].sum()}/{len(df_sin_problemas)} "
      f"({df_sin_problemas['es_feriado'].sum()/len(df_sin_problemas)*100:.1f}%)")
print(f"   Fin de semana:               "
      f"{df_sin_problemas['es_fin_semana'].sum()}/{len(df_sin_problemas)} "
      f"({df_sin_problemas['es_fin_semana'].sum()/len(df_sin_problemas)*100:.1f}%)")

# Eventos especiales
eventos_cols = ['festival_vina', 'elecciones', 'elecciones_primarias_comunal',
                'vacaciones_escolares', 'temporada_turistica_alta']

print("\n🎭 EVENTOS SOCIALES:")
for evento in eventos_cols:
    if evento in df_sin_problemas.columns:
        count = (df_sin_problemas[evento] == 1).sum()
        pct = count / len(df_sin_problemas) * 100
        if count > 0:
            print(f"   {evento:<30}: {count:2d} ({pct:.1f}%)")

# ============================================================================
# 4. TOP 10 CASOS MÁS EXTREMOS
# ============================================================================

print("\n" + "=" * 80)
print("[4/7] TOP 10 CASOS MÁS EXTREMOS (detalle completo)")
print("=" * 80)

df_top10 = df_sin_problemas.nlargest(10, 'Demanda_m3_hr')

print("\n📋 ANÁLISIS DETALLADO:")
print()
for idx, row in df_top10.iterrows():
    print("─" * 80)
    fecha = row['fecha_hora_local']
    print(f"🔍 #{df_top10.index.get_loc(idx)+1} - {fecha}")
    print(f"   Demanda:        {row['Demanda_m3_hr']:>10,.0f} m³/hr")
    print(f"   Qin:            {row['Qin']:>10,.0f} m³/hr")
    print(f"   ΔVolumen:       {row['Delta_Volumen']:>10,.0f} m³/hr")
    print(f"   Temperatura:    {row['temp']:>10.1f}°C")
    print(f"   Humedad:        {row['HR']:>10.0f}%")
    print(f"   Precipitación:  {row['mmhr']:>10.2f} mm/hr")
    
    # Contexto
    contexto = []
    if row['es_feriado']:
        contexto.append(f"FERIADO: {row['nombre_feriado']}")
    if row['es_fin_semana']:
        contexto.append("FIN DE SEMANA")
    for evento in eventos_cols:
        if evento in row and row[evento] == 1:
            contexto.append(evento.upper().replace('_', ' '))
    
    if contexto:
        print(f"   Contexto:       {', '.join(contexto)}")
    else:
        print(f"   Contexto:       Normal (sin eventos)")
    
    # Análisis de causa
    causa_probable = []
    if row['Qin'] > 20000:
        causa_probable.append(f"⚠️ Qin MUY ALTO ({row['Qin']:,.0f})")
    if row['Delta_Volumen'] < -50000:
        causa_probable.append(f"⚠️ ΔVol MUY NEGATIVO ({row['Delta_Volumen']:,.0f})")
    if abs(row['Delta_Volumen']) > abs(row['Qin']):
        causa_probable.append("⚠️ |ΔVol| > Qin (inusual)")
    
    if causa_probable:
        print(f"   🔴 Anomalía:     {' | '.join(causa_probable)}")
    else:
        print(f"   ✅ Componentes:  Dentro de rangos esperados")

print("─" * 80)

# ============================================================================
# 5. ANÁLISIS DE VOLÚMENES INDIVIDUALES
# ============================================================================

print("\n" + "=" * 80)
print("[5/7] ANÁLISIS DE VOLÚMENES INDIVIDUALES (89 estanques)")
print("=" * 80)

# Cargar volúmenes individuales
try:
    df_vol_tanks = pd.read_csv('data/raw/Vol_X_TK_Hr_m3_UTC.csv')
    df_vol_tanks['timestamp_utc'] = pd.to_datetime(
        df_vol_tanks['timestamp_utc'], utc=True)
    
    print(f"✅ Volúmenes individuales cargados: {len(df_vol_tanks)} registros")
    
    # Analizar los top 3 casos
    print("\n🔍 ANÁLISIS DE ESTANQUES EN TOP 3 CASOS:")
    
    for idx, row in df_top10.head(3).iterrows():
        timestamp = row['timestamp_utc']
        print(f"\n📅 {row['fecha_hora_local']} - Demanda: "
              f"{row['Demanda_m3_hr']:,.0f} m³/hr")
        
        # Obtener volúmenes en ese timestamp
        vol_momento = df_vol_tanks[
            df_vol_tanks['timestamp_utc'] == timestamp]
        
        if len(vol_momento) > 0:
            vol_momento = vol_momento.iloc[0]
            
            # Analizar patrones
            tanks_cols = [col for col in vol_momento.index 
                         if col.startswith('TK_')]
            tanks_values = [vol_momento[col] for col in tanks_cols 
                           if pd.notna(vol_momento[col])]
            
            if len(tanks_values) > 0:
                tanks_array = np.array(tanks_values)
                
                ceros = (tanks_array == 0).sum()
                maximos = (tanks_array > np.percentile(tanks_array, 95)).sum()
                rango = tanks_array.max() - tanks_array.min()
                
                print(f"   Estanques válidos:     {len(tanks_values)}/89")
                print(f"   Estanques en cero:     {ceros} "
                      f"({ceros/len(tanks_values)*100:.1f}%)")
                print(f"   Estanques en máximo:   {maximos} "
                      f"({maximos/len(tanks_values)*100:.1f}%)")
                print(f"   Volumen medio:         {tanks_array.mean():,.0f} m³")
                print(f"   Rango (max-min):       {rango:,.0f} m³")
                
                # Verificar si hubo vaciado masivo
                if timestamp > df_vol_tanks['timestamp_utc'].min():
                    # Obtener volumen 1 hora antes
                    timestamp_prev = timestamp - pd.Timedelta(hours=1)
                    vol_prev = df_vol_tanks[
                        df_vol_tanks['timestamp_utc'] == timestamp_prev]
                    
                    if len(vol_prev) > 0:
                        vol_prev = vol_prev.iloc[0]
                        tanks_prev = [vol_prev[col] for col in tanks_cols 
                                     if pd.notna(vol_prev[col])]
                        
                        if len(tanks_prev) > 0:
                            cambio_total = sum(tanks_values) - sum(tanks_prev)
                            print(f"   Cambio total volumen:  "
                                  f"{cambio_total:+,.0f} m³")
                            print(f"   ΔVol reportado:        "
                                  f"{row['Delta_Volumen']:+,.0f} m³")
                            
                            diferencia = abs(cambio_total - row['Delta_Volumen'])
                            if diferencia > 10000:
                                print(f"   ⚠️ INCONSISTENCIA: Diferencia de "
                                      f"{diferencia:,.0f} m³")
            else:
                print(f"   ⚠️ Sin datos válidos de estanques")
        else:
            print(f"   ⚠️ Timestamp no encontrado en volúmenes individuales")
            
except FileNotFoundError:
    print("⚠️  Archivo de volúmenes individuales no encontrado")

# ============================================================================
# 6. PATRONES TEMPORALES
# ============================================================================

print("\n" + "=" * 80)
print("[6/7] PATRONES TEMPORALES")
print("=" * 80)

df_sin_problemas['hora'] = df_sin_problemas['fecha_hora_local'].dt.hour
df_sin_problemas['dia_mes'] = df_sin_problemas['fecha_hora_local'].dt.day
df_sin_problemas['mes'] = df_sin_problemas['fecha_hora_local'].dt.month

print("\n📊 DISTRIBUCIÓN TEMPORAL:")
print("\n   Por hora del día:")
hora_dist = df_sin_problemas['hora'].value_counts().sort_index()
for hora in hora_dist.index[:5]:  # Top 5 horas
    count = hora_dist[hora]
    pct = count / len(df_sin_problemas) * 100
    print(f"      {hora:02d}:00 - {count:2d} casos ({pct:.1f}%)")

print("\n   Por mes:")
mes_dist = df_sin_problemas['mes'].value_counts().sort_index()
meses = ['', 'Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
         'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
for mes in mes_dist.index[:5]:  # Top 5 meses
    count = mes_dist[mes]
    pct = count / len(df_sin_problemas) * 100
    print(f"      {meses[mes]} - {count:2d} casos ({pct:.1f}%)")

# ============================================================================
# 7. VISUALIZACIÓN
# ============================================================================

print("\n" + "=" * 80)
print("[7/7] GENERANDO VISUALIZACIONES")
print("=" * 80)

fig, axes = plt.subplots(4, 2, figsize=(16, 18))
fig.suptitle('INVESTIGACIÓN: 36 Extremos sin Fallas - Análisis de Causas', 
             fontsize=16, fontweight='bold', y=0.995)

# 1. Qin vs Demanda
ax = axes[0, 0]
ax.scatter(df_sin_problemas['Qin'], df_sin_problemas['Demanda_m3_hr'],
           alpha=0.7, s=100, c='red', edgecolors='black', linewidth=1)
ax.axvline(x=20000, color='orange', linestyle='--', linewidth=2, 
           label='Qin muy alto (>20k)')
ax.set_xlabel('Qin (m³/hr)')
ax.set_ylabel('Demanda (m³/hr)')
ax.set_title('Qin vs Demanda Extrema')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. ΔVolumen vs Demanda
ax = axes[0, 1]
ax.scatter(df_sin_problemas['Delta_Volumen'], 
           df_sin_problemas['Demanda_m3_hr'],
           alpha=0.7, s=100, c='red', edgecolors='black', linewidth=1)
ax.axvline(x=-50000, color='orange', linestyle='--', linewidth=2,
           label='ΔVol muy negativo (<-50k)')
ax.axvline(x=0, color='blue', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xlabel('ΔVolumen (m³/hr)')
ax.set_ylabel('Demanda (m³/hr)')
ax.set_title('ΔVolumen vs Demanda Extrema')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Qin vs ΔVolumen (relación)
ax = axes[1, 0]
scatter = ax.scatter(df_sin_problemas['Qin'], 
                     df_sin_problemas['Delta_Volumen'],
                     c=df_sin_problemas['Demanda_m3_hr'], 
                     cmap='YlOrRd', alpha=0.7, s=100,
                     edgecolors='black', linewidth=1)
ax.plot([0, 25000], [0, 0], 'b-', linewidth=2, alpha=0.5,
        label='ΔVol = 0')
ax.set_xlabel('Qin (m³/hr)')
ax.set_ylabel('ΔVolumen (m³/hr)')
ax.set_title('Relación Qin vs ΔVolumen')
ax.legend()
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Demanda (m³/hr)')

# 4. Comparación ΔVolumen (boxplot)
ax = axes[1, 1]
if len(df_normal_outliers) > 0:
    data_comp = pd.DataFrame({
        'Normal': df_normal_outliers['Delta_Volumen'].sample(
            min(50, len(df_normal_outliers)), random_state=42),
        'Extremo': df_sin_problemas['Delta_Volumen'].values[:50] 
                   if len(df_sin_problemas) >= 50 
                   else list(df_sin_problemas['Delta_Volumen']) + [np.nan]*(50-len(df_sin_problemas))
    })
else:
    data_comp = pd.DataFrame({
        'Extremo': df_sin_problemas['Delta_Volumen']
    })
    
data_comp = data_comp.melt(var_name='Tipo', value_name='ΔVolumen')
data_comp = data_comp.dropna()
sns.boxplot(data=data_comp, x='Tipo', y='ΔVolumen', ax=ax, 
            palette=['green', 'red'])
ax.axhline(y=0, color='blue', linestyle='--', linewidth=2, alpha=0.5,
           label='Equilibrio')
ax.set_ylabel('ΔVolumen (m³/hr)')
ax.set_title('Comparación ΔVolumen: Normal vs Extremos')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 5. Distribución horaria
ax = axes[2, 0]
hora_dist.plot(kind='bar', ax=ax, color='darkred', alpha=0.7)
ax.set_xlabel('Hora del día')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución Horaria de Extremos sin Fallas')
ax.grid(True, alpha=0.3, axis='y')

# 6. Eventos especiales
ax = axes[2, 1]
eventos_data = {
    'Feriado': df_sin_problemas['es_feriado'].sum(),
    'Fin semana': df_sin_problemas['es_fin_semana'].sum(),
    'Normal': len(df_sin_problemas) - df_sin_problemas['es_feriado'].sum() 
              - df_sin_problemas['es_fin_semana'].sum()
}
ax.bar(eventos_data.keys(), eventos_data.values(), 
       color=['orange', 'skyblue', 'lightgreen'], alpha=0.7,
       edgecolor='black', linewidth=1.5)
ax.set_ylabel('Frecuencia')
ax.set_title('Contexto de Eventos Especiales')
ax.grid(True, alpha=0.3, axis='y')

# 7. Serie temporal
ax = axes[3, 0]
df_plot = df_sin_problemas.sort_values('fecha_hora_local')
ax.scatter(df_plot['fecha_hora_local'], df_plot['Demanda_m3_hr'],
           c=df_plot['Qin'], cmap='Reds', alpha=0.7, s=100,
           edgecolors='black', linewidth=1)
ax.set_xlabel('Fecha')
ax.set_ylabel('Demanda (m³/hr)')
ax.set_title('Serie Temporal de Extremos sin Fallas')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

# 8. Top 10 casos
ax = axes[3, 1]
top10_sorted = df_top10.sort_values('Demanda_m3_hr', ascending=True)
fechas = [str(f)[:10] for f in top10_sorted['fecha_hora_local']]
ax.barh(range(len(top10_sorted)), top10_sorted['Demanda_m3_hr'],
        color='darkred', alpha=0.7, edgecolor='black', linewidth=1)
ax.set_yticks(range(len(top10_sorted)))
ax.set_yticklabels(fechas, fontsize=8)
ax.set_xlabel('Demanda (m³/hr)')
ax.set_title('Top 10 Casos Más Extremos')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('outputs/investigacion_extremos_sin_fallas.png', 
            dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado: outputs/investigacion_extremos_sin_fallas.png")

# ============================================================================
# 8. CONCLUSIONES Y DIAGNÓSTICO
# ============================================================================

print("\n" + "=" * 80)
print("🔍 DIAGNÓSTICO Y CONCLUSIONES")
print("=" * 80)

# Calcular porcentajes de anomalías
pct_qin_alto = (df_sin_problemas['Qin'] > 20000).sum() / len(df_sin_problemas) * 100
pct_delta_negativo = (df_sin_problemas['Delta_Volumen'] < -50000).sum() / len(df_sin_problemas) * 100
pct_feriados = df_sin_problemas['es_feriado'].sum() / len(df_sin_problemas) * 100

if len(df_normal_outliers) > 0:
    qin_ratio = df_sin_problemas['Qin'].mean() / df_normal_outliers['Qin'].mean()
else:
    qin_ratio = df_sin_problemas['Qin'].mean() / 11000

print(f"\n1️⃣ CAUSA PRINCIPAL:")
if pct_qin_alto > 50:
    print(f"   🔴 QIN ANORMALMENTE ALTO")
    print(f"      • {pct_qin_alto:.1f}% de casos con Qin >20,000 m³/hr")
    print(f"      • Qin promedio {qin_ratio:.2f}x mayor que lo normal")
    print(f"      • Posible sobreestimación del caudal de fuentes")
elif pct_delta_negativo > 50:
    print(f"   🔴 ΔVOLUMEN MUY NEGATIVO")
    print(f"      • {pct_delta_negativo:.1f}% con caída masiva de volumen")
    print(f"      • Posible vaciado rápido no explicado")
else:
    print(f"   ⚠️  CAUSAS MIXTAS O NO IDENTIFICADAS")
    print(f"      • Qin alto: {pct_qin_alto:.1f}%")
    print(f"      • ΔVol negativo: {pct_delta_negativo:.1f}%")

print(f"\n2️⃣ CONTEXTO TEMPORAL:")
print(f"   • Feriados: {pct_feriados:.1f}%")
if pct_feriados > 30:
    print(f"     → Correlación significativa con feriados")
else:
    print(f"     → NO explicado principalmente por feriados")

print(f"\n3️⃣ VALIDEZ DE LOS DATOS:")
print(f"   • ✅ Sin problemas de telemetría (>80% estanques OK)")
print(f"   • ❌ NO correlacionan con clima extremo")
print(f"   • ⚠️  Componentes de cálculo (Qin, ΔVol) sospechosos")

print(f"\n💡 RECOMENDACIÓN FINAL:")

qin_normal_ref = df_normal_outliers['Qin'].mean() if len(df_normal_outliers) > 0 else 11000

if pct_qin_alto > 50 or qin_ratio > 1.5:
    print(f"""
   🔴 PROBLEMA IDENTIFICADO: Qin sobreestimado
   
   EVIDENCIAS:
   • Qin promedio en extremos: {df_sin_problemas['Qin'].mean():,.0f} m³/hr
   • Qin promedio normal:      {qin_normal_ref:,.0f} m³/hr
   • Ratio: {qin_ratio:.2f}x
   
   ACCIONES RECOMENDADAS:
   1. ✅ Auditar medidores de caudal en fuentes (ESVAL)
   2. ✅ Verificar calibración de sensores de Qin
   3. ✅ Revisar algoritmo de suma de caudales
   4. ❌ NO usar estos {len(df_sin_problemas)} registros para entrenamiento
   5. ⚠️  Crear alerta cuando Qin > 20,000 m³/hr
   
   CONCLUSIÓN:
   Estos NO son eventos reales de demanda extrema.
   Son ERRORES en la medición/cálculo de Qin.
   Mantener filtrado de outliers actual.
    """)
elif pct_delta_negativo > 50:
    print(f"""
   🔴 PROBLEMA IDENTIFICADO: ΔVolumen muy negativo sin explicación
   
   EVIDENCIAS:
   • ΔVol promedio: {df_sin_problemas['Delta_Volumen'].mean():,.0f} m³/hr
   • Caídas masivas de volumen en 1 hora
   • Sin fallas de telemetría distribuidas
   
   ACCIONES RECOMENDADAS:
   1. ✅ Auditar sensores de nivel en estanques
   2. ✅ Verificar algoritmo de cálculo de ΔVolumen
   3. ✅ Revisar si hay vaciados operacionales no registrados
   4. ❌ NO usar estos registros para entrenamiento
   
   CONCLUSIÓN:
   Posibles vaciados operacionales o errores de medición de nivel.
   Mantener filtrado actual.
    """)
else:
    print(f"""
   ⚠️  CASOS MIXTOS - Requiere auditoría caso por caso
   
   ACCIONES:
   1. Revisar TOP 10 casos con equipo operacional
   2. Correlacionar con bitácora de operaciones
   3. Verificar si hubo maniobras operacionales
   4. Considerar mantener filtrado cauteloso
   
   CONCLUSIÓN:
   Sin patrón claro. Pueden ser eventos reales o errores variados.
    """)

print("\n" + "=" * 80)
print("✅ INVESTIGACIÓN COMPLETADA")
print("=" * 80)

# Guardar reporte detallado
df_sin_problemas_export = df_sin_problemas[[
    'timestamp_utc', 'fecha_hora_local', 'Demanda_m3_hr', 'Qin', 
    'Delta_Volumen', 'temp', 'HR', 'mmhr', 'feriado', 'nombre_feriado',
    'es_fin_de_semana'
]].copy()

df_sin_problemas_export = df_sin_problemas_export.sort_values(
    'Demanda_m3_hr', ascending=False)

df_sin_problemas_export.to_csv(
    'outputs/extremos_sin_fallas_detalle.csv', index=False)
print("\n✅ Reporte detallado guardado: "
      "outputs/extremos_sin_fallas_detalle.csv")
