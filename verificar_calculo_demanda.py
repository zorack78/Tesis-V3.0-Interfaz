"""
Verificación del Cálculo de Demanda - Análisis de Balance de Masa
==================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos
BASE_DIR = 'c:/Users/socce/Downloads/rafa/Tesis3.0-Interfaz'
DATA_DIR = f'{BASE_DIR}/data/raw'

df_qin = pd.read_csv(f'{DATA_DIR}/BD_Qin_m3_UTC.csv')
df_qin.columns = df_qin.columns.str.strip()
df_qin['timestamp'] = pd.to_datetime(df_qin['timestamp'])

df_qnet = pd.read_csv(f'{DATA_DIR}/BD_Q_net_x_Hr_m3h_LIMPIO.csv')
df_qnet.columns = df_qnet.columns.str.strip()
df_qnet['timestamp'] = pd.to_datetime(df_qnet['timestamp'])

# Merge
df = df_qin.merge(df_qnet, on='timestamp', how='inner')

# Calcular demanda con ambas fórmulas
df['Demanda_correcta'] = df['Qin'] - df['Q_net_m3h']  # Fórmula correcta: Qout = Qin - Q_net
df['Demanda_incorrecta'] = df['Q_net_m3h'] - df['Qin']  # Propuesta inicial incorrecta

print("="*80)
print("VERIFICACIÓN DEL CÁLCULO DE DEMANDA - ANÁLISIS DE BALANCE DE MASA")
print("="*80)
print()

# Mostrar ejemplos
print("📊 EJEMPLOS DE CÁLCULO (primeras 10 horas):")
print()
print(df[['timestamp', 'Qin', 'Q_net_m3h', 'Demanda_correcta', 'Demanda_incorrecta']].head(10).to_string())
print()

# Estadísticas
print("\n📈 ESTADÍSTICAS COMPARATIVAS:")
print()
print(f"Fórmula CORRECTA: Demanda = Qin - Q_net")
print(f"  Media: {df['Demanda_correcta'].mean():,.1f} m³/h")
print(f"  Min:   {df['Demanda_correcta'].min():,.1f} m³/h")
print(f"  Max:   {df['Demanda_correcta'].max():,.1f} m³/h")
print(f"  Valores negativos: {(df['Demanda_correcta'] < 0).sum()} ({(df['Demanda_correcta'] < 0).sum()/len(df)*100:.2f}%)")
print()

print(f"Fórmula INCORRECTA: Demanda = Q_net - Qin")
print(f"  Media: {df['Demanda_incorrecta'].mean():,.1f} m³/h")
print(f"  Min:   {df['Demanda_incorrecta'].min():,.1f} m³/h")
print(f"  Max:   {df['Demanda_incorrecta'].max():,.1f} m³/h")
print(f"  Valores negativos: {(df['Demanda_incorrecta'] < 0).sum()} ({(df['Demanda_incorrecta'] < 0).sum()/len(df)*100:.2f}%)")
print()

# Análisis físico
print("\n🔬 ANÁLISIS FÍSICO:")
print()
print("Balance de masa: Qin - Demanda = ΔV/Δt")
print()

# Ejemplo con Q_net negativo (estanques vaciándose)
ejemplo1 = df[df['Q_net_m3h'] < -5000].iloc[0]
print("CASO 1: Q_net NEGATIVO (estanques vaciándose)")
print(f"  Timestamp: {ejemplo1['timestamp']}")
print(f"  Qin = {ejemplo1['Qin']:,.1f} m³/h (producción)")
print(f"  Q_net = {ejemplo1['Q_net_m3h']:,.1f} m³/h (ΔV/Δt < 0, volumen disminuye)")
print(f"  → Demanda correcta = {ejemplo1['Qin']} - ({ejemplo1['Q_net_m3h']:.1f}) = {ejemplo1['Demanda_correcta']:,.1f} m³/h ✅")
print(f"  → Demanda incorrecta = {ejemplo1['Q_net_m3h']:.1f} - {ejemplo1['Qin']:,.1f} = {ejemplo1['Demanda_incorrecta']:,.1f} m³/h ❌ (negativo!)")
print(f"  Interpretación: Salió {ejemplo1['Demanda_correcta']:,.1f} m³/h (producción + descarga de estanques)")
print()

# Ejemplo con Q_net positivo (estanques llenándose)
ejemplo2 = df[df['Q_net_m3h'] > 5000].iloc[0]
print("CASO 2: Q_net POSITIVO (estanques llenándose)")
print(f"  Timestamp: {ejemplo2['timestamp']}")
print(f"  Qin = {ejemplo2['Qin']:,.1f} m³/h (producción)")
print(f"  Q_net = {ejemplo2['Q_net_m3h']:,.1f} m³/h (ΔV/Δt > 0, volumen aumenta)")
print(f"  → Demanda correcta = {ejemplo2['Qin']:,.1f} - {ejemplo2['Q_net_m3h']:,.1f} = {ejemplo2['Demanda_correcta']:,.1f} m³/h ✅")
print(f"  → Demanda incorrecta = {ejemplo2['Q_net_m3h']:,.1f} - {ejemplo2['Qin']:,.1f} = {ejemplo2['Demanda_incorrecta']:,.1f} m³/h ❌")
print(f"  Interpretación: Salió {ejemplo2['Demanda_correcta']:,.1f} m³/h (parte de producción se quedó en estanques)")
print()

# Gráfico comparativo
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Gráfico 1: Ambas fórmulas
df_plot = df.iloc[:168]  # Primera semana
axes[0].plot(df_plot['timestamp'], df_plot['Demanda_correcta'], 
            color='green', linewidth=2, label='CORRECTA: Demanda = Qin - Q_net', alpha=0.8)
axes[0].plot(df_plot['timestamp'], df_plot['Demanda_incorrecta'], 
            color='red', linewidth=2, label='INCORRECTA: Demanda = Q_net - Qin', alpha=0.8, linestyle='--')
axes[0].axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
axes[0].set_xlabel('Fecha', fontweight='bold')
axes[0].set_ylabel('Demanda (m³/h)', fontweight='bold')
axes[0].set_title('Comparación de Fórmulas de Cálculo de Demanda (Primera Semana)', fontweight='bold', fontsize=13)
axes[0].legend(loc='upper right', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Componentes del balance
axes[1].plot(df_plot['timestamp'], df_plot['Qin'], 
            color='blue', linewidth=2, label='Qin (Producción)', alpha=0.8)
axes[1].plot(df_plot['timestamp'], df_plot['Q_net_m3h'], 
            color='orange', linewidth=2, label='Q_net (ΔV/Δt)', alpha=0.8)
axes[1].plot(df_plot['timestamp'], df_plot['Demanda_correcta'], 
            color='green', linewidth=2, label='Demanda = Qin - Q_net', alpha=0.8)
axes[1].axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
axes[1].set_xlabel('Fecha', fontweight='bold')
axes[1].set_ylabel('Caudal (m³/h)', fontweight='bold')
axes[1].set_title('Balance de Masa: Qin - Demanda = ΔV/Δt', fontweight='bold', fontsize=13)
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(True, alpha=0.3)

# Gráfico 3: Distribución
axes[2].hist(df['Demanda_correcta'].dropna(), bins=50, alpha=0.7, color='green', 
            edgecolor='black', label='CORRECTA: Todos valores positivos')
axes[2].hist(df['Demanda_incorrecta'].dropna(), bins=50, alpha=0.7, color='red', 
            edgecolor='black', label='INCORRECTA: Mayoría negativa')
axes[2].axvline(0, color='black', linestyle='--', linewidth=2)
axes[2].set_xlabel('Demanda (m³/h)', fontweight='bold')
axes[2].set_ylabel('Frecuencia', fontweight='bold')
axes[2].set_title('Distribución de Valores de Demanda', fontweight='bold', fontsize=13)
axes[2].legend(loc='upper right', fontsize=10)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{BASE_DIR}/outputs/verificacion_calculo_demanda.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{BASE_DIR}/outputs/verificacion_calculo_demanda.pdf', bbox_inches='tight')
print("\n📁 Gráfico guardado: outputs/verificacion_calculo_demanda.png/.pdf")

print("\n" + "="*80)
print("✅ CONCLUSIÓN: La fórmula CORRECTA es Demanda = Qin - Q_net")
print("="*80)
print()
print("JUSTIFICACIÓN:")
print("1. Balance de masa: Qin - Demanda = ΔV/Δt")
print("2. Despejando: Demanda = Qin - ΔV/Δt = Qin - Q_net")
print("3. Con esta fórmula, la demanda es siempre positiva (físicamente correcto)")
print("4. Cuando Q_net < 0 (vaciado), Demanda = Qin + |Q_net| (alta demanda)")
print("5. Cuando Q_net > 0 (llenado), Demanda = Qin - Q_net (baja demanda)")
print()
print("="*80)
