#!/usr/bin/env python3
"""
Visualizaciones Avanzadas V3.0 - Sistema Predictivo ESVAL
Genera gráficos detallados del análisis integral V3.0 con métricas operativas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns
from pathlib import Path
import json

# Configurar estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


class VisualizadorAvanzadoV3:
    """Generador de visualizaciones avanzadas para análisis V3.0"""
    
    def __init__(self):
        self.df_metricas = None
        self.df_volumen = None
        self.df_produccion = None
        self.df_clima = None
        self.sistema_info = None
        self.output_path = Path("outputs/figures")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def cargar_datos(self):
        """Carga todos los datos procesados V3.0"""
        print("📊 CARGANDO DATOS PROCESADOS V3.0...")
        
        data_path = Path("data/processed")
        
        # Cargar métricas diarias
        self.df_metricas = pd.read_csv(data_path / "metricas_diarias_v3.csv")
        self.df_metricas['fecha'] = pd.to_datetime(self.df_metricas['fecha'])
        
        # Cargar datos temporales
        self.df_volumen = pd.read_csv(data_path / "volumen_total_chile_v3.csv")
        self.df_volumen['timestamp_chile'] = pd.to_datetime(self.df_volumen['timestamp_chile'])
        
        self.df_produccion = pd.read_csv(data_path / "produccion_chile_v3.csv")
        self.df_produccion['timestamp_chile'] = pd.to_datetime(self.df_produccion['timestamp_chile'])
        
        self.df_clima = pd.read_csv(data_path / "clima_chile_v3.csv")
        self.df_clima['timestamp_chile'] = pd.to_datetime(self.df_clima['timestamp_chile'])
        
        # Cargar info del sistema
        with open(data_path / "sistema_info_v3.json", 'r') as f:
            self.sistema_info = json.load(f)
            
        print(f"✅ Datos cargados:")
        print(f"   • Métricas diarias: {len(self.df_metricas)} días")
        print(f"   • Volumen horario: {len(self.df_volumen)} registros")
        print(f"   • Producción horaria: {len(self.df_produccion)} registros")
        print(f"   • Clima horario: {len(self.df_clima)} registros")
        
    def crear_dashboard_operativo(self):
        """Crea dashboard operativo completo"""
        print("📈 GENERANDO DASHBOARD OPERATIVO...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Evolución temporal de volumen con clasificación de días
        ax1 = plt.subplot(3, 3, 1)
        colores_tipos = {
            'DPC': '#FF4444',    # Rojo - Crítico
            'DOMA': '#FF8800',   # Naranja - Atención Alta  
            'DOMB': '#FFAA00',   # Amarillo - Atención Media
            'DOE': '#44AA44',    # Verde - Especial
            'MIXTO': '#4488CC'   # Azul - Normal
        }
        
        for tipo in colores_tipos:
            mask = self.df_metricas['tipo_dia_operativo'] == tipo
            if mask.any():
                plt.scatter(self.df_metricas[mask]['fecha'], 
                           self.df_metricas[mask]['porcentaje_sistema'],
                           c=colores_tipos[tipo], label=tipo, alpha=0.7, s=30)
        
        plt.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Límite Min (60%)')
        plt.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='Límite Max (90%)')
        plt.ylabel('% Capacidad Sistema')
        plt.title('🎯 Evolución Operativa por Tipo de Día')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Distribución de tipos operativos
        ax2 = plt.subplot(3, 3, 2)
        tipos_count = self.df_metricas['tipo_dia_operativo'].value_counts()
        wedges, texts, autotexts = plt.pie(tipos_count.values, 
                                          labels=[f'{k}\n({v} días)' for k, v in tipos_count.items()],
                                          autopct='%1.1f%%',
                                          colors=[colores_tipos.get(k, '#888888') for k in tipos_count.index])
        plt.title('📊 Distribución Días Operativos')
        
        # 3. Correlación temperatura vs demanda
        ax3 = plt.subplot(3, 3, 3)
        scatter = plt.scatter(self.df_metricas['temp_promedio'], 
                             self.df_metricas['volumen_promedio_m3'],
                             c=self.df_metricas['porcentaje_sistema'], 
                             cmap='coolwarm', alpha=0.6, s=40)
        plt.colorbar(scatter, label='% Sistema')
        plt.xlabel('Temperatura Promedio (°C)')
        plt.ylabel('Volumen Promedio (m³)')
        plt.title('🌡️ Temperatura vs Demanda')
        plt.grid(True, alpha=0.3)
        
        # 4. Producción vs demanda semanal
        ax4 = plt.subplot(3, 3, 4)
        self.df_metricas['semana'] = self.df_metricas['fecha'].dt.isocalendar().week
        weekly_stats = self.df_metricas.groupby('semana').agg({
            'produccion_promedio_m3h': 'mean',
            'volumen_promedio_m3': 'mean'
        }).reset_index()
        
        plt.plot(weekly_stats['semana'], weekly_stats['produccion_promedio_m3h'], 
                'b-', label='Producción', linewidth=2)
        plt.plot(weekly_stats['semana'], weekly_stats['volumen_promedio_m3']/10, 
                'r-', label='Demanda (/10)', linewidth=2)
        plt.xlabel('Semana del Año')
        plt.ylabel('m³/h')
        plt.title('⚡ Producción vs Demanda Semanal')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Eventos climáticos extremos
        ax5 = plt.subplot(3, 3, 5)
        temp_extrema = self.df_metricas[
            (self.df_metricas['temp_max'] > 30) | (self.df_metricas['temp_min'] < 2)
        ]
        lluvia_extrema = self.df_metricas[self.df_metricas['precipitacion_total'] > 5]
        
        plt.scatter(temp_extrema['fecha'], temp_extrema['porcentaje_sistema'],
                   color='red', alpha=0.7, s=40, label=f'Temp Extrema ({len(temp_extrema)})')
        plt.scatter(lluvia_extrema['fecha'], lluvia_extrema['porcentaje_sistema'],
                   color='blue', alpha=0.7, s=40, label=f'Lluvia Alta ({len(lluvia_extrema)})')
        plt.ylabel('% Sistema')
        plt.title('🌈 Eventos Climáticos Extremos')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Eficiencia sistema (producción/demanda)
        ax6 = plt.subplot(3, 3, 6)
        self.df_metricas['eficiencia'] = (self.df_metricas['produccion_total_m3'] / 
                                         self.df_metricas['volumen_promedio_m3'])
        eficiencia_por_tipo = self.df_metricas.groupby('tipo_dia_operativo')['eficiencia'].mean()
        
        bars = plt.bar(eficiencia_por_tipo.index, eficiencia_por_tipo.values,
                      color=[colores_tipos.get(k, '#888888') for k in eficiencia_por_tipo.index])
        plt.ylabel('Ratio Producción/Demanda')
        plt.title('⚙️ Eficiencia por Tipo Día')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 7. Variabilidad mensual
        ax7 = plt.subplot(3, 3, 7)
        self.df_metricas['mes'] = self.df_metricas['fecha'].dt.month
        monthly_stats = self.df_metricas.groupby('mes').agg({
            'porcentaje_sistema': ['mean', 'std'],
            'temp_promedio': 'mean'
        })
        
        months = monthly_stats.index
        y = monthly_stats[('porcentaje_sistema', 'mean')]
        yerr = monthly_stats[('porcentaje_sistema', 'std')]
        
        plt.errorbar(months, y, yerr=yerr, fmt='o-', capsize=5, linewidth=2)
        plt.xlabel('Mes')
        plt.ylabel('% Sistema (promedio ± std)')
        plt.title('📅 Variabilidad Mensual')
        plt.grid(True, alpha=0.3)
        
        # 8. Cambios bruscos por día semana
        ax8 = plt.subplot(3, 3, 8)
        cambios_semana = self.df_metricas.groupby('dia_semana')['cambios_bruscos_count'].sum()
        dias_nombres = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']
        
        plt.bar(range(len(cambios_semana)), cambios_semana.values, 
               color='orange', alpha=0.7)
        plt.xlabel('Día Semana')
        plt.ylabel('Total Cambios Bruscos')
        plt.title('🚨 Cambios Bruscos por Día')
        plt.xticks(range(len(dias_nombres)), dias_nombres)
        plt.grid(True, alpha=0.3)
        
        # 9. Resumen estadístico
        ax9 = plt.subplot(3, 3, 9)
        stats_text = f"""📊 ESTADÍSTICAS SISTEMA V3.0
        
Capacidad Total: {self.sistema_info['capacidad_total_m3']:,.0f} m³
Días Analizados: {self.sistema_info['total_dias_analizados']}
Período: {self.sistema_info['periodo_inicio']} - {self.sistema_info['periodo_fin']}

📈 MÉTRICAS OPERATIVAS:
• Vol. Promedio: {self.df_metricas['volumen_promedio_m3'].mean():,.0f} m³
• % Sistema Prom: {self.df_metricas['porcentaje_sistema'].mean():.1f}%
• Prod. Promedia: {self.df_metricas['produccion_promedio_m3h'].mean():,.0f} m³/h
• Temp Promedio: {self.df_metricas['temp_promedio'].mean():.1f}°C

🎯 DÍAS CRÍTICOS:
• DPC: {tipos_count.get('DPC', 0)} días ({tipos_count.get('DPC', 0)/len(self.df_metricas)*100:.1f}%)
• DOMA: {tipos_count.get('DOMA', 0)} días ({tipos_count.get('DOMA', 0)/len(self.df_metricas)*100:.1f}%)
• DOE: {tipos_count.get('DOE', 0)} días ({tipos_count.get('DOE', 0)/len(self.df_metricas)*100:.1f}%)
"""
        plt.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        plt.axis('off')
        
        plt.tight_layout()
        
        # Guardar
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = self.output_path / f"dashboard_operativo_v3_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   💾 Dashboard guardado: {fig_path}")
        
        return fig_path
        
    def crear_analisis_temporal_detallado(self):
        """Crea análisis temporal de alta resolución"""
        print("⏰ GENERANDO ANÁLISIS TEMPORAL DETALLADO...")
        
        fig, axes = plt.subplots(4, 2, figsize=(18, 16))
        
        # 1. Serie temporal volumen con eventos
        ax1 = axes[0, 0]
        # Tomar muestra para visualización (últimos 30 días)
        sample_vol = self.df_volumen.tail(24*30)  # 30 días de datos horarios
        
        ax1.plot(sample_vol['timestamp_chile'], sample_vol[' Volumen_Total_m3'], 
                'b-', alpha=0.7, linewidth=1)
        ax1.axhline(y=self.sistema_info['limite_minimo_60pct'], color='red', 
                   linestyle='--', alpha=0.7, label='Límite Mín')
        ax1.axhline(y=self.sistema_info['limite_maximo_90pct'], color='orange', 
                   linestyle='--', alpha=0.7, label='Límite Máx')
        ax1.set_ylabel('Volumen (m³)')
        ax1.set_title('📊 Serie Temporal Volumen (Últimos 30 días)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Patrones horarios promedio por tipo día
        ax2 = axes[0, 1]
        # Crear datos horarios simulados basados en métricas diarias
        for tipo in self.df_metricas['tipo_dia_operativo'].unique():
            mask = self.df_metricas['tipo_dia_operativo'] == tipo
            if mask.any():
                vol_promedio = self.df_metricas[mask]['volumen_promedio_m3'].mean()
                # Simular patrón horario típico
                horas = range(24)
                patron = [vol_promedio * (1 + 0.3 * np.sin((h-6)*np.pi/12)) for h in horas]
                ax2.plot(horas, patron, label=tipo, linewidth=2, marker='o', markersize=4)
        
        ax2.set_xlabel('Hora del Día')
        ax2.set_ylabel('Volumen Estimado (m³)')
        ax2.set_title('⏰ Patrones Horarios por Tipo Día')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Correlaciones climáticas
        ax3 = axes[1, 0]
        scatter = ax3.scatter(self.df_metricas['temp_max'], 
                             self.df_metricas['hr_min'],
                             c=self.df_metricas['volumen_promedio_m3'],
                             cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, ax=ax3, label='Volumen (m³)')
        ax3.set_xlabel('Temperatura Máxima (°C)')
        ax3.set_ylabel('Humedad Mínima (%)')
        ax3.set_title('🌡️ Correlación Clima-Demanda')
        ax3.grid(True, alpha=0.3)
        
        # 4. Análisis de tendencias semanales
        ax4 = axes[1, 1]
        self.df_metricas['dia_semana_nombre'] = self.df_metricas['fecha'].dt.day_name()
        orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        orden_dias_es = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']
        
        weekly_pattern = []
        for dia in orden_dias:
            mask = self.df_metricas['dia_semana_nombre'] == dia
            if mask.any():
                weekly_pattern.append(self.df_metricas[mask]['porcentaje_sistema'].mean())
            else:
                weekly_pattern.append(0)
        
        ax4.bar(orden_dias_es, weekly_pattern, color='steelblue', alpha=0.7)
        ax4.set_ylabel('% Sistema Promedio')
        ax4.set_title('📅 Patrón Semanal Demanda')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 5. Distribución de cambios bruscos
        ax5 = axes[2, 0]
        ax5.hist(self.df_metricas['cambios_bruscos_count'], bins=20, 
                color='orange', alpha=0.7, edgecolor='black')
        ax5.axvline(x=self.df_metricas['cambios_bruscos_count'].mean(), 
                   color='red', linestyle='--', label=f'Promedio: {self.df_metricas["cambios_bruscos_count"].mean():.1f}')
        ax5.set_xlabel('Cambios Bruscos por Día')
        ax5.set_ylabel('Frecuencia')
        ax5.set_title('🚨 Distribución Cambios Bruscos')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Eficiencia producción temporal
        ax6 = axes[2, 1]
        self.df_metricas['mes_nombre'] = self.df_metricas['fecha'].dt.month_name()
        monthly_efficiency = self.df_metricas.groupby('mes')['eficiencia'].mean()
        meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                        'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        
        ax6.plot(range(1, len(monthly_efficiency)+1), monthly_efficiency.values, 
                'go-', linewidth=2, markersize=8)
        ax6.set_xlabel('Mes')
        ax6.set_ylabel('Eficiencia (Prod/Demanda)')
        ax6.set_title('⚙️ Eficiencia Mensual')
        ax6.set_xticks(range(1, len(monthly_efficiency)+1))
        ax6.set_xticklabels([meses_nombres[i-1] for i in monthly_efficiency.index])
        ax6.grid(True, alpha=0.3)
        
        # 7. Matriz correlación variables
        ax7 = axes[3, 0]
        variables_corr = ['volumen_promedio_m3', 'porcentaje_sistema', 'produccion_promedio_m3h',
                         'temp_promedio', 'hr_promedio', 'precipitacion_total']
        corr_matrix = self.df_metricas[variables_corr].corr()
        
        im = ax7.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax7.set_xticks(range(len(variables_corr)))
        ax7.set_yticks(range(len(variables_corr)))
        ax7.set_xticklabels([v.replace('_', '\n') for v in variables_corr], rotation=45)
        ax7.set_yticklabels([v.replace('_', '\n') for v in variables_corr])
        ax7.set_title('🔗 Matriz Correlaciones')
        
        # Añadir valores de correlación
        for i in range(len(variables_corr)):
            for j in range(len(variables_corr)):
                ax7.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha="center", va="center", color="black", fontsize=8)
        
        # 8. Resumen eventos especiales
        ax8 = axes[3, 1]
        eventos_especiales = self.df_metricas[self.df_metricas['tiene_evento_social'] == True]
        no_eventos = self.df_metricas[self.df_metricas['tiene_evento_social'] == False]
        
        ax8.boxplot([eventos_especiales['porcentaje_sistema'].values,
                    no_eventos['porcentaje_sistema'].values],
                   labels=['Con Eventos', 'Sin Eventos'])
        ax8.set_ylabel('% Sistema')
        ax8.set_title('🎉 Impacto Eventos Sociales')
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = self.output_path / f"analisis_temporal_detallado_v3_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   💾 Análisis temporal guardado: {fig_path}")
        
        return fig_path


def main():
    """Función principal"""
    print("🎨 VISUALIZACIONES AVANZADAS V3.0 - ESVAL")
    print("=" * 60)
    
    viz = VisualizadorAvanzadoV3()
    
    try:
        # Cargar datos
        viz.cargar_datos()
        
        # Crear visualizaciones
        dashboard_path = viz.crear_dashboard_operativo()
        temporal_path = viz.crear_analisis_temporal_detallado()
        
        print("\n🎉 VISUALIZACIONES COMPLETADAS")
        print(f"📁 Dashboard: {dashboard_path}")
        print(f"📁 Temporal: {temporal_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)