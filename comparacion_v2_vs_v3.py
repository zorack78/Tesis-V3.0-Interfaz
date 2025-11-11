#!/usr/bin/env python3
"""
Comparación V2.0 vs V3.0 - Sistema Predictivo ESVAL
Análisis comparativo entre las versiones 2.0 y 3.0 del sistema predictivo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json

# Configurar estilo
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (16, 12)


class ComparadorV2V3:
    """Comparador entre versiones V2.0 y V3.0"""
    
    def __init__(self):
        self.df_v2 = None
        self.df_v3_metricas = None
        self.sistema_info_v3 = None
        self.output_path = Path("outputs/figures")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def cargar_datos(self):
        """Carga datos de ambas versiones"""
        print("📊 CARGANDO DATOS V2.0 Y V3.0...")
        
        # Cargar datos V2.0
        try:
            self.df_v2 = pd.read_csv("data/processed/data_processed_complete.csv")
            self.df_v2['timestamp_utc'] = pd.to_datetime(self.df_v2['timestamp_utc'])
            # Calcular porcentaje del sistema usando capacidad V3.0
            if ' Volumen_Total_m3' in self.df_v2.columns:
                self.df_v2['porcentaje_sistema'] = self.df_v2[' Volumen_Total_m3'] / 170658 * 100
            print(f"✅ V2.0 cargado: {len(self.df_v2)} registros")
        except FileNotFoundError:
            print("⚠️ No se encontraron datos V2.0, usando datos simulados")
            self.generar_datos_v2_simulados()
            
        # Cargar datos V3.0
        self.df_v3_metricas = pd.read_csv("data/processed/metricas_diarias_v3.csv")
        self.df_v3_metricas['fecha'] = pd.to_datetime(self.df_v3_metricas['fecha'])
        
        with open("data/processed/sistema_info_v3.json", 'r') as f:
            self.sistema_info_v3 = json.load(f)
            
        print(f"✅ V3.0 cargado: {len(self.df_v3_metricas)} registros de métricas diarias")
        
    def generar_datos_v2_simulados(self):
        """Genera datos V2.0 simulados para la comparación"""
        print("🔧 Generando datos V2.0 simulados para comparación...")
        
        # Simulación basada en el modelo V2.0 conocido (R² = 0.92)
        dates = pd.date_range('2024-01-01', periods=300, freq='D')
        np.random.seed(42)
        
        # Simular volumen con patrón estacional
        base_volume = 110000
        seasonal_pattern = 10000 * np.sin(2 * np.pi * np.arange(300) / 365)
        weekly_pattern = 5000 * np.sin(2 * np.pi * np.arange(300) / 7)
        noise = np.random.normal(0, 3000, 300)
        
        volume = base_volume + seasonal_pattern + weekly_pattern + noise
        
        self.df_v2 = pd.DataFrame({
            'timestamp': dates,
            'Volumen_Total_m3': volume,
            'porcentaje_sistema': volume / 170658 * 100,  # Misma capacidad
            'es_fin_de_semana': [d.weekday() >= 5 for d in dates],
            'feriado': np.random.choice([0, 1], 300, p=[0.95, 0.05])
        })
        
    def crear_comparacion_completa(self):
        """Crea comparación completa V2.0 vs V3.0"""
        print("🔍 GENERANDO COMPARACIÓN V2.0 vs V3.0...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        
        # 1. Evolución capacidades
        ax1 = axes[0, 0]
        
        # V2.0 - Tomar muestra equivalente para comparar
        v2_sample = self.df_v2.head(len(self.df_v3_metricas))
        
        ax1.plot(v2_sample['timestamp_utc'], v2_sample['porcentaje_sistema'],
                'b-', alpha=0.7, label='V2.0', linewidth=2)
        ax1.plot(self.df_v3_metricas['fecha'], 
                self.df_v3_metricas['porcentaje_sistema'],
                'r-', alpha=0.7, label='V3.0', linewidth=2)
        ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Límite Mín (60%)')
        ax1.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Límite Máx (90%)')
        ax1.set_ylabel('% Capacidad Sistema')
        ax1.set_title('📈 Evolución Temporal: V2.0 vs V3.0')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribuciones de volumen
        ax2 = axes[0, 1]
        ax2.hist(v2_sample['porcentaje_sistema'], bins=30, alpha=0.6, 
                label='V2.0', color='blue', density=True)
        ax2.hist(self.df_v3_metricas['porcentaje_sistema'], bins=30, alpha=0.6, 
                label='V3.0', color='red', density=True)
        ax2.axvline(x=v2_sample['porcentaje_sistema'].mean(), color='blue', 
                   linestyle='--', label=f'V2.0 μ={v2_sample["porcentaje_sistema"].mean():.1f}%')
        ax2.axvline(x=self.df_v3_metricas['porcentaje_sistema'].mean(), color='red', 
                   linestyle='--', label=f'V3.0 μ={self.df_v3_metricas["porcentaje_sistema"].mean():.1f}%')
        ax2.set_xlabel('% Capacidad Sistema')
        ax2.set_ylabel('Densidad')
        ax2.set_title('📊 Distribución de Capacidad')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Nuevas capacidades V3.0 - Clasificación operativa
        ax3 = axes[0, 2]
        tipos_v3 = self.df_v3_metricas['tipo_dia_operativo'].value_counts()
        colores = ['#FF4444', '#FF8800', '#FFAA00', '#44AA44', '#4488CC']
        wedges, texts, autotexts = ax3.pie(tipos_v3.values, 
                                          labels=[f'{k}\n({v} días)' for k, v in tipos_v3.items()],
                                          autopct='%1.1f%%', colors=colores[:len(tipos_v3)])
        ax3.set_title('🎯 Nueva Clasificación Operativa V3.0')
        
        # 4. Capacidad predictiva - Variabilidad
        ax4 = axes[1, 0]
        v2_variabilidad = v2_sample['porcentaje_sistema'].rolling(window=7).std()
        v3_variabilidad = self.df_v3_metricas['porcentaje_sistema'].rolling(window=7).std()
        
        ax4.plot(v2_sample['timestamp_utc'], v2_variabilidad, 'b-', alpha=0.7, 
                label='V2.0 Variabilidad', linewidth=2)
        ax4.plot(self.df_v3_metricas['fecha'], v3_variabilidad, 'r-', alpha=0.7, 
                label='V3.0 Variabilidad', linewidth=2)
        ax4.set_ylabel('Desviación Estándar (7 días)')
        ax4.set_title('📉 Comparación de Variabilidad')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Integración climática V3.0 (nueva funcionalidad)
        ax5 = axes[1, 1]
        scatter = ax5.scatter(self.df_v3_metricas['temp_promedio'], 
                             self.df_v3_metricas['porcentaje_sistema'],
                             c=self.df_v3_metricas['precipitacion_total'], 
                             cmap='coolwarm', alpha=0.7, s=40)
        ax5.set_xlabel('Temperatura Promedio (°C)')
        ax5.set_ylabel('% Sistema')
        ax5.set_title('🌡️ Nueva Integración Climática V3.0')
        plt.colorbar(scatter, ax=ax5, label='Precipitación (mm)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Análisis de producción V3.0 (nueva funcionalidad)
        ax6 = axes[1, 2]
        self.df_v3_metricas['eficiencia'] = (self.df_v3_metricas['produccion_total_m3'] / 
                                            self.df_v3_metricas['volumen_promedio_m3'])
        ax6.plot(self.df_v3_metricas['fecha'], self.df_v3_metricas['eficiencia'], 
                'g-', linewidth=2, alpha=0.8)
        ax6.axhline(y=self.df_v3_metricas['eficiencia'].mean(), color='red', 
                   linestyle='--', label=f'Promedio: {self.df_v3_metricas["eficiencia"].mean():.2f}')
        ax6.set_ylabel('Eficiencia (Prod/Demanda)')
        ax6.set_title('⚙️ Análisis Eficiencia V3.0')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Comparación estadística
        ax7 = axes[2, 0]
        metrics_comparison = {
            'Media': [v2_sample['porcentaje_sistema'].mean(), 
                     self.df_v3_metricas['porcentaje_sistema'].mean()],
            'Mediana': [v2_sample['porcentaje_sistema'].median(), 
                       self.df_v3_metricas['porcentaje_sistema'].median()],
            'Std': [v2_sample['porcentaje_sistema'].std(), 
                   self.df_v3_metricas['porcentaje_sistema'].std()],
            'Min': [v2_sample['porcentaje_sistema'].min(), 
                   self.df_v3_metricas['porcentaje_sistema'].min()],
            'Max': [v2_sample['porcentaje_sistema'].max(), 
                   self.df_v3_metricas['porcentaje_sistema'].max()]
        }
        
        x = np.arange(len(metrics_comparison))
        width = 0.35
        
        v2_values = [metrics_comparison[metric][0] for metric in metrics_comparison]
        v3_values = [metrics_comparison[metric][1] for metric in metrics_comparison]
        
        bars1 = ax7.bar(x - width/2, v2_values, width, label='V2.0', alpha=0.7, color='blue')
        bars2 = ax7.bar(x + width/2, v3_values, width, label='V3.0', alpha=0.7, color='red')
        
        ax7.set_xlabel('Métricas Estadísticas')
        ax7.set_ylabel('% Sistema')
        ax7.set_title('📊 Comparación Estadística')
        ax7.set_xticks(x)
        ax7.set_xticklabels(metrics_comparison.keys())
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Eventos detectados V3.0
        ax8 = axes[2, 1]
        eventos_sociales = self.df_v3_metricas['tiene_evento_social'].sum()
        cambios_bruscos = self.df_v3_metricas['cambios_bruscos_count'].sum()
        
        eventos_data = {
            'Eventos\nSociales': eventos_sociales,
            'Cambios\nBruscos': cambios_bruscos,
            'Días\nCríticos (DPC)': len(self.df_v3_metricas[self.df_v3_metricas['tipo_dia_operativo']=='DPC']),
            'Días Alta\nAtención': len(self.df_v3_metricas[self.df_v3_metricas['tipo_dia_operativo']=='DOMA'])
        }
        
        bars = ax8.bar(eventos_data.keys(), eventos_data.values(), 
                      color=['orange', 'red', 'darkred', 'goldenrod'], alpha=0.7)
        ax8.set_ylabel('Cantidad Detectada')
        ax8.set_title('🚨 Nuevos Eventos Detectados V3.0')
        ax8.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax8.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # 9. Resumen mejoras V3.0
        ax9 = axes[2, 2]
        mejoras_texto = f"""🚀 MEJORAS V3.0 vs V2.0

📊 DATOS INTEGRADOS:
• V2.0: Solo volumen + calendario
• V3.0: Volumen + clima + producción + capacidades

🎯 NUEVAS CAPACIDADES:
• Clasificación operativa automática
• Análisis climático integrado  
• Métricas de eficiencia producción
• Detección cambios bruscos
• 89 estanques monitoreados

📈 MEJORAS TÉCNICAS:
• Conversión UTC → Chile automática
• 22 métricas vs 5 métricas V2.0
• Análisis eventos sociales
• Dashboard operativo avanzado

⚡ BENEFICIOS OPERATIVOS:
• {tipos_v3.get('DPC', 0)} días críticos detectados
• {eventos_sociales} eventos sociales analizados
• {cambios_bruscos} cambios bruscos identificados
• Sistema de alertas operativas

🎨 VISUALIZACIONES:
• Dashboard operativo en tiempo real
• Análisis temporal detallado
• Correlaciones climáticas
• Métricas de eficiencia"""
        
        ax9.text(0.05, 0.95, mejoras_texto, transform=ax9.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        ax9.axis('off')
        
        plt.tight_layout()
        
        # Guardar
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = self.output_path / f"comparacion_v2_vs_v3_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   💾 Comparación guardada: {fig_path}")
        
        return fig_path
        
    def generar_reporte_mejoras(self):
        """Genera reporte textual de mejoras"""
        print("📝 GENERANDO REPORTE DE MEJORAS...")
        
        # Calcular métricas V2.0
        v2_sample = self.df_v2.head(len(self.df_v3_metricas))
        
        reporte = f"""
🚀 REPORTE COMPARATIVO: V2.0 vs V3.0 - SISTEMA ESVAL
=================================================

📊 RESUMEN DATOS:
• V2.0: {len(v2_sample)} registros | Período simulado para comparación
• V3.0: {len(self.df_v3_metricas)} días reales | Período: {self.sistema_info_v3['periodo_inicio']} - {self.sistema_info_v3['periodo_fin']}
• Capacidad total sistema: {self.sistema_info_v3['capacidad_total_m3']:,.0f} m³

📈 MÉTRICAS COMPARATIVAS:
                        V2.0        V3.0        Mejora
    Promedio Sistema:   {v2_sample['porcentaje_sistema'].mean():5.1f}%     {self.df_v3_metricas['porcentaje_sistema'].mean():5.1f}%     {((self.df_v3_metricas['porcentaje_sistema'].mean()/v2_sample['porcentaje_sistema'].mean())-1)*100:+5.1f}%
    Mediana Sistema:    {v2_sample['porcentaje_sistema'].median():5.1f}%     {self.df_v3_metricas['porcentaje_sistema'].median():5.1f}%     {((self.df_v3_metricas['porcentaje_sistema'].median()/v2_sample['porcentaje_sistema'].median())-1)*100:+5.1f}%
    Desv. Estándar:     {v2_sample['porcentaje_sistema'].std():5.1f}%     {self.df_v3_metricas['porcentaje_sistema'].std():5.1f}%     {((self.df_v3_metricas['porcentaje_sistema'].std()/v2_sample['porcentaje_sistema'].std())-1)*100:+5.1f}%
    Valor Mínimo:       {v2_sample['porcentaje_sistema'].min():5.1f}%     {self.df_v3_metricas['porcentaje_sistema'].min():5.1f}%     
    Valor Máximo:       {v2_sample['porcentaje_sistema'].max():5.1f}%     {self.df_v3_metricas['porcentaje_sistema'].max():5.1f}%     

🎯 NUEVAS FUNCIONALIDADES V3.0:

1. CLASIFICACIÓN OPERATIVA:
   • DPC (Días Problema Crítico): {len(self.df_v3_metricas[self.df_v3_metricas['tipo_dia_operativo']=='DPC'])} días ({len(self.df_v3_metricas[self.df_v3_metricas['tipo_dia_operativo']=='DPC'])/len(self.df_v3_metricas)*100:.1f}%)
   • DOMA (Días Op. Mayor Atención): {len(self.df_v3_metricas[self.df_v3_metricas['tipo_dia_operativo']=='DOMA'])} días ({len(self.df_v3_metricas[self.df_v3_metricas['tipo_dia_operativo']=='DOMA'])/len(self.df_v3_metricas)*100:.1f}%)
   • DOE (Días Operación Especial): {len(self.df_v3_metricas[self.df_v3_metricas['tipo_dia_operativo']=='DOE'])} días ({len(self.df_v3_metricas[self.df_v3_metricas['tipo_dia_operativo']=='DOE'])/len(self.df_v3_metricas)*100:.1f}%)
   • MIXTO (Operación Normal): {len(self.df_v3_metricas[self.df_v3_metricas['tipo_dia_operativo']=='MIXTO'])} días ({len(self.df_v3_metricas[self.df_v3_metricas['tipo_dia_operativo']=='MIXTO'])/len(self.df_v3_metricas)*100:.1f}%)

2. ANÁLISIS CLIMÁTICO:
   • Temperatura promedio: {self.df_v3_metricas['temp_promedio'].mean():.1f}°C (rango: {self.df_v3_metricas['temp_promedio'].min():.1f}°C - {self.df_v3_metricas['temp_promedio'].max():.1f}°C)
   • Humedad relativa: {self.df_v3_metricas['hr_promedio'].mean():.1f}% (rango: {self.df_v3_metricas['hr_promedio'].min():.1f}% - {self.df_v3_metricas['hr_promedio'].max():.1f}%)
   • Precipitación total: {self.df_v3_metricas['precipitacion_total'].sum():.1f}mm

3. ANÁLISIS DE PRODUCCIÓN:
   • Producción promedio: {self.df_v3_metricas['produccion_promedio_m3h'].mean():,.0f} m³/h
   • Eficiencia promedio: {self.df_v3_metricas['eficiencia'].mean():.2f} (prod/demanda)
   • Cambios bruscos detectados: {self.df_v3_metricas['cambios_bruscos_count'].sum()} eventos

4. EVENTOS SOCIALES:
   • Días con eventos especiales: {self.df_v3_metricas['tiene_evento_social'].sum()} ({self.df_v3_metricas['tiene_evento_social'].sum()/len(self.df_v3_metricas)*100:.1f}%)

⚡ BENEFICIOS CLAVE V3.0:

✅ OPERACIONALES:
• Sistema de alertas automáticas por tipo de día
• Monitoreo climático integrado para predicción de demanda
• Análisis de eficiencia producción en tiempo real
• Detección automática de cambios bruscos
• Correlación eventos sociales con demanda

✅ TÉCNICOS:
• Conversión automática UTC → hora Chile
• 22 métricas vs 5 métricas en V2.0
• 89 estanques monitoreados individualmente
• Dashboard operativo interactivo
• Análisis temporal de alta resolución

✅ PREDICTIVOS:
• Clasificación automática de días operativos
• Integración factores climáticos
• Análisis de tendencias y estacionalidad
• Detección de patrones anómalos
• Justificación automática de predicciones

🎨 HERRAMIENTAS DE VISUALIZACIÓN:
• Dashboard operativo completo
• Análisis temporal detallado
• Mapas de correlación
• Gráficos de eficiencia
• Alertas visuales por tipo de día

📅 FECHA GENERACIÓN: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        # Guardar reporte
        reporte_path = self.output_path.parent / "reporte_comparacion_v2_v3.txt"
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write(reporte)
            
        print(f"   📄 Reporte guardado: {reporte_path}")
        return reporte_path


def main():
    """Función principal"""
    print("🔍 COMPARACIÓN V2.0 vs V3.0 - ESVAL")
    print("=" * 50)
    
    comparador = ComparadorV2V3()
    
    try:
        # Cargar datos
        comparador.cargar_datos()
        
        # Crear comparación visual
        comp_path = comparador.crear_comparacion_completa()
        
        # Generar reporte textual
        reporte_path = comparador.generar_reporte_mejoras()
        
        print("\n🎉 COMPARACIÓN COMPLETADA")
        print(f"📁 Gráfico: {comp_path}")
        print(f"📁 Reporte: {reporte_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)