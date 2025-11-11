#!/usr/bin/env python3
"""
Análisis Integral V3.0 - Sistema Predictivo ESVAL
Integra todos los datos: V2.0 + nuevos archivos climáticos, producción y capacidades
Convierte UTC a hora Chile y genera métricas operativas avanzadas
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import load_config


class ESVALDataIntegrator:
    """Integrador completo de datos ESVAL V3.0"""
    
    def __init__(self):
        self.config = load_config()
        self.chile_tz = pytz.timezone('America/Santiago')
        self.utc_tz = pytz.UTC
        
        # Dataframes principales
        self.df_volumen_total = None
        self.df_volumen_detalle = None
        self.df_capacidades = None
        self.df_produccion = None
        self.df_clima = None
        self.df_calendario_social = None
        
        # Métricas derivadas
        self.capacidad_total_sistema = 0
        self.df_metricas_diarias = None
        
    def convert_utc_to_chile(self, df, timestamp_col='timestamp_utc'):
        """Convierte timestamps de UTC a hora Chile."""
        print(f"🕐 Convirtiendo {timestamp_col} de UTC a hora Chile...")
        
        # Convertir a datetime si no lo es
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Asegurar que esté en UTC
        if df[timestamp_col].dt.tz is None:
            df[timestamp_col] = df[timestamp_col].dt.tz_localize(self.utc_tz)
        
        # Convertir a Chile
        df['timestamp_chile'] = df[timestamp_col].dt.tz_convert(self.chile_tz)
        
        return df
    
    def load_all_data(self):
        """Carga todos los archivos de datos."""
        print("📊 CARGANDO TODOS LOS ARCHIVOS DE DATOS...")
        
        data_path = Path("data/raw")
        
        # 1. Volumen total (V2.0 - ya existía)
        print("   📈 Cargando volumen total histórico (V2.0)...")
        vol_total_file = data_path / "BD_VolTotal_X_Hr_m3_UTC.csv"
        self.df_volumen_total = pd.read_csv(vol_total_file)
        self.df_volumen_total = self.convert_utc_to_chile(self.df_volumen_total, 'timestamp')
        
        # 2. Calendario social (V2.0 - ya existía)  
        print("   📅 Cargando calendario social (V2.0)...")
        calendar_file = data_path / "calendar_social_ES_COMPLETO_20240101_20250930.csv"
        self.df_calendario_social = pd.read_csv(calendar_file)
        # Usar el timestamp_utc y convertir a Chile
        self.df_calendario_social['timestamp_utc'] = pd.to_datetime(self.df_calendario_social['timestamp_utc'])
        self.df_calendario_social['timestamp_chile'] = self.df_calendario_social['timestamp_utc'].dt.tz_convert(self.chile_tz)
        
        # 3. Volumen por estanque (V3.0 - nuevo)
        print("   🏛️ Cargando volumen detallado 89 estanques (V3.0)...")
        vol_detail_file = data_path / "Vol_X_TK_Hr_m3_UTC.csv"
        self.df_volumen_detalle = pd.read_csv(vol_detail_file)
        self.df_volumen_detalle = self.convert_utc_to_chile(self.df_volumen_detalle, 'timestamp')
        
        # 4. Capacidades sistema (V3.0 - nuevo)
        print("   📐 Cargando capacidades sistema (V3.0)...")
        capacity_file = data_path / "BD_Capacidad_89Tks_m3.csv"
        self.df_capacidades = pd.read_csv(capacity_file)
        self.capacidad_total_sistema = self.df_capacidades.iloc[:, 1].sum()  # Segunda columna
        
        # 5. Producción (V3.0 - nuevo)
        print("   ⚡ Cargando datos de producción (V3.0)...")
        qin_file = data_path / "BD_Qin_m3_UTC.csv"
        self.df_produccion = pd.read_csv(qin_file)
        self.df_produccion = self.convert_utc_to_chile(self.df_produccion, 'timestamp')
        
        # 6. Clima (V3.0 - nuevo)
        print("   🌡️ Cargando datos climáticos (V3.0)...")
        clima_file = data_path / "BD_Clima2024a202509_UTC.csv"
        self.df_clima = pd.read_csv(clima_file)
        self.df_clima = self.convert_utc_to_chile(self.df_clima, 'timestamp')
        
        print(f"✅ Todos los datos cargados:")
        print(f"   • Capacidad total sistema: {self.capacidad_total_sistema:,.0f} m³")
        print(f"   • Volumen total: {len(self.df_volumen_total)} registros")
        print(f"   • Volumen detalle: {len(self.df_volumen_detalle)} registros")
        print(f"   • Producción: {len(self.df_produccion)} registros")
        print(f"   • Clima: {len(self.df_clima)} registros")
        print(f"   • Calendario: {len(self.df_calendario_social)} eventos")
    
    def calculate_system_metrics(self):
        """Calcula métricas del sistema por día."""
        print("📊 CALCULANDO MÉTRICAS DEL SISTEMA...")
        
        # Preparar datos por día
        daily_metrics = []
        
        # Obtener rango de fechas
        start_date = self.df_volumen_total['timestamp_chile'].dt.date.min()
        end_date = self.df_volumen_total['timestamp_chile'].dt.date.max()
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for fecha in date_range:
            print(f"   📅 Procesando {fecha.strftime('%Y-%m-%d')}...")
            
            fecha_str = fecha.strftime('%Y-%m-%d')
            
            # Filtrar datos del día
            day_vol = self.df_volumen_total[
                self.df_volumen_total['timestamp_chile'].dt.date == fecha.date()
            ]
            day_prod = self.df_produccion[
                self.df_produccion['timestamp_chile'].dt.date == fecha.date()
            ]
            day_clima = self.df_clima[
                self.df_clima['timestamp_chile'].dt.date == fecha.date()
            ]
            
            # Buscar eventos del día en calendario social
            eventos_dia = self.df_calendario_social[
                self.df_calendario_social['timestamp_chile'].dt.date == fecha.date()
            ]
            
            if len(day_vol) == 0:
                continue
                
            # Calcular métricas básicas
            vol_col = ' Volumen_Total_m3'  # Usar mismo nombre que V2.0
            if vol_col not in day_vol.columns:
                vol_col = day_vol.columns[1]  # Segunda columna si no existe
            
            volumen_promedio = day_vol[vol_col].mean()
            volumen_min = day_vol[vol_col].min()
            volumen_max = day_vol[vol_col].max()
            porcentaje_sistema = (volumen_promedio / self.capacidad_total_sistema) * 100
            
            # Métricas de producción
            if len(day_prod) > 0:
                prod_col = day_prod.columns[1]  # Segunda columna
                produccion_promedio = day_prod[prod_col].mean()
                produccion_total = day_prod[prod_col].sum()
                
                # Cambios de producción
                cambios_prod = np.abs(day_prod[prod_col].diff()).fillna(0)
                cambio_maximo = cambios_prod.max()
                cambios_bruscos = (cambios_prod > 900).sum()
                
                # Estabilidad de producción
                cv_produccion = day_prod[prod_col].std() / day_prod[prod_col].mean()
                horas_estables = (cambios_prod <= 50).sum()
            else:
                produccion_promedio = produccion_total = 0
                cambio_maximo = cambios_bruscos = cv_produccion = horas_estables = 0
            
            # Métricas climáticas
            if len(day_clima) > 0:
                temp_max = day_clima['temp'].max()
                temp_min = day_clima['temp'].min()
                temp_promedio = day_clima['temp'].mean()
                hr_min = day_clima['HR'].min()
                hr_promedio = day_clima['HR'].mean()
                precipitacion_total = day_clima['mmhr'].sum()
            else:
                temp_max = temp_min = temp_promedio = 0
                hr_min = hr_promedio = precipitacion_total = 0
            
            # Información de eventos
            tiene_evento = len(eventos_dia) > 0
            tipos_eventos = ', '.join(eventos_dia.columns[eventos_dia.iloc[0] == 1].tolist()) if tiene_evento else 'Ninguno'
            
            # Crear registro del día
            day_metrics = {
                'fecha': fecha_str,
                'dia_semana': fecha.strftime('%A'),
                
                # Métricas de volumen
                'volumen_promedio_m3': volumen_promedio,
                'volumen_min_m3': volumen_min,
                'volumen_max_m3': volumen_max,
                'porcentaje_sistema': porcentaje_sistema,
                'diferencia_vol_dia': volumen_max - volumen_min,
                
                # Métricas de producción
                'produccion_promedio_m3h': produccion_promedio,
                'produccion_total_m3': produccion_total,
                'cambio_maximo_prod': cambio_maximo,
                'cambios_bruscos_count': cambios_bruscos,
                'cv_produccion': cv_produccion,
                'horas_prod_estable': horas_estables,
                
                # Métricas climáticas
                'temp_max': temp_max,
                'temp_min': temp_min,
                'temp_promedio': temp_promedio,
                'hr_min': hr_min,
                'hr_promedio': hr_promedio,
                'precipitacion_total': precipitacion_total,
                
                # Eventos sociales
                'tiene_evento_social': tiene_evento,
                'tipos_eventos': tipos_eventos,
                
                # Clasificación inicial (a definir después)
                'tipo_dia_operativo': 'Por_Clasificar'
            }
            
            daily_metrics.append(day_metrics)
        
        # Crear DataFrame con métricas
        self.df_metricas_diarias = pd.DataFrame(daily_metrics)
        print(f"✅ Métricas calculadas para {len(self.df_metricas_diarias)} días")
        
        return self.df_metricas_diarias
    
    def classify_operational_days(self):
        """Clasifica los días según criterios operativos ESVAL."""
        print("🎯 CLASIFICANDO DÍAS OPERATIVOS...")
        
        if self.df_metricas_diarias is None:
            print("❌ Primero debes calcular las métricas diarias")
            return
        
        # Calcular percentiles para umbrales
        p10_vol = self.df_metricas_diarias['porcentaje_sistema'].quantile(0.1)
        p90_vol = self.df_metricas_diarias['porcentaje_sistema'].quantile(0.9)
        p10_prod = self.df_metricas_diarias['produccion_promedio_m3h'].quantile(0.1)
        p90_prod = self.df_metricas_diarias['produccion_promedio_m3h'].quantile(0.9)
        
        temp_alta = self.df_metricas_diarias['temp_max'].quantile(0.85)
        temp_baja = self.df_metricas_diarias['temp_min'].quantile(0.15)
        hr_baja = self.df_metricas_diarias['hr_min'].quantile(0.15)
        
        print(f"   📊 Umbrales calculados:")
        print(f"      • Volumen bajo: <{p10_vol:.1f}%")
        print(f"      • Volumen alto: >{p90_vol:.1f}%")
        print(f"      • Temp alta: >{temp_alta:.1f}°C")
        print(f"      • Temp baja: <{temp_baja:.1f}°C")
        print(f"      • HR baja: <{hr_baja:.1f}%")
        
        # Aplicar clasificación
        for idx, row in self.df_metricas_diarias.iterrows():
            
            # DPC - Día de Demanda Pico Crítica
            if (row['porcentaje_sistema'] < 70 and 
                row['produccion_promedio_m3h'] > p90_prod and
                row['temp_max'] > temp_alta and
                row['hr_min'] < hr_baja):
                clasificacion = 'DPC'
                
            # DMO - Día de Demanda Mínima Operativa  
            elif (row['porcentaje_sistema'] > 85 and
                  row['produccion_promedio_m3h'] < p10_prod and
                  (row['temp_max'] < temp_baja or row['precipitacion_total'] > 5)):
                clasificacion = 'DMO'
                
            # DOE - Día Operativo Estable
            elif (row['horas_prod_estable'] >= 18 and
                  row['cambios_bruscos_count'] == 0 and
                  row['cv_produccion'] < 0.15):
                clasificacion = 'DOE'
                
            # DOMA - Día Operativo Moderado Alto
            elif (p90_prod * 0.7 <= row['produccion_promedio_m3h'] <= p90_prod and
                  70 <= row['porcentaje_sistema'] <= 85):
                clasificacion = 'DOMA'
                
            # DOMB - Día Operativo Moderado Bajo  
            elif (p10_prod <= row['produccion_promedio_m3h'] <= p10_prod * 3 and
                  row['porcentaje_sistema'] > 85):
                clasificacion = 'DOMB'
                
            else:
                clasificacion = 'MIXTO'
            
            self.df_metricas_diarias.at[idx, 'tipo_dia_operativo'] = clasificacion
        
        # Estadísticas de clasificación
        clasificaciones = self.df_metricas_diarias['tipo_dia_operativo'].value_counts()
        print(f"📊 DISTRIBUCIÓN DE DÍAS:")
        for tipo, cantidad in clasificaciones.items():
            porcentaje = (cantidad / len(self.df_metricas_diarias)) * 100
            print(f"   • {tipo}: {cantidad} días ({porcentaje:.1f}%)")
    
    def save_processed_data(self):
        """Guarda todos los datos procesados."""
        print("💾 GUARDANDO DATOS PROCESADOS...")
        
        processed_path = Path("data/processed")
        processed_path.mkdir(exist_ok=True)
        
        # Guardar métricas diarias
        metrics_file = processed_path / "metricas_diarias_v3.csv"
        self.df_metricas_diarias.to_csv(metrics_file, index=False)
        print(f"   ✅ Métricas diarias: {metrics_file}")
        
        # Guardar datos convertidos a hora Chile
        vol_chile_file = processed_path / "volumen_total_chile_v3.csv"
        self.df_volumen_total.to_csv(vol_chile_file, index=False)
        print(f"   ✅ Volumen total (Chile): {vol_chile_file}")
        
        prod_chile_file = processed_path / "produccion_chile_v3.csv"
        self.df_produccion.to_csv(prod_chile_file, index=False)
        print(f"   ✅ Producción (Chile): {prod_chile_file}")
        
        clima_chile_file = processed_path / "clima_chile_v3.csv"
        self.df_clima.to_csv(clima_chile_file, index=False)
        print(f"   ✅ Clima (Chile): {clima_chile_file}")
        
        # Guardar información del sistema
        system_info = {
            'capacidad_total_m3': float(self.capacidad_total_sistema),
            'total_dias_analizados': int(len(self.df_metricas_diarias)),
            'periodo_inicio': str(self.df_metricas_diarias['fecha'].min()),
            'periodo_fin': str(self.df_metricas_diarias['fecha'].max()),
            'limite_minimo_60pct': float(self.capacidad_total_sistema * 0.6),
            'limite_maximo_90pct': float(self.capacidad_total_sistema * 0.9),
            'cambio_brusco_limite': 900,
            'cambio_maximo_sistema': 6000
        }
        
        import json
        system_file = processed_path / "sistema_info_v3.json"
        with open(system_file, 'w') as f:
            json.dump(system_info, f, indent=2)
        print(f"   ✅ Información sistema: {system_file}")
    
    def create_summary_report(self):
        """Crea reporte resumen del análisis."""
        print("📊 GENERANDO REPORTE RESUMEN...")
        
        # Crear gráfico resumen
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distribución de tipos de días
        clasificaciones = self.df_metricas_diarias['tipo_dia_operativo'].value_counts()
        ax1.pie(clasificaciones.values, labels=clasificaciones.index, autopct='%1.1f%%')
        ax1.set_title('🎯 Distribución Tipos de Días Operativos')
        
        # Evolución temporal del porcentaje del sistema
        self.df_metricas_diarias['fecha'] = pd.to_datetime(self.df_metricas_diarias['fecha'])
        ax2.plot(self.df_metricas_diarias['fecha'], 
                self.df_metricas_diarias['porcentaje_sistema'])
        ax2.axhline(y=60, color='r', linestyle='--', label='Límite mínimo (60%)')
        ax2.axhline(y=90, color='g', linestyle='--', label='Límite máximo (90%)')
        ax2.set_title('📊 Evolución Nivel del Sistema')
        ax2.set_ylabel('Porcentaje del Sistema (%)')
        ax2.legend()
        
        # Relación temperatura vs producción
        scatter = ax3.scatter(self.df_metricas_diarias['temp_max'], 
                            self.df_metricas_diarias['produccion_promedio_m3h'],
                            c=self.df_metricas_diarias['porcentaje_sistema'], 
                            cmap='viridis', alpha=0.6)
        ax3.set_xlabel('Temperatura Máxima (°C)')
        ax3.set_ylabel('Producción Promedio (m³/h)')
        ax3.set_title('🌡️ Temperatura vs Producción')
        plt.colorbar(scatter, ax=ax3, label='% Sistema')
        
        # Cambios bruscos por tipo de día
        cambios_por_tipo = self.df_metricas_diarias.groupby('tipo_dia_operativo')['cambios_bruscos_count'].sum()
        ax4.bar(cambios_por_tipo.index, cambios_por_tipo.values)
        ax4.set_title('🚨 Cambios Bruscos por Tipo de Día')
        ax4.set_ylabel('Total Cambios Bruscos')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Guardar gráfico
        output_dir = Path("outputs/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = output_dir / f"analisis_integral_v3_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   💾 Gráfico guardado: {fig_path}")
        
        plt.show()


def main():
    """Función principal de análisis integral."""
    print("🚀 ANÁLISIS INTEGRAL ESVAL V3.0 - SISTEMA COMPLETO")
    print("=" * 70)
    
    # Inicializar integrador
    integrator = ESVALDataIntegrator()
    
    # Ejecutar análisis completo
    integrator.load_all_data()
    integrator.calculate_system_metrics()
    integrator.classify_operational_days()
    integrator.save_processed_data()
    integrator.create_summary_report()
    
    print("\n🎉 ANÁLISIS INTEGRAL COMPLETADO")
    print("✅ Datos V2.0 + V3.0 integrados exitosamente")
    print("✅ Conversión UTC → Chile aplicada")
    print("✅ Métricas operativas calculadas")
    print("✅ Clasificación de días implementada")
    print("📁 Archivos procesados disponibles en data/processed/")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)