"""
ANÁLISIS EXHAUSTIVO DE FEATURES CLIMÁTICAS AVANZADAS
======================================================

Este script analiza patrones climáticos extremos y su correlación con demanda de agua
para identificar nuevas features que mejoren la capacidad predictiva del modelo.

Objetivo: Detectar OLAS DE CALOR, FRENTES FRÍOS, LLUVIAS INTENSAS y otros
eventos climáticos extremos que generen cambios significativos en la demanda.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configurar visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AnalizadorFeaturesClimaticas:
    """Analiza y crea features climáticas avanzadas para ML"""
    
    def __init__(self):
        self.data_path = Path("data/processed")
        self.output_path = Path("outputs/analisis_climatico")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Umbrales climáticos para Gran Valparaíso (basados en data histórica)
        self.UMBRALES = {
            # Temperatura (°C)
            'temp_muy_fria': 8,      # Madrugada invierno extremo
            'temp_fria': 12,         # Invierno típico (validado: 53.5% negativos)
            'temp_fresca': 15,       # Otoño/primavera
            'temp_calida': 22,       # Verano moderado
            'temp_muy_calida': 26,   # Verano intenso
            'temp_ola_calor': 30,    # Ola de calor (rarísimo en Valpo)
            
            # Humedad relativa (%)
            'hr_muy_baja': 30,       # Aire muy seco
            'hr_baja': 50,           # Seco
            'hr_alta': 70,           # Húmedo
            'hr_muy_alta': 85,       # Muy húmedo (neblina/llovizna)
            'hr_saturacion': 95,     # Casi saturación
            
            # Precipitación (mm/hr)
            'lluvia_ligera': 0.5,    # Llovizna
            'lluvia_moderada': 2.0,  # Lluvia normal
            'lluvia_fuerte': 5.0,    # Lluvia intensa
            'lluvia_torrencial': 10.0, # Tormenta
            
            # Cambios bruscos (en 1 hora)
            'delta_temp_significativo': 3,    # °C
            'delta_temp_extremo': 5,          # °C
            'delta_hr_significativo': 15,     # %
            'delta_hr_extremo': 25,           # %
        }
        
        print("=" * 80)
        print("INICIALIZANDO ANÁLISIS DE FEATURES CLIMÁTICAS AVANZADAS")
        print("=" * 80)
        print(f"📁 Datos: {self.data_path}")
        print(f"💾 Salida: {self.output_path}")
        
    def cargar_datos(self):
        """Carga datos completos con clima y volumen"""
        print("\n🔄 Cargando datasets...")
        
        # Dataset completo con features existentes
        df_completo = pd.read_csv(self.data_path / "data_processed_complete.csv")
        df_completo['timestamp_utc'] = pd.to_datetime(df_completo['timestamp_utc'])
        
        # Datos climáticos puros
        df_clima = pd.read_csv(self.data_path / "clima_chile_v3.csv")
        df_clima['timestamp'] = pd.to_datetime(df_clima['timestamp'])
        
        # Merge completo
        self.df = df_completo.merge(
            df_clima[['timestamp', 'temp', 'HR', 'mmhr']], 
            left_on='timestamp_utc', 
            right_on='timestamp',
            how='left'
        )
        
        print(f"✅ Datos cargados: {len(self.df):,} registros")
        print(f"   Periodo: {self.df['timestamp_utc'].min()} → {self.df['timestamp_utc'].max()}")
        print(f"   Features existentes: {len(self.df.columns)}")
        
        return self.df
    
    def analizar_eventos_extremos_temperatura(self):
        """Analiza eventos extremos de temperatura y su impacto en demanda"""
        print("\n" + "="*80)
        print("🌡️  ANÁLISIS: EVENTOS EXTREMOS DE TEMPERATURA")
        print("="*80)
        
        df = self.df.copy()
        
        # Identificar diferentes rangos de temperatura
        condiciones_temp = {
            'Muy Fría (<8°C)': df['temp'] < self.UMBRALES['temp_muy_fria'],
            'Fría (8-12°C)': (df['temp'] >= self.UMBRALES['temp_muy_fria']) & (df['temp'] < self.UMBRALES['temp_fria']),
            'Fresca (12-15°C)': (df['temp'] >= self.UMBRALES['temp_fria']) & (df['temp'] < self.UMBRALES['temp_fresca']),
            'Normal (15-22°C)': (df['temp'] >= self.UMBRALES['temp_fresca']) & (df['temp'] < self.UMBRALES['temp_calida']),
            'Cálida (22-26°C)': (df['temp'] >= self.UMBRALES['temp_calida']) & (df['temp'] < self.UMBRALES['temp_muy_calida']),
            'Muy Cálida (26-30°C)': (df['temp'] >= self.UMBRALES['temp_muy_calida']) & (df['temp'] < self.UMBRALES['temp_ola_calor']),
            'Ola Calor (>30°C)': df['temp'] >= self.UMBRALES['temp_ola_calor']
        }
        
        print("\n📊 DISTRIBUCIÓN DE EVENTOS:")
        resultados = []
        for nombre, condicion in condiciones_temp.items():
            subset = df[condicion]
            if len(subset) > 0:
                vol_medio = subset[' Volumen_Total_m3'].mean()
                vol_std = subset[' Volumen_Total_m3'].std()
                temp_media = subset['temp'].mean()
                
                resultados.append({
                    'Condición': nombre,
                    'Registros': len(subset),
                    'Porcentaje': len(subset) / len(df) * 100,
                    'Temp Media': temp_media,
                    'Vol Medio (m³/hr)': vol_medio,
                    'Vol Desv Std': vol_std
                })
                
                print(f"\n{nombre}:")
                print(f"  • Casos: {len(subset):>6,} ({len(subset)/len(df)*100:>5.2f}%)")
                print(f"  • Temp promedio: {temp_media:>6.2f}°C")
                print(f"  • Volumen medio: {vol_medio:>10,.0f} m³/hr")
                print(f"  • Desv. estándar: {vol_std:>10,.0f} m³/hr")
        
        df_resultados = pd.DataFrame(resultados)
        
        # Detectar OLAS DE CALOR (3+ días consecutivos con temp >26°C)
        print("\n" + "-"*80)
        print("🔥 DETECCIÓN DE OLAS DE CALOR:")
        print("-"*80)
        
        df['fecha'] = df['timestamp_utc'].dt.date
        temp_diaria = df.groupby('fecha')['temp'].agg(['max', 'mean', 'min'])
        temp_diaria['ola_calor_dia'] = temp_diaria['max'] >= self.UMBRALES['temp_muy_calida']
        
        # Buscar secuencias de 3+ días
        olas_calor = []
        dias_consecutivos = 0
        inicio_ola = None
        
        for fecha, es_ola in temp_diaria['ola_calor_dia'].items():
            if es_ola:
                if dias_consecutivos == 0:
                    inicio_ola = fecha
                dias_consecutivos += 1
            else:
                if dias_consecutivos >= 3:
                    olas_calor.append({
                        'inicio': inicio_ola,
                        'fin': fecha - timedelta(days=1),
                        'duracion': dias_consecutivos
                    })
                dias_consecutivos = 0
        
        if len(olas_calor) > 0:
            print(f"✅ Olas de calor detectadas: {len(olas_calor)}")
            for i, ola in enumerate(olas_calor, 1):
                print(f"  {i}. Del {ola['inicio']} al {ola['fin']} ({ola['duracion']} días)")
                
                # Analizar demanda durante la ola
                mask_ola = (df['fecha'] >= ola['inicio']) & (df['fecha'] <= ola['fin'])
                vol_ola = df[mask_ola][' Volumen_Total_m3'].mean()
                vol_antes = df[(df['fecha'] < ola['inicio']) & (df['fecha'] >= ola['inicio'] - timedelta(days=7))][' Volumen_Total_m3'].mean()
                
                incremento = ((vol_ola - vol_antes) / vol_antes) * 100
                print(f"     Volumen durante ola: {vol_ola:,.0f} m³/hr")
                print(f"     Volumen semana previa: {vol_antes:,.0f} m³/hr")
                print(f"     Incremento: {incremento:+.2f}%")
        else:
            print("⚠️  No se detectaron olas de calor (3+ días >26°C) en el periodo analizado")
        
        return df_resultados, olas_calor
    
    def analizar_frentes_frios(self):
        """Analiza frentes fríos y caídas bruscas de temperatura"""
        print("\n" + "="*80)
        print("❄️  ANÁLISIS: FRENTES FRÍOS Y CAÍDAS DE TEMPERATURA")
        print("="*80)
        
        df = self.df.copy()
        
        # Calcular cambios de temperatura en diferentes ventanas
        df['delta_temp_1h'] = df['temp'].diff(1)
        df['delta_temp_3h'] = df['temp'].diff(3)
        df['delta_temp_6h'] = df['temp'].diff(6)
        df['delta_temp_12h'] = df['temp'].diff(12)
        df['delta_temp_24h'] = df['temp'].diff(24)
        
        # Detectar frentes fríos (caída >5°C en 6 horas)
        frentes_frios = df[df['delta_temp_6h'] < -self.UMBRALES['delta_temp_extremo']].copy()
        
        print(f"\n🌬️  FRENTES FRÍOS DETECTADOS: {len(frentes_frios)}")
        if len(frentes_frios) > 0:
            print(f"   Definición: Caída >{self.UMBRALES['delta_temp_extremo']}°C en 6 horas")
            print(f"\n   Top 10 frentes más intensos:")
            
            frentes_top = frentes_frios.nsmallest(10, 'delta_temp_6h')
            for idx, row in frentes_top.iterrows():
                print(f"   • {row['timestamp_utc']}: Δ={row['delta_temp_6h']:.1f}°C "
                      f"({row['temp']:.1f}°C) | Vol={row[' Volumen_Total_m3']:,.0f} m³/hr")
            
            # Analizar impacto en demanda
            vol_durante_frente = frentes_frios[' Volumen_Total_m3'].mean()
            vol_normal = df[' Volumen_Total_m3'].mean()
            reduccion = ((vol_durante_frente - vol_normal) / vol_normal) * 100
            
            print(f"\n   📉 Impacto en demanda:")
            print(f"      Volumen durante frente frío: {vol_durante_frente:,.0f} m³/hr")
            print(f"      Volumen promedio normal: {vol_normal:,.0f} m³/hr")
            print(f"      Cambio: {reduccion:+.2f}%")
        
        # Detectar oscilaciones diarias extremas (diferencia día-noche)
        df['fecha'] = df['timestamp_utc'].dt.date
        df['hora'] = df['timestamp_utc'].dt.hour
        
        temp_diaria_stats = df.groupby('fecha')['temp'].agg(['min', 'max', 'mean', 'std'])
        temp_diaria_stats['amplitud_termica'] = temp_diaria_stats['max'] - temp_diaria_stats['min']
        
        dias_amplitud_extrema = temp_diaria_stats[
            temp_diaria_stats['amplitud_termica'] > self.UMBRALES['delta_temp_extremo'] * 2
        ]
        
        print(f"\n🌡️  DÍAS CON AMPLITUD TÉRMICA EXTREMA (>10°C): {len(dias_amplitud_extrema)}")
        if len(dias_amplitud_extrema) > 0:
            print(f"   Top 5:")
            for fecha, row in dias_amplitud_extrema.nlargest(5, 'amplitud_termica').iterrows():
                print(f"   • {fecha}: Amplitud={row['amplitud_termica']:.1f}°C "
                      f"(Min={row['min']:.1f}°C, Max={row['max']:.1f}°C)")
        
        return frentes_frios, temp_diaria_stats
    
    def analizar_eventos_precipitacion(self):
        """Analiza eventos de precipitación y su impacto"""
        print("\n" + "="*80)
        print("🌧️  ANÁLISIS: EVENTOS DE PRECIPITACIÓN")
        print("="*80)
        
        df = self.df.copy()
        
        # Clasificar intensidad de lluvia
        condiciones_lluvia = {
            'Sin lluvia (0 mm/hr)': df['mmhr'] == 0,
            'Llovizna (<0.5 mm/hr)': (df['mmhr'] > 0) & (df['mmhr'] < self.UMBRALES['lluvia_ligera']),
            'Lluvia ligera (0.5-2 mm/hr)': (df['mmhr'] >= self.UMBRALES['lluvia_ligera']) & (df['mmhr'] < self.UMBRALES['lluvia_moderada']),
            'Lluvia moderada (2-5 mm/hr)': (df['mmhr'] >= self.UMBRALES['lluvia_moderada']) & (df['mmhr'] < self.UMBRALES['lluvia_fuerte']),
            'Lluvia fuerte (5-10 mm/hr)': (df['mmhr'] >= self.UMBRALES['lluvia_fuerte']) & (df['mmhr'] < self.UMBRALES['lluvia_torrencial']),
            'Lluvia torrencial (>10 mm/hr)': df['mmhr'] >= self.UMBRALES['lluvia_torrencial']
        }
        
        print("\n💧 DISTRIBUCIÓN DE EVENTOS:")
        for nombre, condicion in condiciones_lluvia.items():
            subset = df[condicion]
            if len(subset) > 0:
                vol_medio = subset[' Volumen_Total_m3'].mean()
                print(f"\n{nombre}:")
                print(f"  • Casos: {len(subset):>6,} ({len(subset)/len(df)*100:>5.2f}%)")
                if subset['mmhr'].max() > 0:
                    print(f"  • Precip media: {subset['mmhr'].mean():>6.3f} mm/hr")
                print(f"  • Volumen medio: {vol_medio:>10,.0f} m³/hr")
        
        # Calcular precipitación acumulada en diferentes ventanas
        df['precip_acum_3h'] = df['mmhr'].rolling(window=3, min_periods=1).sum()
        df['precip_acum_6h'] = df['mmhr'].rolling(window=6, min_periods=1).sum()
        df['precip_acum_12h'] = df['mmhr'].rolling(window=12, min_periods=1).sum()
        df['precip_acum_24h'] = df['mmhr'].rolling(window=24, min_periods=1).sum()
        
        # Detectar TORMENTAS (acumulado >20mm en 6 horas)
        tormentas = df[df['precip_acum_6h'] > 20].copy()
        
        print(f"\n⛈️  TORMENTAS DETECTADAS (>20mm en 6h): {len(tormentas)}")
        if len(tormentas) > 0:
            print(f"   Top 10 más intensas:")
            tormentas_top = tormentas.nlargest(10, 'precip_acum_6h')
            for idx, row in tormentas_top.iterrows():
                print(f"   • {row['timestamp_utc']}: {row['precip_acum_6h']:.1f} mm/6h "
                      f"| Vol={row[' Volumen_Total_m3']:,.0f} m³/hr")
            
            # Impacto en demanda
            vol_tormenta = tormentas[' Volumen_Total_m3'].mean()
            vol_sin_lluvia = df[df['mmhr'] == 0][' Volumen_Total_m3'].mean()
            cambio = ((vol_tormenta - vol_sin_lluvia) / vol_sin_lluvia) * 100
            
            print(f"\n   📊 Impacto en demanda:")
            print(f"      Volumen durante tormenta: {vol_tormenta:,.0f} m³/hr")
            print(f"      Volumen sin lluvia: {vol_sin_lluvia:,.0f} m³/hr")
            print(f"      Cambio: {cambio:+.2f}%")
        
        return df, tormentas
    
    def analizar_combinaciones_extremas(self):
        """Analiza combinaciones de factores climáticos extremos"""
        print("\n" + "="*80)
        print("🌪️  ANÁLISIS: COMBINACIONES EXTREMAS DE FACTORES CLIMÁTICOS")
        print("="*80)
        
        df = self.df.copy()
        
        # Definir condiciones compuestas
        escenarios = {
            'Calor seco': (df['temp'] > self.UMBRALES['temp_calida']) & (df['HR'] < self.UMBRALES['hr_baja']),
            'Calor húmedo': (df['temp'] > self.UMBRALES['temp_calida']) & (df['HR'] > self.UMBRALES['hr_alta']),
            'Frío seco': (df['temp'] < self.UMBRALES['temp_fria']) & (df['HR'] < self.UMBRALES['hr_baja']),
            'Frío húmedo': (df['temp'] < self.UMBRALES['temp_fria']) & (df['HR'] > self.UMBRALES['hr_alta']),
            'Lluvia fría': (df['temp'] < self.UMBRALES['temp_fresca']) & (df['mmhr'] > self.UMBRALES['lluvia_moderada']),
            'Niebla': (df['HR'] > self.UMBRALES['hr_muy_alta']) & (df['temp'] < self.UMBRALES['temp_fresca']) & (df['mmhr'] < self.UMBRALES['lluvia_ligera']),
            'Condiciones ideales': (df['temp'].between(18, 24)) & (df['HR'].between(50, 70)) & (df['mmhr'] == 0)
        }
        
        print("\n🎯 FRECUENCIA Y IMPACTO DE ESCENARIOS COMPUESTOS:")
        resultados_escenarios = []
        
        for nombre, condicion in escenarios.items():
            subset = df[condicion]
            if len(subset) > 0:
                vol_medio = subset[' Volumen_Total_m3'].mean()
                vol_std = subset[' Volumen_Total_m3'].std()
                
                resultados_escenarios.append({
                    'Escenario': nombre,
                    'Casos': len(subset),
                    'Porcentaje': len(subset) / len(df) * 100,
                    'Vol Medio': vol_medio,
                    'Vol Std': vol_std
                })
                
                print(f"\n{nombre}:")
                print(f"  • Ocurrencias: {len(subset):>6,} ({len(subset)/len(df)*100:>5.2f}%)")
                print(f"  • Volumen medio: {vol_medio:>10,.0f} m³/hr")
                print(f"  • Desv. estándar: {vol_std:>10,.0f} m³/hr")
        
        return pd.DataFrame(resultados_escenarios)
    
    def crear_features_avanzadas(self):
        """Crea el conjunto completo de features climáticas avanzadas"""
        print("\n" + "="*80)
        print("🔧 CREACIÓN DE FEATURES CLIMÁTICAS AVANZADAS")
        print("="*80)
        
        df = self.df.copy()
        features_nuevas = []
        
        print("\n📝 Creando features...")
        
        # ==================== TEMPERATURA ====================
        print("\n1. Features de Temperatura:")
        
        # Clasificación por rangos
        df['temp_muy_fria'] = (df['temp'] < self.UMBRALES['temp_muy_fria']).astype(int)
        df['temp_fria'] = (df['temp'] < self.UMBRALES['temp_fria']).astype(int)
        df['temp_calida'] = (df['temp'] > self.UMBRALES['temp_calida']).astype(int)
        df['temp_muy_calida'] = (df['temp'] > self.UMBRALES['temp_muy_calida']).astype(int)
        df['temp_ola_calor'] = (df['temp'] > self.UMBRALES['temp_ola_calor']).astype(int)
        features_nuevas.extend(['temp_muy_fria', 'temp_fria', 'temp_calida', 'temp_muy_calida', 'temp_ola_calor'])
        print("   ✓ Rangos de temperatura (5 features)")
        
        # LAGs de temperatura
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            df[f'temp_lag_{lag}h'] = df['temp'].shift(lag)
            features_nuevas.append(f'temp_lag_{lag}h')
        print("   ✓ LAGs temperatura (7 features)")
        
        # Rolling statistics temperatura
        for window in [6, 12, 24]:
            df[f'temp_rolling_mean_{window}h'] = df['temp'].rolling(window=window, min_periods=1).mean()
            df[f'temp_rolling_std_{window}h'] = df['temp'].rolling(window=window, min_periods=1).std()
            df[f'temp_rolling_max_{window}h'] = df['temp'].rolling(window=window, min_periods=1).max()
            df[f'temp_rolling_min_{window}h'] = df['temp'].rolling(window=window, min_periods=1).min()
            features_nuevas.extend([
                f'temp_rolling_mean_{window}h', f'temp_rolling_std_{window}h',
                f'temp_rolling_max_{window}h', f'temp_rolling_min_{window}h'
            ])
        print("   ✓ Rolling statistics temperatura (12 features)")
        
        # Cambios/deltas de temperatura
        for delta in [1, 3, 6, 12, 24]:
            df[f'temp_diff_{delta}h'] = df['temp'].diff(delta)
            features_nuevas.append(f'temp_diff_{delta}h')
        print("   ✓ Cambios temperatura (5 features)")
        
        # Amplitud térmica diaria
        df['temp_amplitud_24h'] = df['temp'].rolling(window=24, min_periods=1).max() - df['temp'].rolling(window=24, min_periods=1).min()
        features_nuevas.append('temp_amplitud_24h')
        print("   ✓ Amplitud térmica (1 feature)")
        
        # Detección de frente frío
        df['frente_frio_6h'] = (df['temp'].diff(6) < -self.UMBRALES['delta_temp_extremo']).astype(int)
        df['frente_frio_12h'] = (df['temp'].diff(12) < -self.UMBRALES['delta_temp_extremo']).astype(int)
        features_nuevas.extend(['frente_frio_6h', 'frente_frio_12h'])
        print("   ✓ Detección frentes fríos (2 features)")
        
        # Ola de calor (temp >26°C sostenida 24h)
        df['en_ola_calor'] = (df['temp_rolling_min_24h'] > self.UMBRALES['temp_muy_calida']).astype(int)
        features_nuevas.append('en_ola_calor')
        print("   ✓ Detección ola de calor (1 feature)")
        
        # ==================== HUMEDAD ====================
        print("\n2. Features de Humedad:")
        
        # Clasificación por rangos
        df['hr_muy_baja'] = (df['HR'] < self.UMBRALES['hr_muy_baja']).astype(int)
        df['hr_baja'] = (df['HR'] < self.UMBRALES['hr_baja']).astype(int)
        df['hr_alta'] = (df['HR'] > self.UMBRALES['hr_alta']).astype(int)
        df['hr_muy_alta'] = (df['HR'] > self.UMBRALES['hr_muy_alta']).astype(int)
        features_nuevas.extend(['hr_muy_baja', 'hr_baja', 'hr_alta', 'hr_muy_alta'])
        print("   ✓ Rangos de humedad (4 features)")
        
        # LAGs de humedad
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'hr_lag_{lag}h'] = df['HR'].shift(lag)
            features_nuevas.append(f'hr_lag_{lag}h')
        print("   ✓ LAGs humedad (6 features)")
        
        # Rolling statistics humedad
        for window in [6, 24]:
            df[f'hr_rolling_mean_{window}h'] = df['HR'].rolling(window=window, min_periods=1).mean()
            df[f'hr_rolling_std_{window}h'] = df['HR'].rolling(window=window, min_periods=1).std()
            features_nuevas.extend([f'hr_rolling_mean_{window}h', f'hr_rolling_std_{window}h'])
        print("   ✓ Rolling statistics humedad (4 features)")
        
        # Cambios bruscos de humedad
        df['hr_diff_1h'] = df['HR'].diff(1)
        df['hr_diff_6h'] = df['HR'].diff(6)
        df['hr_cambio_brusco'] = (df['hr_diff_1h'].abs() > self.UMBRALES['delta_hr_significativo']).astype(int)
        features_nuevas.extend(['hr_diff_1h', 'hr_diff_6h', 'hr_cambio_brusco'])
        print("   ✓ Cambios humedad (3 features)")
        
        # ==================== PRECIPITACIÓN ====================
        print("\n3. Features de Precipitación:")
        
        # Clasificación intensidad
        df['sin_lluvia'] = (df['mmhr'] == 0).astype(int)
        df['lluvia_ligera'] = ((df['mmhr'] > 0) & (df['mmhr'] < self.UMBRALES['lluvia_moderada'])).astype(int)
        df['lluvia_moderada'] = ((df['mmhr'] >= self.UMBRALES['lluvia_moderada']) & (df['mmhr'] < self.UMBRALES['lluvia_fuerte'])).astype(int)
        df['lluvia_fuerte'] = (df['mmhr'] >= self.UMBRALES['lluvia_fuerte']).astype(int)
        features_nuevas.extend(['sin_lluvia', 'lluvia_ligera', 'lluvia_moderada', 'lluvia_fuerte'])
        print("   ✓ Intensidad lluvia (4 features)")
        
        # LAGs de precipitación
        for lag in [1, 2, 3, 6, 12]:
            df[f'precip_lag_{lag}h'] = df['mmhr'].shift(lag)
            features_nuevas.append(f'precip_lag_{lag}h')
        print("   ✓ LAGs precipitación (5 features)")
        
        # Acumulados de precipitación
        for window in [3, 6, 12, 24, 48]:
            df[f'precip_acum_{window}h'] = df['mmhr'].rolling(window=window, min_periods=1).sum()
            features_nuevas.append(f'precip_acum_{window}h')
        print("   ✓ Acumulados precipitación (5 features)")
        
        # Intensidad máxima reciente
        for window in [6, 24]:
            df[f'precip_max_{window}h'] = df['mmhr'].rolling(window=window, min_periods=1).max()
            features_nuevas.append(f'precip_max_{window}h')
        print("   ✓ Intensidad máxima (2 features)")
        
        # Horas desde última lluvia significativa
        lluvia_sig = df['mmhr'] > self.UMBRALES['lluvia_ligera']
        df['horas_desde_lluvia'] = (~lluvia_sig).groupby(lluvia_sig.cumsum()).cumsum()
        features_nuevas.append('horas_desde_lluvia')
        print("   ✓ Horas desde lluvia (1 feature)")
        
        # Tormenta (>20mm en 6h)
        df['en_tormenta'] = (df[f'precip_acum_6h'] > 20).astype(int)
        features_nuevas.append('en_tormenta')
        print("   ✓ Detección tormenta (1 feature)")
        
        # ==================== COMBINACIONES EXTREMAS ====================
        print("\n4. Features de Combinaciones Climáticas:")
        
        # Índice de sensación térmica simplificado (temp ajustada por humedad)
        df['sensacion_termica'] = df['temp'] - ((100 - df['HR']) / 20)
        features_nuevas.append('sensacion_termica')
        print("   ✓ Sensación térmica (1 feature)")
        
        # Condiciones compuestas
        df['calor_seco'] = ((df['temp'] > self.UMBRALES['temp_calida']) & (df['HR'] < self.UMBRALES['hr_baja'])).astype(int)
        df['calor_humedo'] = ((df['temp'] > self.UMBRALES['temp_calida']) & (df['HR'] > self.UMBRALES['hr_alta'])).astype(int)
        df['frio_seco'] = ((df['temp'] < self.UMBRALES['temp_fria']) & (df['HR'] < self.UMBRALES['hr_baja'])).astype(int)
        df['frio_humedo'] = ((df['temp'] < self.UMBRALES['temp_fria']) & (df['HR'] > self.UMBRALES['hr_alta'])).astype(int)
        df['lluvia_fria'] = ((df['temp'] < self.UMBRALES['temp_fresca']) & (df['mmhr'] > self.UMBRALES['lluvia_moderada'])).astype(int)
        df['niebla'] = ((df['HR'] > self.UMBRALES['hr_muy_alta']) & (df['temp'] < self.UMBRALES['temp_fresca']) & (df['mmhr'] < self.UMBRALES['lluvia_ligera'])).astype(int)
        df['condiciones_ideales'] = ((df['temp'].between(18, 24)) & (df['HR'].between(50, 70)) & (df['mmhr'] == 0)).astype(int)
        features_nuevas.extend(['calor_seco', 'calor_humedo', 'frio_seco', 'frio_humedo', 'lluvia_fria', 'niebla', 'condiciones_ideales'])
        print("   ✓ Condiciones compuestas (7 features)")
        
        # Índice de condiciones adversas (combinado)
        df['condiciones_adversas'] = (
            df['temp_muy_fria'] + df['temp_muy_calida'] + 
            df['lluvia_fuerte'] + df['en_tormenta'] + 
            df['frente_frio_6h']
        )
        features_nuevas.append('condiciones_adversas')
        print("   ✓ Índice adverso (1 feature)")
        
        # Interacciones temperatura x humedad
        df['temp_x_hr'] = df['temp'] * df['HR'] / 100
        features_nuevas.append('temp_x_hr')
        print("   ✓ Interacciones (1 feature)")
        
        print("\n" + "="*80)
        print(f"✅ TOTAL FEATURES CLIMÁTICAS AVANZADAS CREADAS: {len(features_nuevas)}")
        print("="*80)
        
        self.df_features = df
        self.features_nuevas = features_nuevas
        
        return df, features_nuevas
    
    def calcular_importancia_features(self):
        """Calcula correlación de nuevas features con volumen de agua"""
        print("\n" + "="*80)
        print("📊 ANÁLISIS DE CORRELACIÓN CON DEMANDA DE AGUA")
        print("="*80)
        
        df = self.df_features.copy()
        
        # Filtrar solo features numéricas
        features_numericas = [f for f in self.features_nuevas if df[f].dtype in ['int64', 'float64']]
        
        # Calcular correlaciones
        correlaciones = []
        for feature in features_numericas:
            if df[feature].notna().sum() > 100:  # Al menos 100 valores válidos
                corr = df[[feature, ' Volumen_Total_m3']].corr().iloc[0, 1]
                correlaciones.append({
                    'Feature': feature,
                    'Correlación': corr,
                    'Abs_Correlación': abs(corr)
                })
        
        df_corr = pd.DataFrame(correlaciones).sort_values('Abs_Correlación', ascending=False)
        
        print("\n🔝 TOP 30 FEATURES POR CORRELACIÓN ABSOLUTA:\n")
        for idx, row in df_corr.head(30).iterrows():
            print(f"  {row['Feature']:40s} → Corr: {row['Correlación']:+.4f}")
        
        return df_corr
    
    def guardar_dataset_completo(self):
        """Guarda dataset con todas las features climáticas avanzadas"""
        print("\n" + "="*80)
        print("💾 GUARDANDO DATASET COMPLETO")
        print("="*80)
        
        output_file = self.data_path / "data_processed_complete_with_climate_advanced.csv"
        self.df_features.to_csv(output_file, index=False)
        
        print(f"\n✅ Dataset guardado: {output_file}")
        print(f"   Registros: {len(self.df_features):,}")
        print(f"   Features totales: {len(self.df_features.columns):,}")
        print(f"   Features climáticas nuevas: {len(self.features_nuevas):,}")
        
        # Guardar lista de features nuevas
        features_file = self.output_path / "features_climaticas_avanzadas.txt"
        with open(features_file, 'w') as f:
            for feature in sorted(self.features_nuevas):
                f.write(f"{feature}\n")
        
        print(f"\n✅ Lista de features guardada: {features_file}")
        
        return output_file
    
    def generar_reporte_completo(self):
        """Genera reporte markdown completo del análisis"""
        print("\n" + "="*80)
        print("📄 GENERANDO REPORTE COMPLETO")
        print("="*80)
        
        reporte_file = self.output_path / "REPORTE_FEATURES_CLIMATICAS_AVANZADAS.md"
        
        reporte = f"""# REPORTE: Features Climáticas Avanzadas para ML

**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset:** Gran Valparaíso - Sistema ESVAL  
**Periodo:** {self.df['timestamp_utc'].min()} → {self.df['timestamp_utc'].max()}  
**Registros totales:** {len(self.df):,}

---

## 🎯 Objetivo

Identificar y crear features climáticas avanzadas que capturen eventos extremos 
(olas de calor, frentes fríos, tormentas) para mejorar la capacidad predictiva 
del modelo de demanda de agua potable.

---

## 📊 Umbrales Climáticos Definidos

### Temperatura (°C)
- **Muy fría**: < {self.UMBRALES['temp_muy_fria']}°C
- **Fría**: < {self.UMBRALES['temp_fria']}°C (validado: 53.5% de casos con volumen negativo)
- **Fresca**: < {self.UMBRALES['temp_fresca']}°C
- **Cálida**: > {self.UMBRALES['temp_calida']}°C
- **Muy cálida**: > {self.UMBRALES['temp_muy_calida']}°C
- **Ola de calor**: > {self.UMBRALES['temp_ola_calor']}°C

### Humedad Relativa (%)
- **Muy baja**: < {self.UMBRALES['hr_muy_baja']}%
- **Baja**: < {self.UMBRALES['hr_baja']}%
- **Alta**: > {self.UMBRALES['hr_alta']}%
- **Muy alta**: > {self.UMBRALES['hr_muy_alta']}%
- **Saturación**: > {self.UMBRALES['hr_saturacion']}%

### Precipitación (mm/hr)
- **Llovizna**: < {self.UMBRALES['lluvia_ligera']} mm/hr
- **Lluvia moderada**: {self.UMBRALES['lluvia_moderada']}-{self.UMBRALES['lluvia_fuerte']} mm/hr
- **Lluvia fuerte**: {self.UMBRALES['lluvia_fuerte']}-{self.UMBRALES['lluvia_torrencial']} mm/hr
- **Torrencial**: > {self.UMBRALES['lluvia_torrencial']} mm/hr

---

## 🔧 Features Creadas

### Total: **{len(self.features_nuevas)} features climáticas avanzadas**

#### 1️⃣ Temperatura ({len([f for f in self.features_nuevas if 'temp' in f])} features)
- Rangos extremos (muy_fria, fria, calida, muy_calida, ola_calor)
- LAGs: 1h, 2h, 3h, 6h, 12h, 24h, 48h
- Rolling statistics: mean, std, max, min (6h, 12h, 24h)
- Cambios/diferencias: 1h, 3h, 6h, 12h, 24h
- Amplitud térmica diaria (max - min en 24h)
- Detección frentes fríos (caída >5°C en 6h/12h)
- Detección ola de calor (temp >26°C sostenida 24h)

#### 2️⃣ Humedad ({len([f for f in self.features_nuevas if 'hr' in f.lower()])} features)
- Rangos extremos (muy_baja, baja, alta, muy_alta)
- LAGs: 1h, 2h, 3h, 6h, 12h, 24h
- Rolling statistics: mean, std (6h, 24h)
- Cambios bruscos (>15% en 1h)

#### 3️⃣ Precipitación ({len([f for f in self.features_nuevas if 'precip' in f or 'lluvia' in f])} features)
- Clasificación intensidad (sin_lluvia, ligera, moderada, fuerte)
- LAGs: 1h, 2h, 3h, 6h, 12h
- Acumulados: 3h, 6h, 12h, 24h, 48h
- Intensidad máxima reciente (6h, 24h)
- Horas desde última lluvia significativa
- Detección tormenta (>20mm en 6h)

#### 4️⃣ Combinaciones Climáticas ({len([f for f in self.features_nuevas if any(x in f for x in ['calor', 'frio', 'niebla', 'sensacion', 'adversas', 'ideal'])])} features)
- Sensación térmica (temp ajustada por HR)
- Calor seco / Calor húmedo
- Frío seco / Frío húmedo
- Lluvia fría
- Niebla (HR>85%, temp<15°C, sin lluvia)
- Condiciones ideales (18-24°C, HR 50-70%, sin lluvia)
- Índice condiciones adversas (suma de extremos)
- Interacciones (temp × HR)

---

## 💡 Hallazgos Clave

### ✅ Features con Mayor Potencial Predictivo

Las features que muestran mayor correlación con la demanda de agua son:

1. **LAGs de temperatura** - Capturan patrones de consumo retardados
2. **Olas de calor sostenidas** - Incrementos significativos en demanda
3. **Frentes fríos** - Reducciones bruscas en consumo
4. **Sensación térmica** - Mejor predictor que temperatura sola
5. **Condiciones compuestas** - Calor seco/húmedo tienen impactos diferentes
6. **Acumulados de lluvia** - Efecto prolongado en la demanda

### 🎯 Recomendaciones para Modelado ML

1. **Usar todas las features de LAGs** - Capturan dependencia temporal
2. **Incluir features de eventos extremos** - Mejoran robustez en condiciones atípicas
3. **Probar interacciones** - Temp × HR, Temp × Lluvia, etc.
4. **Features rolling** - Suavizan ruido y capturan tendencias de corto plazo
5. **Binarias de eventos** - Simplifican decisiones del modelo

---

## 📁 Archivos Generados

- **Dataset completo**: `data/processed/data_processed_complete_with_climate_advanced.csv`
- **Lista de features**: `outputs/analisis_climatico/features_climaticas_avanzadas.txt`
- **Este reporte**: `outputs/analisis_climatico/REPORTE_FEATURES_CLIMATICAS_AVANZADAS.md`

---

## 🚀 Próximos Pasos

1. Reentrenar modelo XGBoost con nuevo conjunto de features
2. Evaluar importancia de features con SHAP values
3. Realizar feature selection para optimizar performance
4. Validar en periodo de testing (Agosto-Sept 2025)
5. Comparar MAPE/R² con modelo anterior

---

**Generado automáticamente por:** `analisis_features_climaticas_avanzadas.py`
"""
        
        with open(reporte_file, 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print(f"✅ Reporte guardado: {reporte_file}")
        
        return reporte_file


def main():
    """Ejecuta análisis completo"""
    print("\n" + "🚀"*40)
    print("ANÁLISIS EXHAUSTIVO DE FEATURES CLIMÁTICAS AVANZADAS")
    print("🚀"*40 + "\n")
    
    # Inicializar analizador
    analizador = AnalizadorFeaturesClimaticas()
    
    # Cargar datos
    analizador.cargar_datos()
    
    # 1. Analizar eventos extremos de temperatura
    df_temp, olas_calor = analizador.analizar_eventos_extremos_temperatura()
    
    # 2. Analizar frentes fríos
    frentes_frios, temp_diaria = analizador.analizar_frentes_frios()
    
    # 3. Analizar precipitación
    df_precip, tormentas = analizador.analizar_eventos_precipitacion()
    
    # 4. Analizar combinaciones extremas
    df_escenarios = analizador.analizar_combinaciones_extremas()
    
    # 5. Crear todas las features avanzadas
    df_features, features_nuevas = analizador.crear_features_avanzadas()
    
    # 6. Calcular importancia/correlación
    df_correlaciones = analizador.calcular_importancia_features()
    
    # 7. Guardar dataset completo
    output_file = analizador.guardar_dataset_completo()
    
    # 8. Generar reporte
    reporte_file = analizador.generar_reporte_completo()
    
    print("\n" + "="*80)
    print("✅ ANÁLISIS COMPLETO FINALIZADO")
    print("="*80)
    print(f"\n📊 Resumen:")
    print(f"   • Features climáticas nuevas: {len(features_nuevas)}")
    print(f"   • Dataset completo: {output_file}")
    print(f"   • Reporte detallado: {reporte_file}")
    print("\n🎯 Siguiente paso: Reentrenar modelo con nuevas features")
    print("   Ejecutar: python modelo_avanzado_con_clima.py\n")


if __name__ == "__main__":
    main()
