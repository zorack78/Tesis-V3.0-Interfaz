"""
Generador de Features Completo para Prediccion de Demanda
Basado en especificacion del usuario - Listado_de_features_para_sondeo_de_correlaciones.csv

Target: Q_net_m3h (flujo neto del sistema = ΔVol/Δt)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

def cargar_datos_raw():
    """Carga todos los datasets raw necesarios"""
    print("Cargando datasets raw...")
    
    # Cargar Q_net (target - flujo neto del sistema)
    qnet = pd.read_csv('data/raw/BD_Q_net_x_Hr_m3h_LIMPIO.csv')
    qnet.columns = qnet.columns.str.strip()
    qnet['timestamp'] = pd.to_datetime(qnet['timestamp'])
    
    # Cargar clima
    clima = pd.read_csv('data/raw/BD_Clima2024a202509_Local.csv')
    clima.columns = clima.columns.str.strip()
    clima['timestamp'] = pd.to_datetime(clima['timestamp'])
    
    # Cargar Qin
    qin = pd.read_csv('data/raw/BD_Qin_m3_Local.csv')
    qin.columns = qin.columns.str.strip()
    qin['timestamp'] = pd.to_datetime(qin['timestamp'])
    
    # Cargar volumen total
    voltotal = pd.read_csv('data/raw/BD_VolTotal_X_Hr_m3_Local.csv')
    voltotal.columns = voltotal.columns.str.strip()
    voltotal['timestamp'] = pd.to_datetime(voltotal['timestamp'])
    
    # Cargar calendario
    calendar = pd.read_csv('data/raw/calendar_social_ES_COMPLETO_20240101_20250930.csv')
    calendar.columns = calendar.columns.str.strip()
    calendar['timestamp'] = pd.to_datetime(calendar['timestamp'])
    
    print(f"  Q_net:   {qnet.shape}")
    print(f"  Clima:   {clima.shape}")
    print(f"  Qin:     {qin.shape}")
    print(f"  Vol:     {voltotal.shape}")
    print(f"  Cal:     {calendar.shape}")
    
    return qnet, clima, qin, voltotal, calendar


def generar_features_clima_base(clima_df):
    """Genera features climáticas básicas con renombrado"""
    df = clima_df.copy()
    
    # Renombrar a nomenclatura especificada
    df = df.rename(columns={
        'temp': 'clima_temp_c',
        'HR': 'clima_HR_pct',
        'mmhr': 'clima_lluvia_mmhr'
    })
    
    return df


def generar_lags_clima(df, var, lags):
    """Genera LAGs de una variable climática"""
    for lag in lags:
        df[f'{var}__lag_{lag}h'] = df[var].shift(lag)
    return df


def generar_rolling_stats(df, var, windows):
    """Genera estadísticas móviles (media, std, min, max) para ventanas"""
    for w in windows:
        df[f'{var}__mean_win_{w}h'] = df[var].rolling(window=w, min_periods=1).mean()
        df[f'{var}__std_win_{w}h'] = df[var].rolling(window=w, min_periods=1).std()
        df[f'{var}__min_win_{w}h'] = df[var].rolling(window=w, min_periods=1).min()
        df[f'{var}__max_win_{w}h'] = df[var].rolling(window=w, min_periods=1).max()
    return df


def generar_pendientes_lineales(df, var, windows):
    """Genera pendientes lineales para ventanas"""
    def compute_slope(series):
        try:
            if len(series) < 2:
                return np.nan
            # Convertir a numpy array si es pandas Series
            if hasattr(series, 'values'):
                y = series.values
            else:
                y = np.array(series)
            
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return np.nan
            
            x = np.arange(len(y))
            slope, _ = np.polyfit(x[mask], y[mask], 1)
            return slope
        except:
            return np.nan
    
    for w in windows:
        df[f'{var}__slope_lin_win_{w}h'] = df[var].rolling(window=w, min_periods=2).apply(compute_slope, raw=True)
    
    return df


def generar_features_lluvia_avanzadas(df):
    """Genera features avanzadas de lluvia"""
    var = 'clima_lluvia_mmhr'
    windows = [6, 12, 24, 48, 72, 168]
    
    # Acumulaciones
    for w in windows:
        df[f'clima_lluvia_mm__accum_win_{w}h'] = df[var].rolling(window=w, min_periods=1).sum()
        df[f'clima_lluvia_intensidad_max_win_{w}h'] = df[var].rolling(window=w, min_periods=1).max()
    
    # Eventos
    df['clima_lluvia_evento_activo'] = (df[var] > 0).astype(int)
    
    # Horas desde última lluvia
    lluvia_mask = df[var] > 0
    horas_desde = []
    last_lluvia = 0
    for i, tiene_lluvia in enumerate(lluvia_mask):
        if tiene_lluvia:
            last_lluvia = i
            horas_desde.append(0)
        else:
            horas_desde.append(i - last_lluvia if last_lluvia > 0 else 999)
    df['clima_horas_desde_ultima_lluvia'] = horas_desde
    
    return df


def generar_features_termodinamicas(df):
    """Genera punto de rocío, VPD, temperatura aparente"""
    T = df['clima_temp_c']
    RH = df['clima_HR_pct']
    
    # Punto de rocío (Magnus formula)
    a = 17.27
    b = 237.7
    alpha = ((a * T) / (b + T)) + np.log(RH / 100)
    df['clima_dewpoint_c'] = (b * alpha) / (a - alpha)
    
    # Presión vapor saturación (kPa)
    df['clima_esat_kPa'] = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
    
    # Déficit de presión de vapor (VPD)
    df['clima_vpd_kPa'] = df['clima_esat_kPa'] * (1 - RH / 100)
    
    # Temperatura aparente (Humidex simplificado)
    e = (RH / 100) * 6.112 * np.exp((17.67 * T) / (T + 243.5))
    df['clima_apparent_temp_c'] = T + 0.5555 * (e - 10)
    
    return df


def generar_deltas_temperatura(df):
    """Genera cambios de temperatura en diferentes horizontes"""
    var = 'clima_temp_c'
    horizons = [1, 3, 6, 12, 24]
    
    for h in horizons:
        df[f'clima_temp_delta_{h}h'] = df[var] - df[var].shift(h)
    
    # Delta vs misma hora ayer
    df['clima_delta_same_hour_vs_ayer'] = df[var] - df[var].shift(24)
    
    return df


def generar_degree_hours(df):
    """Genera CDH (Cooling) y HDH (Heating) Degree Hours"""
    T = df['clima_temp_c']
    
    # CDH (umbral 22°C)
    df['clima_CDH_instantaneo'] = np.maximum(T - 22, 0)
    df['clima_CDH_win_24h'] = df['clima_CDH_instantaneo'].rolling(window=24, min_periods=1).sum()
    df['clima_CDH_win_48h'] = df['clima_CDH_instantaneo'].rolling(window=48, min_periods=1).sum()
    
    # HDH (umbral 18°C)
    df['clima_HDH_instantaneo'] = np.maximum(18 - T, 0)
    df['clima_HDH_win_24h'] = df['clima_HDH_instantaneo'].rolling(window=24, min_periods=1).sum()
    df['clima_HDH_win_48h'] = df['clima_HDH_instantaneo'].rolling(window=48, min_periods=1).sum()
    
    return df


def generar_VPD_agregados(df):
    """Genera estadísticas de VPD en ventanas"""
    df['clima_VPD_mean_win_24h'] = df['clima_vpd_kPa'].rolling(window=24, min_periods=1).mean()
    df['clima_VPD_mean_win_48h'] = df['clima_vpd_kPa'].rolling(window=48, min_periods=1).mean()
    
    # % horas con VPD > 1.5 kPa (estrés hídrico)
    vpd_alto = (df['clima_vpd_kPa'] > 1.5).astype(int)
    df['clima_pct_horas_VPD_gt_1p5_win_24h'] = vpd_alto.rolling(window=24, min_periods=1).mean() * 100
    df['clima_pct_horas_VPD_gt_1p5_win_48h'] = vpd_alto.rolling(window=48, min_periods=1).mean() * 100
    
    return df


def generar_API(df):
    """Genera Antecedent Precipitation Index (API)"""
    lluvia = df['clima_lluvia_mmhr']
    lambda_decay = 0.9
    
    # API 24h
    api_24 = []
    for i in range(len(df)):
        suma = 0
        for j in range(min(24, i + 1)):
            suma += lluvia.iloc[i - j] * (lambda_decay ** j)
        api_24.append(suma)
    df['clima_API_lambda_0p9_win_24h'] = api_24
    
    # API 72h
    api_72 = []
    for i in range(len(df)):
        suma = 0
        for j in range(min(72, i + 1)):
            suma += lluvia.iloc[i - j] * (lambda_decay ** j)
        api_72.append(suma)
    df['clima_API_lambda_0p9_win_72h'] = api_72
    
    return df


def generar_amplitud_diurna(df):
    """Genera amplitud diurna de temperatura"""
    # Agrupar por día y calcular rango
    df['date'] = df['timestamp'].dt.date
    daily_temp = df.groupby('date')['clima_temp_c'].agg(['min', 'max'])
    daily_temp['amplitud'] = daily_temp['max'] - daily_temp['min']
    
    # Mapear de vuelta
    df['clima_amplitud_diurna_temp'] = df['date'].map(daily_temp['amplitud'])
    df.drop('date', axis=1, inplace=True)
    
    return df


def generar_features_calendario_avanzadas(df):
    """Genera features de calendario avanzadas"""
    # Sin/Cos para ciclos
    df['cal_hour_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
    df['cal_hour_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
    df['cal_dow_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['cal_dow_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    
    # Día del año
    df['dia_anio'] = df['timestamp'].dt.dayofyear
    df['cal_doy_sin'] = np.sin(2 * np.pi * df['dia_anio'] / 365)
    df['cal_doy_cos'] = np.cos(2 * np.pi * df['dia_anio'] / 365)
    
    # Fotoperiodo (latitud Valparaíso: -33.05°)
    lat = -33.05
    declinacion = 23.45 * np.sin(np.radians(360 * (df['dia_anio'] - 81) / 365))
    cos_hour_angle = -np.tan(np.radians(lat)) * np.tan(np.radians(declinacion))
    cos_hour_angle = np.clip(cos_hour_angle, -1, 1)
    hour_angle = np.arccos(cos_hour_angle)
    df['cal_fotoperiodo_horas'] = (2 / 15) * np.degrees(hour_angle)
    
    # Distancia a feriados
    feriados_idx = df[df['feriado'] == 1].index
    
    horas_al_prox_feriado = []
    horas_desde_ultimo_feriado = []
    
    for i in range(len(df)):
        # Próximo feriado
        futuros = feriados_idx[feriados_idx > i]
        if len(futuros) > 0:
            horas_al_prox_feriado.append(futuros[0] - i)
        else:
            horas_al_prox_feriado.append(999)
        
        # Último feriado
        pasados = feriados_idx[feriados_idx <= i]
        if len(pasados) > 0:
            horas_desde_ultimo_feriado.append(i - pasados[-1])
        else:
            horas_desde_ultimo_feriado.append(999)
    
    df['cal_prox_feriado_horas'] = horas_al_prox_feriado
    df['cal_desde_ultimo_feriado_horas'] = horas_desde_ultimo_feriado
    
    # Renombrar a nomenclatura especificada
    renombre_cal = {
        'es_fin_de_semana': 'cal_es_fin_de_semana',
        'feriado': 'cal_feriado',
        'feriado_movible': 'cal_feriado_movible',
        'feriado_irrenunciable': 'cal_feriado_irrenunciable',
        'vacaciones_escolares': 'cal_vacaciones_escolares',
        'inicio_clases': 'cal_inicio_clases',
        'temporada_turistica_alta': 'cal_temporada_turistica_alta',
        'festival_vina': 'cal_festival_vina',
        'elecciones': 'cal_elecciones',
        'fiestas_patrias': 'cal_fiestas_patrias',
        'anio_nuevo': 'cal_anio_nuevo'
    }
    
    for old, new in renombre_cal.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    
    return df


def generar_features_hidraulicas(df, qin_df, voltotal_df):
    """Genera features del sistema hidráulico"""
    # Merge Qin
    df = pd.merge(df, qin_df[['timestamp', 'Qin']], on='timestamp', how='left')
    df = df.rename(columns={'Qin': 'sist_Qin_m3h'})
    
    # Merge Volumen
    df = pd.merge(df, voltotal_df[['timestamp', 'Volumen_Total_m3']], on='timestamp', how='left')
    df = df.rename(columns={'Volumen_Total_m3': 'sist_Vtotal_m3'})
    
    return df


def generar_features_target_derivadas(df):
    """
    Genera features derivadas del target (Q_net)
    
    ⚠️ TODAS LAS FEATURES DE Q_NET DESHABILITADAS POR DATA LEAKAGE
    
    Cualquier feature calculada desde Q_net (lags, diffs, EMAs, rolling)
    introduce data leakage porque usa valores del test set para predecir
    otros valores del test set.
    
    Ejemplo: Q_net__lag_168h para predecir Sep 25 usa Sep 18,
    pero Sep 18 está en el test set (después de marzo 23).
    """
    # DESHABILITADO - LAG semanal causa data leakage
    # df['Q_net_m3h__lag_168h'] = df['Q_net_m3h'].shift(168)
    
    # DESHABILITADO - Diferencia vs semana pasada causa data leakage
    # df['Q_net_m3h__diff_168h'] = df['Q_net_m3h'] - df['Q_net_m3h'].shift(168)
    
    # DESHABILITADO - EMAs causan data leakage
    # df['Q_net_m3h__ema_win_6h'] = df['Q_net_m3h'].ewm(span=6, adjust=False).mean()
    # df['Q_net_m3h__ema_win_12h'] = df['Q_net_m3h'].ewm(span=12, adjust=False).mean()
    # df['Q_net_m3h__ema_win_24h'] = df['Q_net_m3h'].ewm(span=24, adjust=False).mean()
    
    return df


def generar_features_ema_externas(df):
    """
    Genera EMAs de variables EXTERNAS (sin data leakage)
    
    ✅ SEGURO: Estas variables son inputs conocidos en tiempo real,
    no dependen del target (Q_net), por lo tanto NO causan leakage.
    
    En producción, estas variables están disponibles antes de predecir.
    """
    print("\n📊 Generando EMAs de variables externas (sin leakage)...")
    
    # EMAs de Temperatura (disponible en tiempo real)
    if 'clima_temp_c' in df.columns:
        df['clima_temp_ema_24h'] = df['clima_temp_c'].ewm(span=24, adjust=False).mean()
        df['clima_temp_ema_168h'] = df['clima_temp_c'].ewm(span=168, adjust=False).mean()
        print("   ✅ Temperatura EMA (24h, 168h)")
    
    # EMAs de Humedad Relativa
    if 'clima_HR_pct' in df.columns:
        df['clima_HR_ema_24h'] = df['clima_HR_pct'].ewm(span=24, adjust=False).mean()
        df['clima_HR_ema_168h'] = df['clima_HR_pct'].ewm(span=168, adjust=False).mean()
        print("   ✅ Humedad Relativa EMA (24h, 168h)")
    
    # EMAs de Qin (input del sistema conocido)
    if 'sist_Qin_m3h' in df.columns:
        df['sist_Qin_ema_24h'] = df['sist_Qin_m3h'].ewm(span=24, adjust=False).mean()
        df['sist_Qin_ema_168h'] = df['sist_Qin_m3h'].ewm(span=168, adjust=False).mean()
        print("   ✅ Qin EMA (24h, 168h)")
    
    # EMAs de Volumen Total (estado del sistema conocido)
    if 'sist_Vtotal_m3' in df.columns:
        df['sist_Vtotal_ema_24h'] = df['sist_Vtotal_m3'].ewm(span=24, adjust=False).mean()
        df['sist_Vtotal_ema_168h'] = df['sist_Vtotal_m3'].ewm(span=168, adjust=False).mean()
        print("   ✅ Volumen Total EMA (24h, 168h)")
    
    # EMAs de variables climáticas derivadas (si existen)
    if 'clima_VPD_kpa' in df.columns:
        df['clima_VPD_ema_24h'] = df['clima_VPD_kpa'].ewm(span=24, adjust=False).mean()
        print("   ✅ VPD EMA (24h)")
    
    if 'clima_CDH_acum' in df.columns:
        df['clima_CDH_ema_24h'] = df['clima_CDH_acum'].ewm(span=24, adjust=False).mean()
        print("   ✅ CDH EMA (24h)")
    
    # EMAs cortas adicionales (útiles para captar cambios rápidos)
    if 'clima_temp_c' in df.columns:
        df['clima_temp_ema_6h'] = df['clima_temp_c'].ewm(span=6, adjust=False).mean()
        print("   ✅ Temperatura EMA corta (6h)")
    
    if 'sist_Qin_m3h' in df.columns:
        df['sist_Qin_ema_6h'] = df['sist_Qin_m3h'].ewm(span=6, adjust=False).mean()
        print("   ✅ Qin EMA corta (6h)")
    
    return df


def generar_features_interaccion(df):
    """Genera features de interacción clima × calendario"""
    # CDH × fin de semana
    if 'clima_CDH_win_24h' in df.columns and 'cal_es_fin_de_semana' in df.columns:
        df['mix_CDH24_x_finsemana'] = df['clima_CDH_win_24h'] * df['cal_es_fin_de_semana']
    
    # API × día laboral
    if 'clima_API_lambda_0p9_win_72h' in df.columns and 'cal_es_fin_de_semana' in df.columns:
        df['mix_API72_x_laboral'] = df['clima_API_lambda_0p9_win_72h'] * (1 - df['cal_es_fin_de_semana'])
    
    # VPD × temporada alta
    if 'clima_VPD_mean_win_24h' in df.columns and 'cal_temporada_turistica_alta' in df.columns:
        df['mix_VPD24_x_temporada_alta'] = df['clima_VPD_mean_win_24h'] * df['cal_temporada_turistica_alta']
    
    # Temperatura alta + HR baja × turismo
    if all(col in df.columns for col in ['clima_temp_c', 'clima_HR_pct', 'cal_temporada_turistica_alta']):
        calor_seco = ((df['clima_temp_c'] > 22) & (df['clima_HR_pct'] < 40)).astype(int)
        df['mix_Tgt22_HRlt40_x_turismo'] = calor_seco * df['cal_temporada_turistica_alta']
    
    return df


def main():
    """Pipeline principal de generación de features"""
    print("=" * 100)
    print("GENERADOR DE FEATURES COMPLETO")
    print("=" * 100)
    
    # 1. Cargar datos
    qnet, clima, qin, voltotal, calendar = cargar_datos_raw()
    
    # 2. Merge inicial (Q_net + clima)
    print("\nGenerando features climaticas...")
    clima = generar_features_clima_base(clima)
    
    df = pd.merge(qnet, clima, on='timestamp', how='inner')
    print(f"  Merge Q_net + clima: {df.shape}")
    
    # 3. LAGs climáticos
    print("\nGenerando LAGs climaticos...")
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    for var in ['clima_temp_c', 'clima_HR_pct', 'clima_lluvia_mmhr']:
        df = generar_lags_clima(df, var, lags)
    
    # 4. Rolling stats
    print("\nGenerando estadisticas moviles...")
    windows = [6, 12, 24, 48, 72, 168]
    for var in ['clima_temp_c', 'clima_HR_pct']:
        df = generar_rolling_stats(df, var, windows)
        df = generar_pendientes_lineales(df, var, windows)
    
    # 5. Features lluvia avanzadas
    print("\nGenerando features de lluvia...")
    df = generar_features_lluvia_avanzadas(df)
    
    # 6. Features termodinámicas
    print("\nGenerando features termodinamicas...")
    df = generar_features_termodinamicas(df)
    df = generar_deltas_temperatura(df)
    df = generar_degree_hours(df)
    df = generar_VPD_agregados(df)
    df = generar_API(df)
    df = generar_amplitud_diurna(df)
    
    # 7. Merge calendario
    print("\nMerge con calendario...")
    df = pd.merge(df, calendar, on='timestamp', how='left')
    df = generar_features_calendario_avanzadas(df)
    
    # 8. Features hidráulicas
    print("\nGenerando features hidraulicas...")
    df = generar_features_hidraulicas(df, qin, voltotal)
    
    # 9. EMAs de variables externas - DESHABILITADO (probado, no mejora performance)
    # df = generar_features_ema_externas(df)
    
    # 10. Features derivadas del target
    print("\nGenerando features derivadas del target...")
    df = generar_features_target_derivadas(df)
    
    # 11. Features de interacción
    print("\nGenerando features de interaccion...")
    df = generar_features_interaccion(df)
    
    # 11. Limpiar y guardar
    print("\n" + "=" * 100)
    print("DATASET FINAL")
    print("=" * 100)
    print(f"\nShape: {df.shape}")
    print(f"Registros: {df.shape[0]:,}")
    print(f"Features: {df.shape[1]:,}")
    
    # Identificar columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Features numericas: {len(numeric_cols)}")
    
    # Guardar
    output_path = 'data/processed/dataset_features_completo.csv'
    df.to_csv(output_path, index=False)
    print(f"\nGuardado: {output_path}")
    
    # Reporte de categorías
    print("\n" + "=" * 100)
    print("CATEGORIAS DE FEATURES")
    print("=" * 100)
    
    categorias = {
        'CLIMA_BASE': [c for c in df.columns if c.startswith('clima_') and '__' not in c and 'lag' not in c],
        'CLIMA_LAGS': [c for c in df.columns if 'lag_' in c],
        'CLIMA_ROLLING': [c for c in df.columns if 'win_' in c or 'accum_' in c],
        'CALENDARIO': [c for c in df.columns if c.startswith('cal_')],
        'HIDRAULICA': [c for c in df.columns if c.startswith('sist_')],
        'TARGET_DERIVADO': [c for c in df.columns if c.startswith('Q_net_')],
        'INTERACCION': [c for c in df.columns if c.startswith('mix_')]
    }
    
    for cat, cols in categorias.items():
        print(f"\n{cat}: {len(cols)} features")
        if len(cols) <= 10:
            for col in cols:
                print(f"  - {col}")
        else:
            print(f"  (primeras 5)")
            for col in cols[:5]:
                print(f"  - {col}")
    
    return df


if __name__ == '__main__':
    df_final = main()
    print("\n" + "=" * 100)
    print("COMPLETADO!")
    print("=" * 100)
