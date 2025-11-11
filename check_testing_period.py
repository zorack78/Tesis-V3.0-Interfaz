import pandas as pd

df = pd.read_csv('data/processed/data_processed_demanda_valid.csv')
df.columns = df.columns.str.strip()
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])

n = len(df)
val_end = int(n * 0.85)
df_test = df.iloc[val_end:]

print(f'Período testing (15% final):')
print(f'  Inicio: {df_test.iloc[0]["timestamp_utc"]}')
print(f'  Fin: {df_test.iloc[-1]["timestamp_utc"]}')
print(f'  Registros: {len(df_test)}')

print(f'\nDemanda en testing:')
print(f'  Min: {df_test["Demanda_m3_hr"].min():,.0f}')
print(f'  Max: {df_test["Demanda_m3_hr"].max():,.0f}')
print(f'  Media: {df_test["Demanda_m3_hr"].mean():,.0f}')

negativos = (df_test['Demanda_m3_hr'] < 0).sum()
print(f'  Valores negativos: {negativos} ({negativos/len(df_test)*100:.2f}%)')

# Verificar si agosto-septiembre 2025
print(f'\nDatos de Agosto-Septiembre 2025:')
mask = (df['timestamp_utc'] >= '2025-08-01') & (df['timestamp_utc'] <= '2025-09-30')
df_ago_sep = df[mask]
print(f'  Registros: {len(df_ago_sep)}')
if len(df_ago_sep) > 0:
    print(f'  Inicio: {df_ago_sep.iloc[0]["timestamp_utc"]}')
    print(f'  Fin: {df_ago_sep.iloc[-1]["timestamp_utc"]}')
