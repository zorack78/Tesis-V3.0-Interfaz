import joblib
from pathlib import Path

xgb_path = Path('models/forecasting/modelo_forecasting_xgboost.pkl')
print(f'XGBoost existe: {xgb_path.exists()}')

if xgb_path.exists():
    data = joblib.load(xgb_path)
    print(f'Metadata: R2={data["metadata"]["metricas_test"]["r2"]:.4f}')
    print(f'MAE={data["metadata"]["metricas_test"]["mae"]:.0f}')
