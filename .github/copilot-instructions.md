# Copilot Instructions - Modelo Predictivo Demanda Agua Potable

## Project Overview
This is a machine learning project for predicting water demand in Gran Valparaíso, Chile. The project uses time series forecasting and machine learning models.

## Development Guidelines
- Use Spanish for comments and documentation when appropriate
- Follow PEP 8 style guide for Python code
- Keep data processing modular and reusable
- Document all model parameters and assumptions
- Use timezone-aware datetime operations (UTC and Chile local time)

## Data Context
- Water volume data is in cubic meters (m3)
- Timestamps are in UTC format
- Calendar data includes Chilean holidays and social events
- Time series is hourly frequency

## Model Development
- Start with exploratory data analysis
- Consider seasonality, trends, and external factors
- Test multiple modeling approaches (ARIMA, Prophet, ML models)
- Validate models with appropriate metrics (RMSE, MAE, MAPE)
- Account for Chilean holidays and special events

## Code Organization
- `data/`: Raw and processed datasets
- `notebooks/`: Jupyter notebooks for analysis and experimentation
- `src/`: Production-ready Python modules
- `models/`: Saved model artifacts
- `config/`: Configuration files and parameters
