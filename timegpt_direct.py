#!/usr/bin/env python3
"""
Direct TimeGPT Usage Example
This shows how to use TimeGPT directly without CORS restrictions
"""

from nixtla import NixtlaClient
import pandas as pd
import json

# Initialize TimeGPT client
nixtla_client = NixtlaClient(
    api_key="nixak-NsKC5AcRal1bByr7Bp3JzJEpd8hS0r8X1GYoElLZww5smMtKfCyPISaE8oR8DZ7nqvTG2y93NmeEo1Jl"
)

def forecast_with_timegpt(data, horizon=24):
    """
    Direct TimeGPT forecasting
    
    Args:
        data: List of dicts with 'date' and 'value' keys
        horizon: Number of periods to forecast
    
    Returns:
        List of forecast points
    """
    
    # Convert to pandas DataFrame (TimeGPT format)
    df = pd.DataFrame([
        {
            'ds': item['date'],  # Date column
            'y': float(item['value'])  # Value column
        }
        for item in data
    ])
    
    print(f"📊 Forecasting with {len(df)} data points, horizon: {horizon}")
    print(f"📈 Data range: {df['ds'].min()} to {df['ds'].max()}")
    
    try:
        # Generate forecast using TimeGPT
        forecast_df = nixtla_client.forecast(
            df=df,
            h=horizon,      # Forecast horizon
            freq='MS',      # Monthly start frequency
            time_col='ds',  # Time column name
            target_col='y'  # Target column name
        )
        
        print(f"✅ TimeGPT forecast successful: {len(forecast_df)} points")
        
        # Convert back to JSON format
        forecast_data = []
        for _, row in forecast_df.iterrows():
            forecast_data.append({
                'date': row['ds'].strftime('%Y-%m-%d'),
                'value': round(float(row['TimeGPT']), 2)
            })
            
        return forecast_data
        
    except Exception as e:
        print(f"❌ TimeGPT Error: {e}")
        return []

# Example usage
if __name__ == "__main__":
    # Sample unemployment data
    sample_data = [
        {"date": "2023-01-01", "value": 3.5},
        {"date": "2023-02-01", "value": 3.6},
        {"date": "2023-03-01", "value": 3.5},
        {"date": "2023-04-01", "value": 3.4},
        {"date": "2023-05-01", "value": 3.7},
        {"date": "2023-06-01", "value": 3.6},
    ]
    
    # Generate forecast
    forecast = forecast_with_timegpt(sample_data, horizon=12)
    
    print("\n📈 TimeGPT Forecast Results:")
    for point in forecast[:6]:  # Show first 6 points
        print(f"  {point['date']}: {point['value']}%")