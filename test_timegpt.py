#!/usr/bin/env python3
"""
TimeGPT SDK Test - Real Unemployment Forecasting
"""

from nixtla import NixtlaClient
import pandas as pd
import json
from datetime import datetime, timedelta

# Initialize TimeGPT client with your API key
nixtla_client = NixtlaClient(
    api_key="nixak-NsKC5AcRal1bByr7Bp3JzJEpd8hS0r8X1GYoElLZww5smMtKfCyPISaE8oR8DZ7nqvTG2y93NmeEo1Jl"
)

def test_timegpt_forecast():
    """Test TimeGPT with realistic unemployment data"""
    
    print("Testing TimeGPT SDK...")
    print("API Key: nixak-NsKC5A... (configured)")
    
    # Create sample unemployment data (monthly from 2020-2024)
    dates = pd.date_range(start='2020-01-01', end='2024-08-01', freq='MS')
    
    # Realistic unemployment rates with COVID impact
    unemployment_rates = [
        3.5, 3.5, 4.4, 14.8, 13.3, 11.1, 10.2, 8.4, 6.9, 6.7, 6.3, 6.7,  # 2020
        6.3, 6.2, 6.0, 5.8, 5.4, 5.2, 4.8, 4.6, 4.2, 4.2, 3.9, 3.9,      # 2021
        4.0, 3.8, 3.6, 3.6, 3.6, 3.6, 3.7, 3.5, 3.5, 3.7, 3.7, 3.5,      # 2022
        3.4, 3.6, 3.5, 3.4, 3.7, 3.6, 3.5, 3.8, 3.8, 3.9, 3.7, 3.7,      # 2023
        3.7, 3.9, 3.8, 3.9, 4.0, 4.0, 4.3, 4.3                           # 2024 (partial)
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'ds': dates[:len(unemployment_rates)],
        'y': unemployment_rates
    })
    
    print(f"Data points: {len(df)}")
    print(f"Date range: {df['ds'].min().strftime('%Y-%m')} to {df['ds'].max().strftime('%Y-%m')}")
    print(f"Current unemployment: {df['y'].iloc[-1]}%")
    
    try:
        # Generate 8-quarter (24 month) forecast
        print("\nGenerating TimeGPT forecast...")
        forecast_df = nixtla_client.forecast(
            df=df,
            h=24,           # 24 months (8 quarters)
            freq='MS',      # Monthly frequency
            time_col='ds',
            target_col='y'
        )
        
        print(f"TimeGPT forecast successful!")
        print(f"Forecast points: {len(forecast_df)}")
        
        # Display results
        print(f"\n8Q Unemployment Forecast (Next 24 months):")
        for i in range(0, len(forecast_df), 3):  # Show every 3 months (quarterly)
            row = forecast_df.iloc[i]
            quarter = i // 3 + 1
            print(f"  Q{quarter} ({row['ds'].strftime('%Y-%m')}): {row['TimeGPT']:.1f}%")
            
        # Show trend
        latest_actual = df['y'].iloc[-1]
        forecast_end = forecast_df['TimeGPT'].iloc[-1]
        change = forecast_end - latest_actual
        trend = "increase" if change > 0 else "decrease"
        
        print(f"\nForecast Summary:")
        print(f"  Current: {latest_actual}%")
        print(f"  End forecast: {forecast_end:.1f}%")
        print(f"  Change: {change:+.1f}% ({trend})")
        
        return True
        
    except Exception as e:
        print(f"TimeGPT Error: {e}")
        return False

if __name__ == "__main__":
    success = test_timegpt_forecast()
    if success:
        print("\nTimeGPT SDK is working correctly!")
    else:
        print("\nTimeGPT SDK test failed")