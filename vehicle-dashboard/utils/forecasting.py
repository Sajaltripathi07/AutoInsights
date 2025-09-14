"""
Time series forecasting utilities using ARIMA model.
Provides forecasting capabilities for vehicle registration trends.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')


class VehicleForecasting:
    """ARIMA-based forecasting for vehicle registration data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.forecast_results = {}
    
    def prepare_time_series(self, category: str = None, manufacturer: str = None) -> pd.Series:
        """
        Prepare time series data for forecasting.
        
        Args:
            category: Vehicle category filter (optional)
            manufacturer: Manufacturer filter (optional)
            
        Returns:
            Time series data
        """
        filtered_df = self.df.copy()
        
        if category:
            filtered_df = filtered_df[filtered_df['Category'] == category]
        
        if manufacturer:
            filtered_df = filtered_df[filtered_df['Manufacturer'] == manufacturer]
        
        # Aggregate by date
        ts_data = filtered_df.groupby('Date')['Registrations'].sum().sort_index()
        
        # Fill missing dates with forward fill
        date_range = pd.date_range(start=ts_data.index.min(), end=ts_data.index.max(), freq='M')
        ts_data = ts_data.reindex(date_range, method='ffill')
        
        return ts_data
    
    def find_best_arima_params(self, ts_data: pd.Series, max_p: int = 3, 
                              max_d: int = 2, max_q: int = 3) -> Tuple[int, int, int]:
        """
        Find best ARIMA parameters using AIC.
        
        Args:
            ts_data: Time series data
            max_p: Maximum autoregressive terms
            max_d: Maximum differencing terms
            max_q: Maximum moving average terms
            
        Returns:
            Best (p, d, q) parameters
        """
        best_aic = np.inf
        best_params = (0, 0, 0)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(ts_data, order=(p, d, q))
                        fitted_model = model.fit()
                        aic = fitted_model.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        return best_params
    
    def forecast_registrations(self, category: str = None, manufacturer: str = None, 
                             periods: int = 12, auto_params: bool = True) -> Dict:
        """
        Forecast vehicle registrations using ARIMA.
        
        Args:
            category: Vehicle category filter (optional)
            manufacturer: Manufacturer filter (optional)
            periods: Number of periods to forecast
            auto_params: Whether to automatically find best parameters
            
        Returns:
            Dictionary containing forecast results
        """
        ts_data = self.prepare_time_series(category, manufacturer)
        
        if len(ts_data) < 12:  # Need at least 12 months of data
            return {
                'forecast': pd.Series(),
                'confidence_intervals': pd.DataFrame(),
                'model_summary': None,
                'error': 'Insufficient data for forecasting'
            }
        
        try:
            if auto_params:
                p, d, q = self.find_best_arima_params(ts_data)
            else:
                p, d, q = (1, 1, 1)  # Default parameters
            
            # Fit ARIMA model
            model = ARIMA(ts_data, order=(p, d, q))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=periods)
            conf_int = fitted_model.get_forecast(steps=periods).conf_int()
            
            # Create future dates
            last_date = ts_data.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=periods,
                freq='M'
            )
            
            forecast_series = pd.Series(forecast.values, index=future_dates)
            conf_int.index = future_dates
            
            return {
                'forecast': forecast_series,
                'confidence_intervals': conf_int,
                'model_summary': fitted_model.summary(),
                'params': (p, d, q),
                'aic': fitted_model.aic,
                'error': None
            }
            
        except Exception as e:
            return {
                'forecast': pd.Series(),
                'confidence_intervals': pd.DataFrame(),
                'model_summary': None,
                'error': str(e)
            }
    
    def get_forecast_accuracy(self, category: str = None, manufacturer: str = None) -> Dict:
        """
        Calculate forecast accuracy using train-test split.
        
        Args:
            category: Vehicle category filter (optional)
            manufacturer: Manufacturer filter (optional)
            
        Returns:
            Dictionary containing accuracy metrics
        """
        ts_data = self.prepare_time_series(category, manufacturer)
        
        if len(ts_data) < 24:  # Need at least 24 months for train-test split
            return {'error': 'Insufficient data for accuracy calculation'}
        
        # Split data (use last 6 months for testing)
        train_data = ts_data[:-6]
        test_data = ts_data[-6:]
        
        try:
            # Find best parameters on training data
            p, d, q = self.find_best_arima_params(train_data)
            
            # Fit model on training data
            model = ARIMA(train_data, order=(p, d, q))
            fitted_model = model.fit()
            
            # Forecast test period
            forecast = fitted_model.forecast(steps=len(test_data))
            
            # Calculate accuracy metrics
            mae = np.mean(np.abs(forecast - test_data.values))
            mse = np.mean((forecast - test_data.values) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((test_data.values - forecast) / test_data.values)) * 100
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'params': (p, d, q)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_seasonal_decomposition(self, category: str = None, manufacturer: str = None) -> Dict:
        """
        Perform seasonal decomposition of time series.
        
        Args:
            category: Vehicle category filter (optional)
            manufacturer: Manufacturer filter (optional)
            
        Returns:
            Dictionary containing decomposition components
        """
        ts_data = self.prepare_time_series(category, manufacturer)
        
        if len(ts_data) < 24:  # Need at least 24 months for seasonal decomposition
            return {'error': 'Insufficient data for seasonal decomposition'}
        
        try:
            decomposition = seasonal_decompose(ts_data, model='additive', period=12)
            
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed
            }
            
        except Exception as e:
            return {'error': str(e)}
