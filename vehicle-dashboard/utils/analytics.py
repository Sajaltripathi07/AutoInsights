"""
Analytics utilities for vehicle registration data analysis.
Provides functions for calculating growth metrics and trend analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime, timedelta


class VehicleAnalytics:
    """Analytics engine for vehicle registration data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
    
    def calculate_yoy_growth(self, category: str = None, manufacturer: str = None) -> float:
        """
        Calculate Year-over-Year growth rate.
        
        Args:
            category: Vehicle category filter (optional)
            manufacturer: Manufacturer filter (optional)
            
        Returns:
            YoY growth percentage
        """
        filtered_df = self._apply_filters(category, manufacturer)
        
        if filtered_df.empty:
            return 0.0
        
        latest_year = filtered_df['Year'].max()
        prev_year = latest_year - 1
        
        current_year_total = filtered_df[filtered_df['Year'] == latest_year]['Registrations'].sum()
        prev_year_total = filtered_df[filtered_df['Year'] == prev_year]['Registrations'].sum()
        
        if prev_year_total > 0:
            return ((current_year_total - prev_year_total) / prev_year_total) * 100
        return 0.0
    
    def calculate_qoq_growth(self, category: str = None, manufacturer: str = None) -> float:
        """
        Calculate Quarter-over-Quarter growth rate.
        
        Args:
            category: Vehicle category filter (optional)
            manufacturer: Manufacturer filter (optional)
            
        Returns:
            QoQ growth percentage
        """
        filtered_df = self._apply_filters(category, manufacturer)
        
        if filtered_df.empty:
            return 0.0
        
        latest_date = filtered_df['Date'].max()
        prev_quarter_date = latest_date - pd.DateOffset(months=3)
        
        latest_quarter_total = filtered_df[filtered_df['Date'] == latest_date]['Registrations'].sum()
        prev_quarter_total = filtered_df[filtered_df['Date'] == prev_quarter_date]['Registrations'].sum()
        
        if prev_quarter_total > 0:
            return ((latest_quarter_total - prev_quarter_total) / prev_quarter_total) * 100
        return 0.0
    
    def get_category_performance(self) -> pd.DataFrame:
        """Get performance metrics by category."""
        performance = []
        
        for category in self.df['Category'].unique():
            yoy = self.calculate_yoy_growth(category=category)
            qoq = self.calculate_qoq_growth(category=category)
            total_reg = self.df[self.df['Category'] == category]['Registrations'].sum()
            
            performance.append({
                'Category': category,
                'Total_Registrations': total_reg,
                'YoY_Growth': yoy,
                'QoQ_Growth': qoq
            })
        
        return pd.DataFrame(performance).sort_values('Total_Registrations', ascending=False)
    
    def get_manufacturer_performance(self) -> pd.DataFrame:
        """Get performance metrics by manufacturer."""
        performance = []
        
        for manufacturer in self.df['Manufacturer'].unique():
            yoy = self.calculate_yoy_growth(manufacturer=manufacturer)
            qoq = self.calculate_qoq_growth(manufacturer=manufacturer)
            total_reg = self.df[self.df['Manufacturer'] == manufacturer]['Registrations'].sum()
            
            performance.append({
                'Manufacturer': manufacturer,
                'Total_Registrations': total_reg,
                'YoY_Growth': yoy,
                'QoQ_Growth': qoq
            })
        
        return pd.DataFrame(performance).sort_values('Total_Registrations', ascending=False)
    
    def get_trend_data(self, category: str = None, manufacturer: str = None) -> pd.DataFrame:
        """Get time series trend data."""
        filtered_df = self._apply_filters(category, manufacturer)
        
        if filtered_df.empty:
            return pd.DataFrame()
        
        trend_data = filtered_df.groupby(['Date', 'Category'])['Registrations'].sum().reset_index()
        return trend_data.sort_values('Date')
    
    def get_seasonal_patterns(self) -> Dict[str, pd.DataFrame]:
        """Analyze seasonal patterns in registration data."""
        patterns = {}
        
        for category in self.df['Category'].unique():
            category_data = self.df[self.df['Category'] == category].copy()
            category_data['Month'] = category_data['Date'].dt.month
            
            monthly_avg = category_data.groupby('Month')['Registrations'].mean().reset_index()
            monthly_avg['Category'] = category
            patterns[category] = monthly_avg
        
        return patterns
    
    def _apply_filters(self, category: str = None, manufacturer: str = None) -> pd.DataFrame:
        """Apply category and manufacturer filters to the dataframe."""
        filtered_df = self.df.copy()
        
        if category:
            filtered_df = filtered_df[filtered_df['Category'] == category]
        
        if manufacturer:
            filtered_df = filtered_df[filtered_df['Manufacturer'] == manufacturer]
        
        return filtered_df
