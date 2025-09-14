"""
Data generation utilities for vehicle registration dashboard.
Generates synthetic data that mimics real-world vehicle registration patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class VehicleDataGenerator:
    """Generates synthetic vehicle registration data for dashboard demonstration."""
    
    def __init__(self):
        self.categories = ['2W', '3W', '4W']
        self.manufacturers = {
            '2W': ['Hero', 'Honda', 'Bajaj', 'TVS', 'Royal Enfield'],
            '3W': ['Bajaj', 'Piaggio', 'Mahindra', 'TVS'],
            '4W': ['Maruti', 'Hyundai', 'Tata', 'Mahindra', 'Toyota', 'Honda']
        }
        self.base_registrations = {'2W': 1000, '3W': 500, '4W': 200}
    
    def generate_data(self, years: int = 5) -> pd.DataFrame:
        """
        Generate synthetic vehicle registration data.
        
        Args:
            years: Number of years of data to generate
            
        Returns:
            DataFrame with vehicle registration data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        data = []
        for date in dates:
            year = date.year
            quarter = (date.month - 1) // 3 + 1
            
            for category in self.categories:
                base = self.base_registrations[category]
                
                for manufacturer in self.manufacturers[category]:
                    registrations = self._calculate_registrations(
                        date, year, quarter, category, manufacturer, base
                    )
                    
                    data.append({
                        'Date': date,
                        'Year': year,
                        'Quarter': f"Q{quarter}",
                        'Category': category,
                        'Manufacturer': manufacturer,
                        'Registrations': registrations
                    })
        
        return pd.DataFrame(data)
    
    def _calculate_registrations(self, date: datetime, year: int, quarter: int, 
                               category: str, manufacturer: str, base: int) -> int:
        """Calculate registration numbers with realistic patterns."""
        # Annual growth factor
        annual_growth = 1 + 0.1 * (year - 2020)  # 10% annual growth
        
        # Quarterly variation (higher in Q4 due to festive season)
        quarterly_variation = 1 + 0.02 * quarter + (0.1 if quarter == 4 else 0)
        
        # Manufacturer-specific factor
        manufacturer_factor = 0.8 + 0.4 * ((hash(manufacturer + category) % 100) / 100)
        
        # Seasonal variation (higher in certain months)
        seasonal_factor = 1 + 0.05 * np.sin(2 * np.pi * date.month / 12)
        
        # Random noise for realism
        noise_factor = 1 + np.random.normal(0, 0.05)
        
        registrations = int(
            base * annual_growth * quarterly_variation * 
            manufacturer_factor * seasonal_factor * noise_factor
        )
        
        return max(0, registrations)  # Ensure non-negative values
