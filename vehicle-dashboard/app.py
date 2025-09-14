"""
Enhanced Vehicle Registration Dashboard
An interactive dashboard for analyzing vehicle registration trends with investor perspective.
Features: ARIMA forecasting, advanced analytics, export capabilities, and modular architecture.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Import custom modules
from utils.data_generator import VehicleDataGenerator
from utils.analytics import VehicleAnalytics
from utils.forecasting import VehicleForecasting
from utils.export_utils import ExportManager

# Page configuration
st.set_page_config(
    page_title="Vehicle Registration Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .forecast-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .export-section {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize classes
@st.cache_data
def load_data():
    """Load vehicle registration data."""
    generator = VehicleDataGenerator()
    return generator.generate_data(years=5)

# Load data
df = load_data()
analytics = VehicleAnalytics(df)
forecasting = VehicleForecasting(df)
export_manager = ExportManager()

# Header
st.markdown('<h1 class="main-header">🚗 Vehicle Registration Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Advanced analytics and forecasting for vehicle registration trends from an investor's perspective
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header('🔍 Filters & Controls')

# Date range filter
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
date_range = st.sidebar.date_input(
    '📅 Select Date Range',
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Choose the time period for analysis"
)

# Category filter
categories = sorted(df['Category'].unique())
selected_categories = st.sidebar.multiselect(
    '🚙 Select Vehicle Categories',
    options=categories,
    default=categories,
    help="Filter by vehicle types (2W, 3W, 4W)"
)

# Manufacturer filter
manufacturers = sorted(df['Manufacturer'].unique())
selected_manufacturers = st.sidebar.multiselect(
    '🏭 Select Manufacturers',
    options=manufacturers,
    default=manufacturers,
    help="Filter by specific manufacturers"
)

# Forecasting controls
st.sidebar.header('🔮 Forecasting')
forecast_periods = st.sidebar.slider(
    'Forecast Periods (Months)',
    min_value=3,
    max_value=24,
    value=12,
    help="Number of months to forecast ahead"
)

enable_forecasting = st.sidebar.checkbox(
    'Enable ARIMA Forecasting',
    value=True,
    help="Enable time-series forecasting using ARIMA model"
)

# Apply filters
filtered_df = df[
    (df['Date'].dt.date >= date_range[0]) &
    (df['Date'].dt.date <= date_range[1]) &
    (df['Category'].isin(selected_categories)) &
    (df['Manufacturer'].isin(selected_manufacturers))
]

# Update analytics with filtered data
analytics_filtered = VehicleAnalytics(filtered_df)
forecasting_filtered = VehicleForecasting(filtered_df)

# Main dashboard content
if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filters. Please adjust your selection.")
    st.stop()

# Key Performance Indicators
st.subheader('📊 Key Performance Indicators')
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_reg = filtered_df['Registrations'].sum()
    st.metric("Total Registrations", f"{total_reg:,}")

with col2:
    yoy_growth = analytics_filtered.calculate_yoy_growth()
    st.metric("YoY Growth", f"{yoy_growth:.1f}%")

with col3:
    qoq_growth = analytics_filtered.calculate_qoq_growth()
    st.metric("QoQ Growth", f"{qoq_growth:.1f}%")

with col4:
    avg_monthly = filtered_df.groupby('Date')['Registrations'].sum().mean()
    st.metric("Avg Monthly", f"{avg_monthly:,.0f}")

# Advanced Analytics Section
st.subheader('📈 Advanced Analytics')

# Category Performance Analysis
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Category Performance")
    category_perf = analytics_filtered.get_category_performance()
    
    fig_category = px.bar(
        category_perf, 
        x='Category', 
        y='Total_Registrations',
        color='YoY_Growth',
        color_continuous_scale='RdYlGn',
        title='Total Registrations by Category',
        labels={'Total_Registrations': 'Total Registrations', 'YoY_Growth': 'YoY Growth (%)'}
    )
    fig_category.update_layout(showlegend=False)
    st.plotly_chart(fig_category, use_container_width=True)

with col2:
    st.markdown("#### Top Manufacturers")
    manufacturer_perf = analytics_filtered.get_manufacturer_performance().head(10)
    
    fig_manufacturer = px.bar(
        manufacturer_perf, 
        x='Total_Registrations', 
        y='Manufacturer',
        orientation='h',
        color='YoY_Growth',
        color_continuous_scale='RdYlGn',
        title='Top 10 Manufacturers by Registrations',
        labels={'Total_Registrations': 'Total Registrations', 'YoY_Growth': 'YoY Growth (%)'}
    )
    fig_manufacturer.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_manufacturer, use_container_width=True)

# Time Series Analysis
st.subheader('📊 Time Series Analysis')

# Registration trends
trend_data = analytics_filtered.get_trend_data()
fig_trend = go.Figure()

for category in trend_data['Category'].unique():
    category_data = trend_data[trend_data['Category'] == category]
    fig_trend.add_trace(go.Scatter(
        x=category_data['Date'],
        y=category_data['Registrations'],
        mode='lines+markers',
        name=category,
        line=dict(width=3),
        marker=dict(size=6)
    ))

fig_trend.update_layout(
    title='Monthly Vehicle Registrations by Category',
    xaxis_title='Date',
    yaxis_title='Registrations',
    hovermode='x unified',
    height=500,
    template='plotly_white'
)

# Add forecasting if enabled
if enable_forecasting:
    st.markdown("#### 🔮 ARIMA Forecasting")
    
    # Forecast for each category
    forecast_cols = st.columns(len(selected_categories))
    
    for i, category in enumerate(selected_categories):
        with forecast_cols[i]:
            st.markdown(f"**{category} Forecast**")
            
            # Get forecast
            forecast_result = forecasting_filtered.forecast_registrations(
                category=category, 
                periods=forecast_periods
            )
            
            if forecast_result['error'] is None:
                # Create forecast visualization
                fig_forecast = go.Figure()
                
                # Historical data
                hist_data = trend_data[trend_data['Category'] == category]
                fig_forecast.add_trace(go.Scatter(
                    x=hist_data['Date'],
                    y=hist_data['Registrations'],
                    mode='lines+markers',
                    name=f'{category} Historical',
                    line=dict(color='blue', width=2)
                ))
                
                # Forecast data
                forecast_data = forecast_result['forecast']
                conf_int = forecast_result['confidence_intervals']
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data.values,
                    mode='lines+markers',
                    name=f'{category} Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Confidence interval
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=conf_int.iloc[:, 1],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=conf_int.iloc[:, 0],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    name='Confidence Interval',
                    hoverinfo='skip'
                ))
                
                fig_forecast.update_layout(
                    title=f'{category} Forecast (ARIMA)',
                    xaxis_title='Date',
                    yaxis_title='Registrations',
                    height=300,
                    showlegend=True
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Display forecast accuracy
                accuracy = forecasting_filtered.get_forecast_accuracy(category=category)
                if 'error' not in accuracy:
                    st.markdown(f"""
                    **Forecast Accuracy:**
                    - RMSE: {accuracy['rmse']:.0f}
                    - MAPE: {accuracy['mape']:.1f}%
                    """)
            else:
                st.error(f"Forecasting error: {forecast_result['error']}")

# Seasonal Analysis
st.subheader('📅 Seasonal Analysis')

seasonal_patterns = analytics_filtered.get_seasonal_patterns()
if seasonal_patterns:
    fig_seasonal = go.Figure()
    
    for category, data in seasonal_patterns.items():
        fig_seasonal.add_trace(go.Scatter(
            x=data['Month'],
            y=data['Registrations'],
            mode='lines+markers',
            name=category,
            line=dict(width=3)
        ))
    
    fig_seasonal.update_layout(
        title='Seasonal Patterns by Category',
        xaxis_title='Month',
        yaxis_title='Average Registrations',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig_seasonal, use_container_width=True)

# Export Section
st.subheader('📤 Export & Reports')

export_col1, export_col2 = st.columns(2)

with export_col1:
    st.markdown("#### Export Data")
    
    # CSV Export
    csv_data = export_manager.export_to_csv(filtered_df, "vehicle_registration_data.csv")
    export_manager.get_download_button(
        csv_data, 
        "vehicle_data.csv", 
        "📊 Download CSV",
        "text/csv"
    )
    
    # PDF Report
    pdf_data = export_manager.export_to_pdf(
        filtered_df, 
        "Vehicle Registration Analysis Report"
    )
    export_manager.get_download_button(
        pdf_data, 
        "vehicle_report.pdf", 
        "📄 Download PDF Report",
        "application/pdf"
    )

with export_col2:
    st.markdown("#### Quick Stats")
    
    # Display key statistics
    st.markdown(f"""
    **Data Summary:**
    - Total Records: {len(filtered_df):,}
    - Date Range: {filtered_df['Date'].min().strftime('%Y-%m-%d')} to {filtered_df['Date'].max().strftime('%Y-%m-%d')}
    - Categories: {filtered_df['Category'].nunique()}
    - Manufacturers: {filtered_df['Manufacturer'].nunique()}
    - Avg Monthly Growth: {qoq_growth:.1f}%
    """)

# Raw Data View
if st.checkbox('📋 Show Raw Data'):
    st.subheader('Raw Data')
    st.dataframe(
        filtered_df.sort_values('Date', ascending=False), 
        use_container_width=True,
        height=400
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🚗 Vehicle Registration Dashboard | Built with Streamlit, Plotly, and ARIMA Forecasting</p>
    <p><em>Data Source: Synthetic data for demonstration purposes</em></p>
</div>
""", unsafe_allow_html=True)