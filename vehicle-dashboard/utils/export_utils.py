"""
Export utilities for generating CSV and PDF reports.
Provides functionality to export dashboard data and visualizations.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
import base64
from typing import Dict, List, Optional
import streamlit as st


class ExportManager:
    """Handles export functionality for dashboard data and reports."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for PDF generation."""
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12
        )
        
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )
    
    def export_to_csv(self, df: pd.DataFrame, filename: str = "vehicle_data.csv") -> bytes:
        """
        Export DataFrame to CSV format.
        
        Args:
            df: DataFrame to export
            filename: Name of the file
            
        Returns:
            CSV data as bytes
        """
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        csv_buffer.close()
        return csv_data
    
    def create_summary_table(self, df: pd.DataFrame) -> Table:
        """Create a summary table for PDF export."""
        # Calculate summary statistics
        total_registrations = df['Registrations'].sum()
        avg_monthly = df.groupby('Date')['Registrations'].sum().mean()
        categories = df['Category'].nunique()
        manufacturers = df['Manufacturer'].nunique()
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Registrations', f"{total_registrations:,}"],
            ['Average Monthly Registrations', f"{avg_monthly:,.0f}"],
            ['Number of Categories', str(categories)],
            ['Number of Manufacturers', str(manufacturers)],
            ['Date Range', f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"]
        ]
        
        table = Table(summary_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def create_category_breakdown_table(self, df: pd.DataFrame) -> Table:
        """Create category breakdown table for PDF export."""
        category_data = df.groupby('Category')['Registrations'].sum().reset_index()
        category_data['Percentage'] = (category_data['Registrations'] / category_data['Registrations'].sum() * 100).round(2)
        
        table_data = [['Category', 'Total Registrations', 'Percentage (%)']]
        for _, row in category_data.iterrows():
            table_data.append([
                row['Category'],
                f"{row['Registrations']:,}",
                f"{row['Percentage']:.1f}%"
            ])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def create_manufacturer_breakdown_table(self, df: pd.DataFrame, top_n: int = 10) -> Table:
        """Create manufacturer breakdown table for PDF export."""
        manufacturer_data = df.groupby('Manufacturer')['Registrations'].sum().reset_index()
        manufacturer_data = manufacturer_data.nlargest(top_n, 'Registrations')
        manufacturer_data['Percentage'] = (manufacturer_data['Registrations'] / df['Registrations'].sum() * 100).round(2)
        
        table_data = [['Manufacturer', 'Total Registrations', 'Percentage (%)']]
        for _, row in manufacturer_data.iterrows():
            table_data.append([
                row['Manufacturer'],
                f"{row['Registrations']:,}",
                f"{row['Percentage']:.1f}%"
            ])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def export_to_pdf(self, df: pd.DataFrame, title: str = "Vehicle Registration Report", 
                     include_charts: bool = False) -> bytes:
        """
        Export DataFrame and analysis to PDF format.
        
        Args:
            df: DataFrame to export
            title: Title of the report
            include_charts: Whether to include chart images
            
        Returns:
            PDF data as bytes
        """
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        story = []
        
        # Title
        story.append(Paragraph(title, self.title_style))
        story.append(Spacer(1, 20))
        
        # Summary section
        story.append(Paragraph("Executive Summary", self.heading_style))
        story.append(self.create_summary_table(df))
        story.append(Spacer(1, 20))
        
        # Category breakdown
        story.append(Paragraph("Category Breakdown", self.heading_style))
        story.append(self.create_category_breakdown_table(df))
        story.append(Spacer(1, 20))
        
        # Top manufacturers
        story.append(Paragraph("Top Manufacturers", self.heading_style))
        story.append(self.create_manufacturer_breakdown_table(df))
        story.append(Spacer(1, 20))
        
        # Data table (first 50 rows)
        story.append(Paragraph("Sample Data", self.heading_style))
        sample_df = df.head(50)
        
        # Prepare data for table
        table_data = [list(sample_df.columns)]
        for _, row in sample_df.iterrows():
            table_data.append([
                row['Date'].strftime('%Y-%m-%d'),
                row['Year'],
                row['Quarter'],
                row['Category'],
                row['Manufacturer'],
                f"{row['Registrations']:,}"
            ])
        
        data_table = Table(table_data)
        data_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(data_table)
        
        # Build PDF
        doc.build(story)
        pdf_data = pdf_buffer.getvalue()
        pdf_buffer.close()
        
        return pdf_data
    
    def get_download_button(self, data: bytes, filename: str, label: str, 
                          mime_type: str = "application/octet-stream"):
        """Create a Streamlit download button."""
        return st.download_button(
            label=label,
            data=data,
            file_name=filename,
            mime=mime_type
        )
