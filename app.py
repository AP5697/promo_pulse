"""
Promo Pulse Dashboard - Main Streamlit Application
UAE Retail Analytics & Promotion Simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import io

# Import custom modules
from data_generator import generate_all_data
from cleaner import DataCleaner
from simulator import KPICalculator, PromoSimulator, format_number_short, format_percentage

# ============== PAGE CONFIGURATION ==============
st.set_page_config(
    page_title="Promo Pulse Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #4361ee;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .dashboard-intro {
        background: linear-gradient(135deg, #4361ee 0%, #3730a3 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    
    .dashboard-intro h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .dashboard-intro p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ============== SESSION STATE INITIALIZATION ==============
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'data_loaded': False,
        'use_clean_data': True,
        'current_view': 'Manager',
        'uploaded_data': None,
        'raw_data': {},
        'clean_data': {},
        'issues_log': pd.DataFrame(),
        'cleaning_summary': {},
        'use_uploaded_data': False,
        'filters': {},
        'upload_ready': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_or_generate_data():
    """Load existing data or generate new synthetic data"""
    raw_dir = 'data/raw'
    clean_dir = 'data/clean'
    
    if not os.path.exists(f'{raw_dir}/sales_raw.csv'):
        with st.spinner('Generating synthetic data with intentional errors...'):
            generate_all_data(raw_dir)
    
    if not os.path.exists(f'{clean_dir}/sales_clean.csv'):
        with st.spinner('Cleaning data and generating error logs...'):
            cleaner = DataCleaner(clean_dir)
            result = cleaner.clean_all_data(raw_dir)
            st.session_state.cleaning_summary = result.get('summary', {})
    
    st.session_state.raw_data = {
        'products': pd.read_csv(f'{raw_dir}/products_raw.csv'),
        'stores': pd.read_csv(f'{raw_dir}/stores_raw.csv'),
        'customers': pd.read_csv(f'{raw_dir}/customers_raw.csv'),
        'sales': pd.read_csv(f'{raw_dir}/sales_raw.csv'),
        'inventory': pd.read_csv(f'{raw_dir}/inventory_raw.csv'),
        'campaigns': pd.read_csv(f'{raw_dir}/campaigns_raw.csv'),
        'departments': pd.read_csv(f'{raw_dir}/departments.csv')
    }
    
    st.session_state.clean_data = {
        'products': pd.read_csv(f'{clean_dir}/products_clean.csv'),
        'stores': pd.read_csv(f'{clean_dir}/stores_clean.csv'),
        'customers': pd.read_csv(f'{clean_dir}/customers_clean.csv'),
        'sales': pd.read_csv(f'{clean_dir}/sales_clean.csv'),
        'inventory': pd.read_csv(f'{clean_dir}/inventory_clean.csv'),
        'campaigns': pd.read_csv(f'{clean_dir}/campaigns_clean.csv')
    }
    
    if os.path.exists(f'{clean_dir}/logs/issues_log.csv'):
        st.session_state.issues_log = pd.read_csv(f'{clean_dir}/logs/issues_log.csv')
    
    st.session_state.data_loaded = True


def render_kpi_card(col, title: str, value):
    """Render a KPI card using Streamlit's metric component - NO reference"""
    with col:
        value_str = str(value) if value is not None else "N/A"
        st.metric(label=title, value=value_str)


def get_data():
    """Get currently selected data"""
    if st.session_state.use_uploaded_data and st.session_state.upload_ready and st.session_state.uploaded_data:
        return st.session_state.uploaded_data
    elif st.session_state.use_clean_data:
        return st.session_state.clean_data
    return st.session_state.raw_data


def validate_uploaded_data(data: Dict) -> tuple:
    """Validate uploaded data has required tables and columns"""
    required_tables = ['sales', 'products', 'stores', 'inventory']
    missing_tables = [t for t in required_tables if t not in data]
    
    if missing_tables:
        return False, f"Missing required tables: {', '.join(missing_tables)}"
    
    required_columns = {
        'sales': ['order_id', 'product_id', 'store_id', 'qty', 'selling_price_aed'],
        'products': ['product_id', 'category'],
        'stores': ['store_id', 'city', 'channel'],
        'inventory': ['product_id', 'store_id', 'stock_on_hand']
    }
    
    for table, cols in required_columns.items():
        if table in data:
            missing_cols = [c for c in cols if c not in data[table].columns]
            if missing_cols:
                return False, f"Table '{table}' missing columns: {', '.join(missing_cols)}"
    
    return True, "Data validation passed"


def clean_data_for_dashboard(data: Dict) -> tuple:
    """Clean uploaded data and return cleaned data with issues log"""
    issues = []
    cleaned_data = {}
    
    for table_name, df in data.items():
        df_clean = df.copy()
        
        # Clean city names if present
        if 'city' in df_clean.columns:
            city_corrections = {
                'dubaai': 'Dubai', 'dubay': 'Dubai', 'dubaii': 'Dubai', 
                'dubai': 'Dubai', 'DUBAI': 'Dubai', 'dxb': 'Dubai',
                'abu dhabi': 'Abu Dhabi', 'ABU DHABI': 'Abu Dhabi', 
                'abudhabi': 'Abu Dhabi', 'AbuDhabi': 'Abu Dhabi',
                'SHARJAH': 'Sharjah', 'sharjah': 'Sharjah', 'sharjha': 'Sharjah'
            }
            for old_val, new_val in city_corrections.items():
                mask = df_clean['city'].astype(str).str.lower() == old_val.lower()
                if mask.any():
                    count = mask.sum()
                    issues.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'table': table_name,
                        'record_id': 'Multiple',
                        'issue_type': 'INVALID_CITY_NAME',
                        'issue_detail': f'{count} city names corrected to {new_val}',
                        'original_value': old_val,
                        'corrected_value': new_val,
                        'action_taken': 'Corrected',
                        'department': 'Sales Operations'
                    })
                    df_clean.loc[mask, 'city'] = new_val
        
        # Fix year if present in date columns
        date_cols = [c for c in df_clean.columns if 'date' in c.lower() or 'time' in c.lower()]
        for col in date_cols:
            try:
                dates = pd.to_datetime(df_clean[col], errors='coerce')
                mask_2025 = dates.dt.year == 2025
                if mask_2025.any():
                    count = mask_2025.sum()
                    issues.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'table': table_name,
                        'record_id': 'Multiple',
                        'issue_type': 'WRONG_YEAR',
                        'issue_detail': f'{count} dates with year 2025 corrected to 2024',
                        'original_value': '2025',
                        'corrected_value': '2024',
                        'action_taken': 'Corrected',
                        'department': 'Sales Operations'
                    })
                    corrected_dates = dates.apply(
                        lambda x: x.replace(year=2024) if pd.notna(x) and x.year == 2025 else x
                    )
                    df_clean[col] = corrected_dates
            except:
                pass
        
        # Handle missing values in numeric columns
        for col in df_clean.columns:
            null_count = df_clean[col].isna().sum()
            if null_count > 0 and col not in ['customer_id', 'notes', 'comments']:
                if df_clean[col].dtype in ['float64', 'int64']:
                    median_val = df_clean[col].median()
                    if pd.notna(median_val):
                        df_clean[col].fillna(median_val, inplace=True)
                        issues.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'table': table_name,
                            'record_id': 'Multiple',
                            'issue_type': 'MISSING_VALUE',
                            'issue_detail': f'{null_count} missing values in {col}',
                            'original_value': 'NULL',
                            'corrected_value': str(round(median_val, 2)),
                            'action_taken': 'Imputed',
                            'department': 'Data Quality'
                        })
        
        # Add payment_status if missing (needed for KPI calculations)
        if table_name == 'sales' and 'payment_status' not in df_clean.columns:
            df_clean['payment_status'] = 'Paid'
        
        # Add reorder_point if missing in inventory
        if table_name == 'inventory' and 'reorder_point' not in df_clean.columns:
            df_clean['reorder_point'] = 10
        
        # Add snapshot_date if missing in inventory
        if table_name == 'inventory' and 'snapshot_date' not in df_clean.columns:
            df_clean['snapshot_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Add unit_cost_aed if missing in products
        if table_name == 'products' and 'unit_cost_aed' not in df_clean.columns:
            if 'base_price_aed' in df_clean.columns:
                df_clean['unit_cost_aed'] = df_clean['base_price_aed'] * 0.6
            else:
                df_clean['unit_cost_aed'] = 50
        
        # Add base_price_aed if missing in products
        if table_name == 'products' and 'base_price_aed' not in df_clean.columns:
            df_clean['base_price_aed'] = 100
        
        cleaned_data[table_name] = df_clean
    
    issues_df = pd.DataFrame(issues) if issues else pd.DataFrame()
    return cleaned_data, issues_df


# ============== SIDEBAR ==============
def render_sidebar():
    """Render the sidebar with filters and controls"""
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        st.markdown("### 📁 Data Source")
        data_source = st.radio(
            "Select Dataset",
            ["Pre-built (Synthetic)", "Upload Custom"],
            key="data_source_radio"
        )
        
        if data_source == "Upload Custom":
            render_upload_section()
        else:
            st.session_state.use_uploaded_data = False
            st.session_state.upload_ready = False
        
        st.markdown("---")
        
        if data_source == "Pre-built (Synthetic)":
            st.markdown("### 🔄 Data Type")
            use_clean = st.toggle("Use Cleaned Data", value=True, key="clean_toggle")
            st.session_state.use_clean_data = use_clean
            
            if use_clean:
                st.success("✓ Using cleaned data")
            else:
                st.warning("⚠ Using raw data (may contain errors)")
            
            st.markdown("---")
        
        st.markdown("### 👁️ Dashboard View")
        view = st.radio("Select View", ["Manager", "Executive"], key="view_toggle")
        st.session_state.current_view = view
        
        st.markdown("---")
        
        st.markdown("### 🔍 Filters")
        
        data = get_data()
        
        if data and 'sales' in data and len(data['sales']) > 0:
            if 'stores' in data and len(data['stores']) > 0:
                stores_df = data['stores']
                cities = ['All'] + sorted([str(c) for c in stores_df['city'].dropna().unique()])
                channels = ['All'] + sorted([str(c) for c in stores_df['channel'].dropna().unique()])
            else:
                cities = ['All']
                channels = ['All']
            
            if 'products' in data and len(data['products']) > 0:
                products_df = data['products']
                categories = ['All'] + sorted([str(c) for c in products_df['category'].dropna().unique()])
                brands_list = [str(b) for b in products_df['brand'].dropna().unique()] if 'brand' in products_df.columns else []
                brands = ['All'] + sorted(brands_list)[:20]
            else:
                categories = ['All']
                brands = ['All']
            
            selected_city = st.selectbox("City", cities, key="filter_city")
            selected_channel = st.selectbox("Channel", channels, key="filter_channel")
            selected_category = st.selectbox("Category", categories, key="filter_category")
            selected_brand = st.selectbox("Brand", brands, key="filter_brand")
            
            st.markdown("#### Date Range")
            date_start = st.date_input("From", datetime(2024, 1, 1), key="filter_date_start")
            date_end = st.date_input("To", datetime(2024, 12, 31), key="filter_date_end")
            
            st.session_state.filters = {
                'city': selected_city if selected_city != 'All' else None,
                'channel': selected_channel if selected_channel != 'All' else None,
                'category': selected_category if selected_category != 'All' else None,
                'brand': selected_brand if selected_brand != 'All' else None,
                'date_start': str(date_start),
                'date_end': str(date_end)
            }
        else:
            st.session_state.filters = {}
        
        st.markdown("---")
        
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Regenerate Data", key="regen_btn"):
            with st.spinner("Regenerating data..."):
                generate_all_data('data/raw')
                cleaner = DataCleaner('data/clean')
                cleaner.clean_all_data('data/raw')
                st.session_state.data_loaded = False
                st.rerun()


def render_upload_section():
    """Render file upload section for custom datasets"""
    st.markdown("#### Upload Datasets")
    st.info("📌 Required: sales, products, stores, inventory")
    
    uploaded_files = st.file_uploader(
        "Upload CSV/Excel files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        uploaded_data = {}
        for file in uploaded_files:
            try:
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                
                file_key = file.name.split('.')[0].lower().replace('_raw', '').replace('_clean', '')
                
                key_mapping = {
                    'sales': 'sales', 'products': 'products', 'stores': 'stores',
                    'inventory': 'inventory', 'customers': 'customers', 'campaigns': 'campaigns'
                }
                
                mapped_key = key_mapping.get(file_key, file_key)
                uploaded_data[mapped_key] = df
                st.success(f"✓ {file.name} → {mapped_key} ({len(df):,} rows)")
            except Exception as e:
                st.error(f"Error: {file.name} - {str(e)}")
        
        if uploaded_data:
            is_valid, message = validate_uploaded_data(uploaded_data)
            
            if not is_valid:
                st.error(f"❌ {message}")
                st.session_state.upload_ready = False
            else:
                st.success("✓ All required tables found!")
                
                if st.button("🧹 Clean & Use Data", key="clean_uploaded_btn", type="primary"):
                    with st.spinner("Cleaning uploaded data..."):
                        try:
                            cleaned_data, issues = clean_data_for_dashboard(uploaded_data)
                            st.session_state.uploaded_data = cleaned_data
                            st.session_state.issues_log = issues
                            st.session_state.use_uploaded_data = True
                            st.session_state.upload_ready = True
                            st.success(f"✓ Cleaned! {len(issues)} issues fixed.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Cleaning error: {str(e)}")
                            st.session_state.upload_ready = False


# ============== VISUALIZATION FUNCTIONS ==============
def create_bcg_matrix(data: Dict) -> go.Figure:
    """Create BCG Matrix for channel/category analysis"""
    try:
        sales_df = data['sales'].merge(
            data['products'][['product_id', 'category']],
            on='product_id', how='left'
        ).merge(
            data['stores'][['store_id', 'city', 'channel']],
            on='store_id', how='left'
        )
        
        paid_sales = sales_df[sales_df['payment_status'] == 'Paid'] if 'payment_status' in sales_df.columns else sales_df
        
        channel_metrics = paid_sales.groupby('channel').agg({
            'selling_price_aed': 'sum',
            'qty': 'sum',
            'order_id': 'nunique'
        }).reset_index()
        
        channel_metrics.columns = ['channel', 'revenue', 'units_sold', 'orders']
        
        total_revenue = channel_metrics['revenue'].sum()
        channel_metrics['market_share'] = channel_metrics['revenue'] / total_revenue * 100
        
        np.random.seed(42)
        n_channels = len(channel_metrics)
        channel_metrics['growth_rate'] = [15 + np.random.randn() * 5 for _ in range(n_channels)]
        
        fig = go.Figure()
        
        fig.add_shape(type="rect", x0=0, y0=10, x1=50, y1=30, fillcolor="rgba(16, 185, 129, 0.1)", line_width=0)
        fig.add_shape(type="rect", x0=50, y0=10, x1=100, y1=30, fillcolor="rgba(59, 130, 246, 0.1)", line_width=0)
        fig.add_shape(type="rect", x0=0, y0=-10, x1=50, y1=10, fillcolor="rgba(156, 163, 175, 0.1)", line_width=0)
        fig.add_shape(type="rect", x0=50, y0=-10, x1=100, y1=10, fillcolor="rgba(245, 158, 11, 0.1)", line_width=0)
        
        fig.add_annotation(x=25, y=25, text="Question Marks", showarrow=False, font=dict(size=14, color="#6b7280"))
        fig.add_annotation(x=75, y=25, text="Stars ⭐", showarrow=False, font=dict(size=14, color="#6b7280"))
        fig.add_annotation(x=25, y=0, text="Dogs", showarrow=False, font=dict(size=14, color="#6b7280"))
        fig.add_annotation(x=75, y=0, text="Cash Cows 🐄", showarrow=False, font=dict(size=14, color="#6b7280"))
        
        colors = {'App': '#4361ee', 'Web': '#f72585', 'Marketplace': '#4cc9f0'}
        
        for _, row in channel_metrics.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['market_share']],
                y=[row['growth_rate']],
                mode='markers+text',
                marker=dict(size=row['revenue'] / total_revenue * 150, color=colors.get(row['channel'], '#6b7280'), opacity=0.7),
                text=row['channel'],
                textposition='top center',
                name=row['channel']
            ))
        
        fig.update_layout(
            title=dict(text="BCG Matrix - Channel Performance", font=dict(size=18)),
            xaxis_title="Relative Market Share (%)",
            yaxis_title="Market Growth Rate (%)",
            showlegend=True,
            height=500,
            xaxis=dict(range=[0, 100], dtick=25),
            yaxis=dict(range=[-10, 35], dtick=10),
            plot_bgcolor='white'
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig


def create_sunburst_chart(data: Dict) -> go.Figure:
    """Create sunburst chart for hierarchical revenue breakdown"""
    try:
        sales_df = data['sales'].merge(
            data['products'][['product_id', 'category']],
            on='product_id', how='left'
        ).merge(
            data['stores'][['store_id', 'city', 'channel']],
            on='store_id', how='left'
        )
        
        paid_sales = sales_df[sales_df['payment_status'] == 'Paid'] if 'payment_status' in sales_df.columns else sales_df
        
        hierarchy_data = paid_sales.groupby(['city', 'channel', 'category']).agg({
            'selling_price_aed': 'sum'
        }).reset_index()
        hierarchy_data.columns = ['city', 'channel', 'category', 'revenue']
        
        fig = px.sunburst(
            hierarchy_data,
            path=['city', 'channel', 'category'],
            values='revenue',
            color='revenue',
            color_continuous_scale='Blues',
            title='Revenue Hierarchy (City → Channel → Category)'
        )
        
        fig.update_layout(height=550)
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig


def create_heatmap(data: Dict) -> go.Figure:
    """Create heatmap of category performance by channel"""
    try:
        sales_df = data['sales'].merge(
            data['products'][['product_id', 'category']],
            on='product_id', how='left'
        ).merge(
            data['stores'][['store_id', 'channel']],
            on='store_id', how='left'
        )
        
        paid_sales = sales_df[sales_df['payment_status'] == 'Paid'] if 'payment_status' in sales_df.columns else sales_df
        
        pivot = paid_sales.pivot_table(
            values='selling_price_aed',
            index='category',
            columns='channel',
            aggfunc='sum',
            fill_value=0
        )
        
        text_matrix = pivot.apply(lambda x: x.apply(lambda v: format_number_short(v)))
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            text=text_matrix.values,
            texttemplate="%{text}",
            colorscale='RdYlGn'
        ))
        
        fig.update_layout(
            title=dict(text='Revenue Heatmap (Category × Channel)', font=dict(size=18)),
            xaxis_title='Channel',
            yaxis_title='Category',
            height=450
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig


def create_revenue_trend(data: Dict, filters: Dict = None) -> go.Figure:
    """Create revenue trend chart"""
    try:
        sales_df = data['sales'].copy()
        
        if 'order_time' in sales_df.columns:
            sales_df['date'] = pd.to_datetime(sales_df['order_time'], errors='coerce').dt.date
        
        if filters and filters.get('date_start'):
            sales_df = sales_df[sales_df['date'] >= pd.to_datetime(filters['date_start']).date()]
        if filters and filters.get('date_end'):
            sales_df = sales_df[sales_df['date'] <= pd.to_datetime(filters['date_end']).date()]
        
        paid_sales = sales_df[sales_df['payment_status'] == 'Paid'] if 'payment_status' in sales_df.columns else sales_df
        
        daily_revenue = paid_sales.groupby('date').agg({'selling_price_aed': 'sum'}).reset_index()
        daily_revenue.columns = ['date', 'revenue']
        daily_revenue['revenue_ma7'] = daily_revenue['revenue'].rolling(7).mean()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=daily_revenue['revenue'],
            mode='lines',
            name='Daily Revenue',
            line=dict(color='rgba(67, 97, 238, 0.3)', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=daily_revenue['revenue_ma7'],
            mode='lines',
            name='7-Day Moving Avg',
            line=dict(color='#4361ee', width=3)
        ))
        
        fig.update_layout(
            title=dict(text='Daily Revenue Trend', font=dict(size=18)),
            xaxis_title='Date',
            yaxis_title='Revenue (AED)',
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            hovermode='x unified'
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig


def create_category_margin_chart(data: Dict) -> go.Figure:
    """Create margin by category chart"""
    try:
        sales_df = data['sales'].merge(
            data['products'][['product_id', 'category', 'unit_cost_aed']],
            on='product_id', how='left'
        )
        
        paid_sales = sales_df[sales_df['payment_status'] == 'Paid'] if 'payment_status' in sales_df.columns else sales_df
        
        paid_sales['revenue'] = paid_sales['selling_price_aed'] * paid_sales['qty']
        paid_sales['cogs'] = paid_sales['unit_cost_aed'] * paid_sales['qty']
        paid_sales['margin'] = paid_sales['revenue'] - paid_sales['cogs']
        
        category_margins = paid_sales.groupby('category').agg({
            'revenue': 'sum',
            'margin': 'sum'
        }).reset_index()
        
        category_margins['margin_pct'] = (category_margins['margin'] / category_margins['revenue'] * 100).round(1)
        category_margins = category_margins.sort_values('margin_pct', ascending=True)
        
        colors = ['#ef4444' if m < 20 else '#f59e0b' if m < 30 else '#10b981' for m in category_margins['margin_pct']]
        
        fig = go.Figure(go.Bar(
            x=category_margins['margin_pct'],
            y=category_margins['category'],
            orientation='h',
            marker_color=colors,
            text=category_margins['margin_pct'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside'
        ))
        
        fig.update_layout(
            title=dict(text='Gross Margin by Category', font=dict(size=18)),
            xaxis_title='Gross Margin %',
            yaxis_title='',
            height=400,
            showlegend=False
        )
        
        fig.add_vline(x=25, line_dash="dash", line_color="#6b7280", annotation_text="Target: 25%")
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig


def create_stockout_risk_chart(data: Dict) -> go.Figure:
    """Create stockout risk visualization"""
    try:
        inventory_df = data['inventory'].copy()
        inventory_df['snapshot_date'] = pd.to_datetime(inventory_df['snapshot_date'])
        latest_inv = inventory_df.sort_values('snapshot_date').groupby(['product_id', 'store_id']).last().reset_index()
        
        latest_inv = latest_inv.merge(data['stores'][['store_id', 'city', 'channel']], on='store_id', how='left')
        latest_inv['at_risk'] = latest_inv['stock_on_hand'] <= latest_inv['reorder_point']
        
        risk_by_city = latest_inv.groupby('city').agg({'at_risk': ['sum', 'count']}).reset_index()
        risk_by_city.columns = ['city', 'at_risk_count', 'total']
        risk_by_city['risk_pct'] = (risk_by_city['at_risk_count'] / risk_by_city['total'] * 100).round(1)
        
        risk_by_channel = latest_inv.groupby('channel').agg({'at_risk': ['sum', 'count']}).reset_index()
        risk_by_channel.columns = ['channel', 'at_risk_count', 'total']
        risk_by_channel['risk_pct'] = (risk_by_channel['at_risk_count'] / risk_by_channel['total'] * 100).round(1)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('By City', 'By Channel'))
        
        colors_city = ['#ef4444' if r > 15 else '#f59e0b' if r > 10 else '#10b981' for r in risk_by_city['risk_pct']]
        colors_channel = ['#ef4444' if r > 15 else '#f59e0b' if r > 10 else '#10b981' for r in risk_by_channel['risk_pct']]
        
        fig.add_trace(go.Bar(x=risk_by_city['city'], y=risk_by_city['risk_pct'], marker_color=colors_city,
                            text=risk_by_city['risk_pct'].apply(lambda x: f'{x:.1f}%'), textposition='outside', showlegend=False), row=1, col=1)
        
        fig.add_trace(go.Bar(x=risk_by_channel['channel'], y=risk_by_channel['risk_pct'], marker_color=colors_channel,
                            text=risk_by_channel['risk_pct'].apply(lambda x: f'{x:.1f}%'), textposition='outside', showlegend=False), row=1, col=2)
        
        fig.update_layout(title=dict(text='Stockout Risk Distribution', font=dict(size=18)), height=400)
        fig.update_yaxes(title_text='Risk %', row=1, col=1)
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig


def create_issues_pareto(issues_df: pd.DataFrame) -> go.Figure:
    """Create Pareto chart of data quality issues"""
    if len(issues_df) == 0 or 'issue_type' not in issues_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No issues found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    issue_counts = issues_df['issue_type'].value_counts().reset_index()
    issue_counts.columns = ['issue_type', 'count']
    issue_counts = issue_counts.sort_values('count', ascending=False)
    
    total = issue_counts['count'].sum()
    issue_counts['cumulative_pct'] = (issue_counts['count'].cumsum() / total * 100).round(1)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(x=issue_counts['issue_type'], y=issue_counts['count'], name='Count',
                        marker_color='#4361ee', text=issue_counts['count'], textposition='outside'), secondary_y=False)
    
    fig.add_trace(go.Scatter(x=issue_counts['issue_type'], y=issue_counts['cumulative_pct'], name='Cumulative %',
                            mode='lines+markers', line=dict(color='#ef4444', width=2), marker=dict(size=8)), secondary_y=True)
    
    fig.update_layout(title=dict(text='Issues Pareto Chart', font=dict(size=18)), height=400,
                     showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02))
    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
    
    return fig


def create_simulation_waterfall(summary: Dict) -> go.Figure:
    """Create waterfall chart for simulation financial breakdown"""
    fig = go.Figure(go.Waterfall(
        name="Financial Breakdown",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Revenue", "COGS", "Gross Margin", "Promo Spend", "Net Profit"],
        y=[
            summary['total_simulated_revenue'],
            -summary['total_simulated_cogs'],
            0,
            -summary['total_promo_spend'],
            0
        ],
        textposition="outside",
        text=[
            format_number_short(summary['total_simulated_revenue']),
            format_number_short(-summary['total_simulated_cogs']),
            format_number_short(summary['total_simulated_margin']),
            format_number_short(-summary['total_promo_spend']),
            format_number_short(summary['profit_proxy'])
        ],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#10b981"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#4361ee"}}
    ))
    
    fig.update_layout(title=dict(text="Simulation Financial Breakdown", font=dict(size=18)), showlegend=False, height=400)
    
    return fig


# ============== MAIN VIEW RENDERERS ==============
def render_comparison_view():
    """Render raw vs clean data comparison - side by side"""
    st.markdown("### 📊 Data Comparison: Raw vs Clean")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("🔴 **Raw Data (With Errors)**")
        
        if 'sales' in st.session_state.raw_data:
            raw_sales = st.session_state.raw_data['sales'].head(100)
            display_cols = ['order_id', 'order_time', 'product_id', 'qty', 'selling_price_aed', 'discount_pct']
            available_cols = [c for c in display_cols if c in raw_sales.columns]
            
            st.dataframe(raw_sales[available_cols], height=350, use_container_width=True)
            
            st.markdown("**Detected Issues:**")
            issues = st.session_state.issues_log
            if len(issues) > 0 and 'table' in issues.columns:
                sales_issues = issues[issues['table'] == 'sales']
                if len(sales_issues) > 0 and 'issue_type' in sales_issues.columns:
                    top_issues = sales_issues['issue_type'].value_counts().head(5)
                    for issue, count in top_issues.items():
                        st.warning(f"⚠️ {issue}: {count:,}")
    
    with col2:
        st.success("🟢 **Clean Data (Corrected)**")
        
        if 'sales' in st.session_state.clean_data:
            clean_sales = st.session_state.clean_data['sales'].head(100)
            display_cols = ['order_id', 'order_time', 'product_id', 'qty', 'selling_price_aed', 'discount_pct']
            available_cols = [c for c in display_cols if c in clean_sales.columns]
            
            st.dataframe(clean_sales[available_cols], height=350, use_container_width=True)
            
            st.markdown("**Cleaning Actions:**")
            issues = st.session_state.issues_log
            if len(issues) > 0 and 'action_taken' in issues.columns:
                action_counts = issues['action_taken'].value_counts()
                for action, count in action_counts.items():
                    st.success(f"✅ {action}: {count:,}")


def render_manager_view():
    """Render Manager/Operations view"""
    data = get_data()
    
    if not data or 'sales' not in data:
        st.warning("No data available. Please load data first.")
        return
    
    filters = st.session_state.get('filters', {})
    issues = st.session_state.issues_log
    
    kpi_calc = KPICalculator(
        data['sales'], data['products'], data['stores'], 
        data['inventory'], data.get('customers')
    )
    
    all_kpis = kpi_calc.get_all_kpis(filters, issues)
    
    st.markdown("### 📈 Manager Dashboard - Operational KPIs")
    
    inv_kpis = all_kpis['inventory']
    dq_kpis = all_kpis['data_quality']
    
    col1, col2, col3, col4 = st.columns(4)
    render_kpi_card(col1, "Stockout Risk", f"{inv_kpis['stockout_rate']:.1f}%")
    render_kpi_card(col2, "Return Rate", f"{dq_kpis['return_rate']:.1f}%")
    render_kpi_card(col3, "Payment Failure", f"{dq_kpis['payment_failure_rate']:.1f}%")
    render_kpi_card(col4, "High-Risk SKUs", f"{inv_kpis['low_stock_items']:,}")
    
    col1, col2, col3, col4 = st.columns(4)
    render_kpi_card(col1, "Total Stock Units", f"{inv_kpis['total_stock_units']:,}")
    render_kpi_card(col2, "Avg Stock/SKU", f"{inv_kpis['avg_stock_per_sku']:.0f}")
    render_kpi_card(col3, "Inventory Turnover", f"{inv_kpis['inventory_turnover']:.1f}x")
    render_kpi_card(col4, "Data Issues", f"{dq_kpis['total_data_issues']:,}")
    
    st.markdown("---")
    st.plotly_chart(create_stockout_risk_chart(data), use_container_width=True)
    
    st.markdown("---")
    st.plotly_chart(create_issues_pareto(issues), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🚨 Top 10 Stockout Risk Items")
    
    try:
        inventory_df = data['inventory'].copy()
        inventory_df['snapshot_date'] = pd.to_datetime(inventory_df['snapshot_date'])
        latest_inv = inventory_df.sort_values('snapshot_date').groupby(['product_id', 'store_id']).last().reset_index()
        latest_inv = latest_inv.merge(data['stores'][['store_id', 'city', 'channel']], on='store_id', how='left')
        latest_inv = latest_inv.merge(data['products'][['product_id', 'category']], on='product_id', how='left')
        latest_inv['stock_coverage'] = latest_inv['stock_on_hand'] / latest_inv['reorder_point'].replace(0, 1)
        
        risk_items = latest_inv[latest_inv['stock_on_hand'] <= latest_inv['reorder_point']].nsmallest(10, 'stock_coverage')
        
        if len(risk_items) > 0:
            display_cols = ['product_id', 'store_id', 'city', 'channel', 'category', 'stock_on_hand', 'reorder_point']
            available_cols = [c for c in display_cols if c in risk_items.columns]
            st.dataframe(risk_items[available_cols], use_container_width=True, hide_index=True)
        else:
            st.success("No items currently at stockout risk! ✓")
    except Exception as e:
        st.warning(f"Could not calculate risk items: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 🏢 Issues by Department")
    
    if len(issues) > 0 and 'department' in issues.columns:
        dept_issues = issues.groupby('department').agg({
            'issue_type': 'count',
            'action_taken': lambda x: (x == 'Corrected').sum()
        }).reset_index()
        dept_issues.columns = ['Department', 'Total Issues', 'Auto-Corrected']
        dept_issues['Manual Review Needed'] = dept_issues['Total Issues'] - dept_issues['Auto-Corrected']
        
        st.dataframe(dept_issues, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        fig = px.pie(dept_issues, values='Total Issues', names='Department', title='Issues Distribution by Department')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_executive_view():
    """Render Executive view"""
    data = get_data()
    
    if not data or 'sales' not in data:
        st.warning("No data available. Please load data first.")
        return
    
    filters = st.session_state.get('filters', {})
    issues = st.session_state.issues_log
    
    kpi_calc = KPICalculator(
        data['sales'], data['products'], data['stores'], 
        data['inventory'], data.get('customers')
    )
    
    all_kpis = kpi_calc.get_all_kpis(filters, issues)
    biz_kpis = all_kpis['business']
    promo_kpis = all_kpis['promotion']
    
    st.markdown("### 📊 Executive Dashboard - Financial KPIs")
    
    col1, col2, col3, col4 = st.columns(4)
    render_kpi_card(col1, "Net Revenue", format_number_short(biz_kpis['net_revenue']))
    render_kpi_card(col2, "Gross Margin", f"{biz_kpis['gross_margin_pct']:.1f}%")
    render_kpi_card(col3, "Avg Order Value", f"AED {biz_kpis['avg_order_value']:,.0f}")
    render_kpi_card(col4, "Revenue Growth", f"{biz_kpis['revenue_growth_pct']:+.1f}%")
    
    col1, col2, col3, col4 = st.columns(4)
    render_kpi_card(col1, "Total Orders", f"{biz_kpis['total_orders']:,}")
    render_kpi_card(col2, "Avg Discount", f"{biz_kpis['avg_discount_pct']:.1f}%")
    render_kpi_card(col3, "Promo ROI", f"{promo_kpis['promo_roi']:.1f}x")
    render_kpi_card(col4, "Budget Utilization", f"{promo_kpis['budget_utilization_pct']:.0f}%")
    
    st.markdown("---")
    st.plotly_chart(create_revenue_trend(data, filters), use_container_width=True)
    
    st.markdown("---")
    st.plotly_chart(create_category_margin_chart(data), use_container_width=True)
    
    st.markdown("---")
    
    try:
        sales_enriched = data['sales'].merge(data['stores'][['store_id', 'city']], on='store_id', how='left')
        paid_sales = sales_enriched[sales_enriched['payment_status'] == 'Paid'] if 'payment_status' in sales_enriched.columns else sales_enriched
        city_revenue = paid_sales.groupby('city')['selling_price_aed'].sum().reset_index()
        
        fig = px.bar(city_revenue, x='city', y='selling_price_aed', title='Revenue by City',
                    labels={'selling_price_aed': 'Revenue (AED)', 'city': 'City'},
                    color='city', color_discrete_sequence=['#4361ee', '#f72585', '#4cc9f0'])
        fig.update_layout(showlegend=False, height=400)
        fig.update_traces(texttemplate='AED %{y:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not create city chart: {str(e)}")
    
    st.markdown("---")
    
    try:
        sales_enriched = data['sales'].merge(data['stores'][['store_id', 'channel']], on='store_id', how='left')
        paid_sales = sales_enriched[sales_enriched['payment_status'] == 'Paid'] if 'payment_status' in sales_enriched.columns else sales_enriched
        channel_revenue = paid_sales.groupby('channel')['selling_price_aed'].sum().reset_index()
        
        fig = px.pie(channel_revenue, values='selling_price_aed', names='channel', title='Revenue Share by Channel',
                    color_discrete_sequence=['#4361ee', '#f72585', '#4cc9f0'])
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not create channel chart: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 📊 Strategic Analysis")
    
    st.plotly_chart(create_bcg_matrix(data), use_container_width=True)
    st.caption("BCG Matrix shows channel positioning based on market share and growth rate.")
    
    st.markdown("---")
    st.plotly_chart(create_sunburst_chart(data), use_container_width=True)
    st.caption("Sunburst chart shows hierarchical revenue breakdown from City → Channel → Category.")
    
    st.markdown("---")
    st.plotly_chart(create_heatmap(data), use_container_width=True)
    st.caption("Heatmap shows revenue intensity across categories and channels.")


def render_simulation_view():
    """Render promotional campaign simulation"""
    st.markdown("### 🎯 Promotional Campaign Simulator")
    
    data = get_data()
    
    if not data or 'sales' not in data:
        st.warning("No data available. Please load data first.")
        return
    
    st.markdown("#### Configure Campaign Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        discount_pct = st.slider("Discount %", 5, 50, 20, key="sim_discount")
    with col2:
        promo_budget = st.number_input("Promo Budget (AED)", 50000, 1000000, 200000, step=10000, key="sim_budget")
    with col3:
        margin_floor = st.slider("Margin Floor %", 5, 30, 15, key="sim_margin_floor")
    with col4:
        sim_days = st.selectbox("Simulation Period", [7, 14, 21, 30], index=1, key="sim_days")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cities = ['All'] + list(data['stores']['city'].dropna().unique())
        sim_city = st.selectbox("City", cities, key="sim_city")
    with col2:
        channels = ['All'] + list(data['stores']['channel'].dropna().unique())
        sim_channel = st.selectbox("Channel", channels, key="sim_channel")
    with col3:
        categories = ['All'] + list(data['products']['category'].dropna().unique())
        sim_category = st.selectbox("Category", categories, key="sim_category")
    
    if st.button("🚀 Run Simulation", type="primary", key="run_sim"):
        with st.spinner("Running simulation..."):
            simulator = PromoSimulator(data['sales'], data['products'], data['stores'], data['inventory'])
            
            results = simulator.run_simulation(
                discount_pct=discount_pct,
                promo_budget=promo_budget,
                margin_floor_pct=margin_floor,
                simulation_days=sim_days,
                city=sim_city,
                channel=sim_channel,
                category=sim_category
            )
            
            st.session_state.sim_results = results
            st.session_state.simulator = simulator
    
    if 'sim_results' in st.session_state:
        results = st.session_state.sim_results
        summary = results['summary']
        
        st.markdown("---")
        st.markdown("### 📊 Simulation Results")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        render_kpi_card(col1, "Projected Revenue", format_number_short(summary['total_simulated_revenue']))
        render_kpi_card(col2, "Projected Margin", f"{summary['total_margin_pct']:.1f}%")
        render_kpi_card(col3, "Promo Spend", format_number_short(summary['total_promo_spend']))
        render_kpi_card(col4, "Net Profit", format_number_short(summary['profit_proxy']))
        render_kpi_card(col5, "Stockout Risk", f"{summary['stockout_risk_pct']:.1f}%")
        
        if summary['constraint_violations'] > 0:
            st.warning(f"⚠️ {summary['constraint_violations']} constraint violations detected!")
        
        st.markdown("---")
        st.plotly_chart(create_simulation_waterfall(summary), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 💡 AI Recommendation")
        
        recommendation = st.session_state.simulator.generate_recommendation(results)
        st.info(recommendation)


def render_error_logs():
    """Render detailed error logs view"""
    st.markdown("### 📋 Data Quality Error Logs")
    
    issues = st.session_state.issues_log
    
    if len(issues) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tables = ['All'] + list(issues['table'].unique()) if 'table' in issues.columns else ['All']
            selected_table = st.selectbox("Filter by Table", tables, key="log_filter_table")
        with col2:
            types = ['All'] + list(issues['issue_type'].unique()) if 'issue_type' in issues.columns else ['All']
            selected_type = st.selectbox("Filter by Issue Type", types, key="log_filter_type")
        with col3:
            depts = ['All'] + list(issues['department'].unique()) if 'department' in issues.columns else ['All']
            selected_dept = st.selectbox("Filter by Department", depts, key="log_filter_dept")
        
        filtered_issues = issues.copy()
        if selected_table != 'All' and 'table' in filtered_issues.columns:
            filtered_issues = filtered_issues[filtered_issues['table'] == selected_table]
        if selected_type != 'All' and 'issue_type' in filtered_issues.columns:
            filtered_issues = filtered_issues[filtered_issues['issue_type'] == selected_type]
        if selected_dept != 'All' and 'department' in filtered_issues.columns:
            filtered_issues = filtered_issues[filtered_issues['department'] == selected_dept]
        
        st.markdown(f"**Showing {len(filtered_issues):,} of {len(issues):,} total issues**")
        
        st.dataframe(filtered_issues, use_container_width=True, hide_index=True, height=500)
        
        csv = filtered_issues.to_csv(index=False)
        st.download_button("📥 Download Filtered Log", csv, "filtered_issues.csv", "text/csv")
    else:
        st.info("No issues logged. Data appears clean or hasn't been processed yet.")


# ============== MAIN APPLICATION ==============
def main():
    """Main application entry point"""
    init_session_state()
    
    if not st.session_state.data_loaded:
        load_or_generate_data()
    
    render_sidebar()
    
    st.markdown("""
    <div class="dashboard-intro">
        <h1>📊 Promo Pulse Dashboard</h1>
        <p>UAE Retail Analytics & Promotion Simulator | Clean • Minimal • Business-Ready</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🎯 Simulation", "🔄 Data Comparison", "📋 Error Logs"])
    
    with tab1:
        if st.session_state.current_view == 'Manager':
            render_manager_view()
        else:
            render_executive_view()
    
    with tab2:
        render_simulation_view()
    
    with tab3:
        render_comparison_view()
    
    with tab4:
        render_error_logs()


if __name__ == "__main__":
    main()
