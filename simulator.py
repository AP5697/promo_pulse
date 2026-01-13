"""
Promo Pulse Dashboard - Simulation Engine
Rule-based promotional campaign simulator with KPI calculations
No ML/DL - uses transparent business rules for demand uplift
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============== UPLIFT CONFIGURATION ==============
# Channel sensitivity to discounts (higher = more responsive)
CHANNEL_SENSITIVITY = {
    'Marketplace': 1.5,  # Most price-sensitive
    'App': 1.3,
    'Web': 1.0  # Baseline
}

# Category sensitivity to discounts
CATEGORY_SENSITIVITY = {
    'Electronics': 1.4,
    'Fashion': 1.3,
    'Beauty': 1.2,
    'Sports': 1.2,
    'Home & Garden': 1.1,
    'Toys': 1.1,
    'Grocery': 0.8,  # Less elastic
    'Books': 0.7
}

# City market factors
CITY_FACTORS = {
    'Dubai': 1.2,  # Highest purchasing power
    'Abu Dhabi': 1.1,
    'Sharjah': 1.0
}

# Reference budgets and targets for KPIs
REFERENCE_VALUES = {
    'total_promo_budget': 1000000,  # AED 1M reference budget
    'target_margin_pct': 25,  # 25% target margin
    'target_stockout_rate': 5,  # 5% acceptable stockout rate
    'target_return_rate': 5,  # 5% acceptable return rate
    'target_payment_failure': 3,  # 3% acceptable payment failure
}


class KPICalculator:
    """Calculate business KPIs from sales and inventory data"""
    
    def __init__(self, sales_df: pd.DataFrame, products_df: pd.DataFrame, 
                 stores_df: pd.DataFrame, inventory_df: pd.DataFrame,
                 customers_df: pd.DataFrame = None):
        self.sales = sales_df.copy()
        self.products = products_df.copy()
        self.stores = stores_df.copy()
        self.inventory = inventory_df.copy()
        self.customers = customers_df.copy() if customers_df is not None else pd.DataFrame()
        
        # Merge data for calculations
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare merged datasets for KPI calculations"""
        # Merge sales with products and stores
        self.sales_enriched = self.sales.merge(
            self.products[['product_id', 'category', 'brand', 'base_price_aed', 'unit_cost_aed']],
            on='product_id', how='left'
        ).merge(
            self.stores[['store_id', 'city', 'channel', 'fulfillment_type']],
            on='store_id', how='left'
        )
        
        # Calculate line-level metrics
        self.sales_enriched['revenue'] = self.sales_enriched['selling_price_aed'] * self.sales_enriched['qty']
        self.sales_enriched['cogs'] = self.sales_enriched['unit_cost_aed'] * self.sales_enriched['qty']
        self.sales_enriched['gross_margin'] = self.sales_enriched['revenue'] - self.sales_enriched['cogs']
    
    def calculate_business_kpis(self, filters: Dict = None) -> Dict:
        """Calculate business/finance KPIs"""
        df = self._apply_filters(self.sales_enriched, filters)
        
        # Only consider paid orders for revenue
        paid_df = df[df['payment_status'] == 'Paid']
        refund_df = df[df['payment_status'] == 'Refunded']
        
        gross_revenue = paid_df['revenue'].sum()
        refund_amount = refund_df['revenue'].sum()
        net_revenue = gross_revenue - refund_amount
        cogs = paid_df['cogs'].sum()
        gross_margin_aed = net_revenue - cogs
        gross_margin_pct = (gross_margin_aed / net_revenue * 100) if net_revenue > 0 else 0
        avg_discount = df['discount_pct'].mean()
        avg_order_value = paid_df['revenue'].mean() if len(paid_df) > 0 else 0
        total_orders = len(paid_df)
        
        # Revenue growth (mock: compare first and second half of data)
        if len(paid_df) > 100:
            mid = len(paid_df) // 2
            first_half_rev = paid_df.iloc[:mid]['revenue'].sum()
            second_half_rev = paid_df.iloc[mid:]['revenue'].sum()
            revenue_growth = ((second_half_rev - first_half_rev) / first_half_rev * 100) if first_half_rev > 0 else 0
        else:
            revenue_growth = 0
        
        return {
            'gross_revenue': gross_revenue,
            'refund_amount': refund_amount,
            'net_revenue': net_revenue,
            'cogs': cogs,
            'gross_margin_aed': gross_margin_aed,
            'gross_margin_pct': round(gross_margin_pct, 2),
            'avg_discount_pct': round(avg_discount, 2),
            'avg_order_value': round(avg_order_value, 2),
            'total_orders': total_orders,
            'revenue_growth_pct': round(revenue_growth, 2),
            'reference_margin_target': REFERENCE_VALUES['target_margin_pct']
        }
    
    def calculate_inventory_kpis(self, filters: Dict = None) -> Dict:
        """Calculate inventory KPIs"""
        inv_df = self.inventory.copy()
        if filters:
            if 'store_id' in filters and filters['store_id']:
                inv_df = inv_df[inv_df['store_id'].isin(filters['store_id'])]
            if 'product_id' in filters and filters['product_id']:
                inv_df = inv_df[inv_df['product_id'].isin(filters['product_id'])]
        
        # Get latest snapshot per product-store
        latest_inv = inv_df.sort_values('snapshot_date').groupby(['product_id', 'store_id']).last().reset_index()
        
        total_stock = latest_inv['stock_on_hand'].sum()
        avg_stock = latest_inv['stock_on_hand'].mean()
        
        # Items below reorder point
        below_reorder = latest_inv[latest_inv['stock_on_hand'] <= latest_inv['reorder_point']]
        low_stock_items = len(below_reorder)
        low_stock_pct = (low_stock_items / len(latest_inv) * 100) if len(latest_inv) > 0 else 0
        
        # Zero stock items
        zero_stock = latest_inv[latest_inv['stock_on_hand'] == 0]
        zero_stock_items = len(zero_stock)
        stockout_rate = (zero_stock_items / len(latest_inv) * 100) if len(latest_inv) > 0 else 0
        
        # Inventory turnover (simplified)
        sales_qty = self._apply_filters(self.sales_enriched, filters)['qty'].sum()
        inventory_turnover = (sales_qty / avg_stock) if avg_stock > 0 else 0
        
        return {
            'total_stock_units': int(total_stock),
            'avg_stock_per_sku': round(avg_stock, 2),
            'low_stock_items': low_stock_items,
            'low_stock_pct': round(low_stock_pct, 2),
            'zero_stock_items': zero_stock_items,
            'stockout_rate': round(stockout_rate, 2),
            'inventory_turnover': round(inventory_turnover, 2),
            'reference_stockout_target': REFERENCE_VALUES['target_stockout_rate']
        }
    
    def calculate_channel_kpis(self, filters: Dict = None) -> Dict:
        """Calculate channel performance KPIs"""
        df = self._apply_filters(self.sales_enriched, filters)
        paid_df = df[df['payment_status'] == 'Paid']
        
        # Revenue by channel
        channel_revenue = paid_df.groupby('channel')['revenue'].sum().to_dict()
        total_revenue = sum(channel_revenue.values())
        
        channel_share = {k: round(v / total_revenue * 100, 2) if total_revenue > 0 else 0 
                        for k, v in channel_revenue.items()}
        
        # Orders by channel
        channel_orders = paid_df.groupby('channel')['order_id'].nunique().to_dict()
        
        # AOV by channel
        channel_aov = {}
        for channel in paid_df['channel'].unique():
            ch_df = paid_df[paid_df['channel'] == channel]
            channel_aov[channel] = round(ch_df['revenue'].sum() / len(ch_df), 2) if len(ch_df) > 0 else 0
        
        # Best performing channel
        best_channel = max(channel_revenue, key=channel_revenue.get) if channel_revenue else 'N/A'
        
        return {
            'revenue_by_channel': channel_revenue,
            'channel_share_pct': channel_share,
            'orders_by_channel': channel_orders,
            'aov_by_channel': channel_aov,
            'best_performing_channel': best_channel,
            'total_channels': len(channel_revenue)
        }
    
    def calculate_customer_kpis(self, filters: Dict = None) -> Dict:
        """Calculate customer KPIs"""
        df = self._apply_filters(self.sales_enriched, filters)
        
        # Unique customers
        unique_customers = df['customer_id'].nunique()
        
        # Orders per customer
        orders_per_customer = df.groupby('customer_id')['order_id'].nunique()
        avg_orders_per_customer = orders_per_customer.mean()
        
        # Revenue per customer
        revenue_per_customer = df[df['payment_status'] == 'Paid'].groupby('customer_id')['revenue'].sum()
        avg_revenue_per_customer = revenue_per_customer.mean()
        
        # Repeat customers (more than 1 order)
        repeat_customers = (orders_per_customer > 1).sum()
        repeat_rate = (repeat_customers / unique_customers * 100) if unique_customers > 0 else 0
        
        # Customer segments if available
        segment_breakdown = {}
        if 'customer_segment' in df.columns or (len(self.customers) > 0 and 'customer_segment' in self.customers.columns):
            if len(self.customers) > 0:
                df = df.merge(self.customers[['customer_id', 'customer_segment']], on='customer_id', how='left')
            if 'customer_segment' in df.columns:
                segment_breakdown = df.groupby('customer_segment')['customer_id'].nunique().to_dict()
        
        return {
            'unique_customers': unique_customers,
            'avg_orders_per_customer': round(avg_orders_per_customer, 2),
            'avg_revenue_per_customer': round(avg_revenue_per_customer, 2),
            'repeat_customers': repeat_customers,
            'repeat_rate_pct': round(repeat_rate, 2),
            'segment_breakdown': segment_breakdown
        }
    
    def calculate_promotion_kpis(self, filters: Dict = None, promo_budget: float = None) -> Dict:
        """Calculate promotion-specific KPIs"""
        df = self._apply_filters(self.sales_enriched, filters)
        
        # Discounted vs non-discounted
        discounted_sales = df[df['discount_pct'] > 0]
        non_discounted_sales = df[df['discount_pct'] == 0]
        
        discounted_revenue = discounted_sales['revenue'].sum()
        non_discounted_revenue = non_discounted_sales['revenue'].sum()
        
        # Discount efficiency
        total_discount_given = df['revenue'].sum() * df['discount_pct'].mean() / 100
        
        # Promo ROI (revenue generated per AED of discount)
        promo_roi = (discounted_revenue / total_discount_given) if total_discount_given > 0 else 0
        
        # Budget utilization
        budget = promo_budget or REFERENCE_VALUES['total_promo_budget']
        budget_utilization = (total_discount_given / budget * 100) if budget > 0 else 0
        
        return {
            'discounted_revenue': discounted_revenue,
            'non_discounted_revenue': non_discounted_revenue,
            'total_discount_given': round(total_discount_given, 2),
            'promo_roi': round(promo_roi, 2),
            'budget_utilized': round(total_discount_given, 2),
            'budget_utilization_pct': round(min(budget_utilization, 100), 2),
            'reference_budget': budget,
            'avg_discount_pct': round(df['discount_pct'].mean(), 2)
        }
    
    def calculate_data_quality_kpis(self, issues_df: pd.DataFrame = None) -> Dict:
        """Calculate data quality KPIs"""
        df = self.sales_enriched
        
        # Basic data quality metrics
        total_records = len(df)
        null_counts = df.isnull().sum().sum()
        null_rate = (null_counts / (total_records * len(df.columns)) * 100) if total_records > 0 else 0
        
        # Payment failure rate
        payment_failures = len(df[df['payment_status'] == 'Failed'])
        payment_failure_rate = (payment_failures / total_records * 100) if total_records > 0 else 0
        
        # Return rate
        returns = len(df[df['return_flag'] == True])
        return_rate = (returns / total_records * 100) if total_records > 0 else 0
        
        # Issues breakdown if available
        issues_by_type = {}
        issues_by_dept = {}
        total_issues = 0
        
        if issues_df is not None and len(issues_df) > 0:
            total_issues = len(issues_df)
            issues_by_type = issues_df['issue_type'].value_counts().to_dict()
            issues_by_dept = issues_df['department'].value_counts().to_dict()
        
        return {
            'total_records': total_records,
            'null_rate_pct': round(null_rate, 4),
            'payment_failure_rate': round(payment_failure_rate, 2),
            'return_rate': round(return_rate, 2),
            'total_data_issues': total_issues,
            'issues_by_type': issues_by_type,
            'issues_by_department': issues_by_dept,
            'reference_return_target': REFERENCE_VALUES['target_return_rate'],
            'reference_failure_target': REFERENCE_VALUES['target_payment_failure']
        }
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict = None) -> pd.DataFrame:
        """Apply filters to dataframe"""
        if not filters:
            return df
        
        result = df.copy()
        
        if 'city' in filters and filters['city'] and filters['city'] != 'All':
            if isinstance(filters['city'], list):
                result = result[result['city'].isin(filters['city'])]
            else:
                result = result[result['city'] == filters['city']]
        
        if 'channel' in filters and filters['channel'] and filters['channel'] != 'All':
            if isinstance(filters['channel'], list):
                result = result[result['channel'].isin(filters['channel'])]
            else:
                result = result[result['channel'] == filters['channel']]
        
        if 'category' in filters and filters['category'] and filters['category'] != 'All':
            if isinstance(filters['category'], list):
                result = result[result['category'].isin(filters['category'])]
            else:
                result = result[result['category'] == filters['category']]
        
        if 'brand' in filters and filters['brand']:
            if isinstance(filters['brand'], list):
                result = result[result['brand'].isin(filters['brand'])]
            else:
                result = result[result['brand'] == filters['brand']]
        
        if 'date_start' in filters and filters['date_start']:
            result = result[result['order_time'] >= filters['date_start']]
        
        if 'date_end' in filters and filters['date_end']:
            result = result[result['order_time'] <= filters['date_end']]
        
        return result
    
    def get_all_kpis(self, filters: Dict = None, issues_df: pd.DataFrame = None, 
                     promo_budget: float = None) -> Dict:
        """Get all KPIs organized by category"""
        return {
            'business': self.calculate_business_kpis(filters),
            'inventory': self.calculate_inventory_kpis(filters),
            'channel': self.calculate_channel_kpis(filters),
            'customer': self.calculate_customer_kpis(filters),
            'promotion': self.calculate_promotion_kpis(filters, promo_budget),
            'data_quality': self.calculate_data_quality_kpis(issues_df)
        }


class PromoSimulator:
    """Rule-based promotional campaign simulator"""
    
    def __init__(self, sales_df: pd.DataFrame, products_df: pd.DataFrame,
                 stores_df: pd.DataFrame, inventory_df: pd.DataFrame):
        self.sales = sales_df.copy()
        self.products = products_df.copy()
        self.stores = stores_df.copy()
        self.inventory = inventory_df.copy()
        
        # Calculate baseline metrics
        self._calculate_baseline()
    
    def _calculate_baseline(self):
        """Calculate baseline demand from historical data"""
        # Merge sales with dimensions
        sales_enriched = self.sales.merge(
            self.products[['product_id', 'category', 'base_price_aed', 'unit_cost_aed']],
            on='product_id', how='left'
        ).merge(
            self.stores[['store_id', 'city', 'channel']],
            on='store_id', how='left'
        )
        
        # Only use paid orders
        paid_sales = sales_enriched[sales_enriched['payment_status'] == 'Paid']
        
        # Parse dates
        paid_sales['order_date'] = pd.to_datetime(paid_sales['order_time']).dt.date
        
        # Calculate daily demand per product-store
        daily_demand = paid_sales.groupby(['product_id', 'store_id', 'order_date'])['qty'].sum().reset_index()
        
        # Average daily demand (last 30 days of data)
        self.baseline_demand = daily_demand.groupby(['product_id', 'store_id'])['qty'].mean().reset_index()
        self.baseline_demand.columns = ['product_id', 'store_id', 'baseline_daily_demand']
        
        # Store enriched sales for later
        self.sales_enriched = sales_enriched
    
    def calculate_uplift_factor(self, discount_pct: float, channel: str, 
                               category: str, city: str) -> float:
        """Calculate demand uplift factor based on discount and dimensions"""
        # Base uplift from discount (diminishing returns)
        # 10% discount = ~15% uplift, 20% = ~25%, 30% = ~32%
        base_uplift = 1 + (discount_pct / 100) * 1.2 * (1 - discount_pct / 200)
        
        # Apply channel sensitivity
        channel_factor = CHANNEL_SENSITIVITY.get(channel, 1.0)
        
        # Apply category sensitivity
        category_factor = CATEGORY_SENSITIVITY.get(category, 1.0)
        
        # Apply city factor
        city_factor = CITY_FACTORS.get(city, 1.0)
        
        # Combined uplift (multiplicative but with dampening)
        total_uplift = base_uplift * (channel_factor * category_factor * city_factor) ** 0.5
        
        # Cap at 3x baseline
        return min(total_uplift, 3.0)
    
    def run_simulation(self, discount_pct: float, promo_budget: float,
                      margin_floor_pct: float, simulation_days: int = 14,
                      city: str = 'All', channel: str = 'All',
                      category: str = 'All') -> Dict:
        """Run promotional campaign simulation"""
        
        # Get latest inventory
        latest_inventory = self.inventory.sort_values('snapshot_date').groupby(
            ['product_id', 'store_id']
        ).last().reset_index()[['product_id', 'store_id', 'stock_on_hand']]
        
        # Merge baseline demand with inventory and product info
        sim_data = self.baseline_demand.merge(latest_inventory, on=['product_id', 'store_id'], how='left')
        sim_data = sim_data.merge(
            self.products[['product_id', 'category', 'base_price_aed', 'unit_cost_aed']],
            on='product_id', how='left'
        )
        sim_data = sim_data.merge(
            self.stores[['store_id', 'city', 'channel']],
            on='store_id', how='left'
        )
        
        # Apply filters
        if city != 'All':
            sim_data = sim_data[sim_data['city'] == city]
        if channel != 'All':
            sim_data = sim_data[sim_data['channel'] == channel]
        if category != 'All':
            sim_data = sim_data[sim_data['category'] == category]
        
        if len(sim_data) == 0:
            return self._empty_simulation_result()
        
        # Calculate simulated metrics for each product-store
        results = []
        constraint_violations = []
        
        for idx, row in sim_data.iterrows():
            # Calculate uplift factor
            uplift = self.calculate_uplift_factor(
                discount_pct, row['channel'], row['category'], row['city']
            )
            
            # Simulated daily demand
            baseline_demand = row['baseline_daily_demand'] if pd.notna(row['baseline_daily_demand']) else 0
            sim_daily_demand = baseline_demand * uplift
            sim_total_demand = sim_daily_demand * simulation_days
            
            # Stock constraint
            available_stock = row['stock_on_hand'] if pd.notna(row['stock_on_hand']) else 0
            sim_qty_sold = min(sim_total_demand, available_stock)
            
            # Calculate financials
            selling_price = row['base_price_aed'] * (1 - discount_pct / 100)
            sim_revenue = sim_qty_sold * selling_price
            sim_cogs = sim_qty_sold * row['unit_cost_aed'] if pd.notna(row['unit_cost_aed']) else 0
            sim_margin = sim_revenue - sim_cogs
            sim_margin_pct = (sim_margin / sim_revenue * 100) if sim_revenue > 0 else 0
            
            # Discount cost (promo spend)
            discount_amount = sim_qty_sold * row['base_price_aed'] * (discount_pct / 100)
            
            # Check constraints
            stockout_risk = sim_total_demand > available_stock
            margin_violation = sim_margin_pct < margin_floor_pct
            
            if stockout_risk:
                constraint_violations.append({
                    'product_id': row['product_id'],
                    'store_id': row['store_id'],
                    'city': row['city'],
                    'channel': row['channel'],
                    'category': row['category'],
                    'violation_type': 'STOCKOUT_RISK',
                    'detail': f'Demand {sim_total_demand:.0f} > Stock {available_stock:.0f}'
                })
            
            if margin_violation:
                constraint_violations.append({
                    'product_id': row['product_id'],
                    'store_id': row['store_id'],
                    'city': row['city'],
                    'channel': row['channel'],
                    'category': row['category'],
                    'violation_type': 'MARGIN_FLOOR',
                    'detail': f'Margin {sim_margin_pct:.1f}% < Floor {margin_floor_pct}%'
                })
            
            results.append({
                'product_id': row['product_id'],
                'store_id': row['store_id'],
                'city': row['city'],
                'channel': row['channel'],
                'category': row['category'],
                'baseline_demand': baseline_demand * simulation_days,
                'simulated_demand': sim_total_demand,
                'available_stock': available_stock,
                'simulated_qty_sold': sim_qty_sold,
                'simulated_revenue': sim_revenue,
                'simulated_cogs': sim_cogs,
                'simulated_margin': sim_margin,
                'simulated_margin_pct': sim_margin_pct,
                'discount_amount': discount_amount,
                'uplift_factor': uplift,
                'stockout_risk': stockout_risk
            })
        
        results_df = pd.DataFrame(results)
        violations_df = pd.DataFrame(constraint_violations)
        
        # Aggregate results
        total_revenue = results_df['simulated_revenue'].sum()
        total_cogs = results_df['simulated_cogs'].sum()
        total_margin = results_df['simulated_margin'].sum()
        total_margin_pct = (total_margin / total_revenue * 100) if total_revenue > 0 else 0
        total_promo_spend = results_df['discount_amount'].sum()
        profit_proxy = total_margin - total_promo_spend
        
        # Budget check
        budget_exceeded = total_promo_spend > promo_budget
        budget_utilization = (total_promo_spend / promo_budget * 100) if promo_budget > 0 else 0
        
        if budget_exceeded:
            constraint_violations.append({
                'product_id': 'TOTAL',
                'store_id': 'ALL',
                'city': city,
                'channel': channel,
                'category': category,
                'violation_type': 'BUDGET_EXCEEDED',
                'detail': f'Promo spend AED {total_promo_spend:,.0f} > Budget AED {promo_budget:,.0f}'
            })
        
        # Stockout risk calculation
        stockout_items = results_df[results_df['stockout_risk']].copy()
        stockout_risk_pct = (len(stockout_items) / len(results_df) * 100) if len(results_df) > 0 else 0
        
        # Top risk items
        top_stockout_risk = results_df.nlargest(10, 'simulated_demand')[
            ['product_id', 'store_id', 'city', 'channel', 'category', 
             'simulated_demand', 'available_stock', 'stockout_risk']
        ]
        
        return {
            'summary': {
                'simulation_days': simulation_days,
                'discount_pct': discount_pct,
                'promo_budget': promo_budget,
                'margin_floor_pct': margin_floor_pct,
                'total_simulated_revenue': total_revenue,
                'total_simulated_cogs': total_cogs,
                'total_simulated_margin': total_margin,
                'total_margin_pct': round(total_margin_pct, 2),
                'total_promo_spend': total_promo_spend,
                'profit_proxy': profit_proxy,
                'budget_utilization_pct': round(min(budget_utilization, 100), 2),
                'budget_exceeded': budget_exceeded,
                'stockout_risk_pct': round(stockout_risk_pct, 2),
                'items_with_stockout_risk': len(stockout_items),
                'total_items_simulated': len(results_df),
                'avg_uplift_factor': round(results_df['uplift_factor'].mean(), 2),
                'constraint_violations': len(constraint_violations)
            },
            'detailed_results': results_df,
            'violations': violations_df if len(violations_df) > 0 else pd.DataFrame(),
            'top_stockout_risk': top_stockout_risk,
            'by_city': results_df.groupby('city').agg({
                'simulated_revenue': 'sum',
                'simulated_margin': 'sum',
                'stockout_risk': 'sum'
            }).reset_index(),
            'by_channel': results_df.groupby('channel').agg({
                'simulated_revenue': 'sum',
                'simulated_margin': 'sum',
                'stockout_risk': 'sum'
            }).reset_index(),
            'by_category': results_df.groupby('category').agg({
                'simulated_revenue': 'sum',
                'simulated_margin': 'sum',
                'stockout_risk': 'sum'
            }).reset_index()
        }
    
    def _empty_simulation_result(self) -> Dict:
        """Return empty simulation result when no data matches filters"""
        return {
            'summary': {
                'simulation_days': 0,
                'discount_pct': 0,
                'promo_budget': 0,
                'margin_floor_pct': 0,
                'total_simulated_revenue': 0,
                'total_simulated_cogs': 0,
                'total_simulated_margin': 0,
                'total_margin_pct': 0,
                'total_promo_spend': 0,
                'profit_proxy': 0,
                'budget_utilization_pct': 0,
                'budget_exceeded': False,
                'stockout_risk_pct': 0,
                'items_with_stockout_risk': 0,
                'total_items_simulated': 0,
                'avg_uplift_factor': 0,
                'constraint_violations': 0
            },
            'detailed_results': pd.DataFrame(),
            'violations': pd.DataFrame(),
            'top_stockout_risk': pd.DataFrame(),
            'by_city': pd.DataFrame(),
            'by_channel': pd.DataFrame(),
            'by_category': pd.DataFrame()
        }
    
    def generate_recommendation(self, simulation_result: Dict) -> str:
        """Generate AI-style recommendation based on simulation results"""
        summary = simulation_result['summary']
        
        recommendations = []
        
        # Budget analysis
        if summary['budget_exceeded']:
            recommendations.append(
                f"⚠️ **Budget Alert**: Projected promo spend (AED {summary['total_promo_spend']:,.0f}) "
                f"exceeds budget by {summary['total_promo_spend'] - summary['promo_budget']:,.0f} AED. "
                "Consider reducing discount percentage or narrowing campaign scope."
            )
        elif summary['budget_utilization_pct'] < 50:
            recommendations.append(
                f"💡 **Budget Opportunity**: Only {summary['budget_utilization_pct']:.0f}% of budget utilized. "
                "Consider increasing discount or expanding to more categories/channels."
            )
        
        # Margin analysis
        if summary['total_margin_pct'] < summary['margin_floor_pct']:
            recommendations.append(
                f"⚠️ **Margin Warning**: Projected margin ({summary['total_margin_pct']:.1f}%) "
                f"is below target floor ({summary['margin_floor_pct']}%). "
                "Reduce discount or focus on higher-margin categories."
            )
        elif summary['total_margin_pct'] > 35:
            recommendations.append(
                f"✅ **Healthy Margins**: {summary['total_margin_pct']:.1f}% margin maintained. "
                "Room to increase discount for volume growth."
            )
        
        # Stockout analysis
        if summary['stockout_risk_pct'] > 20:
            recommendations.append(
                f"⚠️ **Stockout Risk**: {summary['stockout_risk_pct']:.0f}% of items at risk of stockout. "
                f"Review inventory for {summary['items_with_stockout_risk']} high-risk SKUs before campaign launch."
            )
        elif summary['stockout_risk_pct'] > 5:
            recommendations.append(
                f"📦 **Inventory Check**: {summary['items_with_stockout_risk']} items may face stockouts. "
                "Prioritize replenishment for top-selling products."
            )
        else:
            recommendations.append(
                "✅ **Inventory Ready**: Low stockout risk across portfolio."
            )
        
        # ROI assessment
        if summary['profit_proxy'] > 0:
            recommendations.append(
                f"📈 **Positive ROI**: Campaign projected to generate AED {summary['profit_proxy']:,.0f} "
                f"in profit after promo costs."
            )
        else:
            recommendations.append(
                f"📉 **ROI Concern**: Campaign may result in AED {abs(summary['profit_proxy']):,.0f} loss. "
                "Review discount strategy."
            )
        
        # Overall recommendation
        if summary['budget_exceeded'] or summary['stockout_risk_pct'] > 30 or summary['profit_proxy'] < 0:
            overall = "🔴 **Overall Assessment**: Campaign parameters need adjustment before launch."
        elif summary['stockout_risk_pct'] > 10 or summary['total_margin_pct'] < summary['margin_floor_pct']:
            overall = "🟡 **Overall Assessment**: Proceed with caution. Address highlighted concerns."
        else:
            overall = "🟢 **Overall Assessment**: Campaign parameters are within acceptable range. Ready to proceed."
        
        recommendations.append(overall)
        
        return "\n\n".join(recommendations)


def format_number_short(num: float) -> str:
    """Format large numbers in short form (e.g., AED 20M, AED 1.5B)"""
    if pd.isna(num) or num == 0:
        return "AED 0"
    
    abs_num = abs(num)
    sign = "-" if num < 0 else ""
    
    if abs_num >= 1_000_000_000:
        return f"{sign}AED {abs_num/1_000_000_000:.1f}B"
    elif abs_num >= 1_000_000:
        return f"{sign}AED {abs_num/1_000_000:.1f}M"
    elif abs_num >= 1_000:
        return f"{sign}AED {abs_num/1_000:.1f}K"
    else:
        return f"{sign}AED {abs_num:.0f}"


def format_percentage(num: float) -> str:
    """Format percentage with appropriate precision"""
    if pd.isna(num):
        return "0%"
    return f"{num:.1f}%"
