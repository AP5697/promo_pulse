"""
Promo Pulse Dashboard - Data Cleaner
Validates, cleans, and logs all data quality issues
All dates are standardized to 2024, city names corrected, and outliers handled
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============== VALIDATION CONFIGURATION ==============
VALID_CITIES = ['Dubai', 'Abu Dhabi', 'Sharjah']
VALID_CHANNELS = ['App', 'Web', 'Marketplace']
VALID_FULFILLMENT = ['Own', '3PL']
VALID_PAYMENT_STATUS = ['Paid', 'Failed', 'Refunded']
VALID_CATEGORIES = ['Electronics', 'Fashion', 'Grocery', 'Home & Garden', 'Beauty', 'Sports', 'Toys', 'Books']
VALID_SEGMENTS = ['Premium', 'Regular', 'New']
TARGET_YEAR = 2024

# City name correction mappings
CITY_CORRECTIONS = {
    'dubaai': 'Dubai', 'dubay': 'Dubai', 'dubaii': 'Dubai', 'dubai': 'Dubai', 'DUBAI': 'Dubai',
    'dxb': 'Dubai', 'dubayy': 'Dubai', 'dibai': 'Dubai',
    'abu dhabi': 'Abu Dhabi', 'ABU DHABI': 'Abu Dhabi', 'abudhabi': 'Abu Dhabi', 'abu-dhabi': 'Abu Dhabi',
    'AbuDhabi': 'Abu Dhabi', 'abu dabi': 'Abu Dhabi', 'abudhaby': 'Abu Dhabi',
    'SHARJAH': 'Sharjah', 'sharjah': 'Sharjah', 'sharjha': 'Sharjah', 'sharja': 'Sharjah',
    'sharjh': 'Sharjah', 'al sharjah': 'Sharjah', 'Al Sharjah': 'Sharjah'
}

# Category correction mappings
CATEGORY_CORRECTIONS = {
    'electronicss': 'Electronics', 'electronic': 'Electronics', 'ELECTRONICS': 'Electronics',
    'fashion': 'Fashion', 'FASHION': 'Fashion', 'fasion': 'Fashion',
    'grocery': 'Grocery', 'GROCERY': 'Grocery', 'groceries': 'Grocery',
    'home&garden': 'Home & Garden', 'home & garden': 'Home & Garden', 'HOME & GARDEN': 'Home & Garden',
    'beautty': 'Beauty', 'beauty': 'Beauty', 'BEAUTY': 'Beauty',
    'sport': 'Sports', 'sports': 'Sports', 'SPORTS': 'Sports',
    'toyz': 'Toys', 'toy': 'Toys', 'toys': 'Toys', 'TOYS': 'Toys',
    'book': 'Books', 'books': 'Books', 'BOOKS': 'Books'
}

# Outlier thresholds
OUTLIER_CONFIG = {
    'qty_max': 20,  # Maximum reasonable quantity per order
    'qty_min': 1,
    'price_multiplier_max': 5,  # Max 5x base price considered normal
    'price_multiplier_min': 0.2,  # Min 20% of base price
    'stock_max': 5000,  # Maximum reasonable stock
    'stock_min': 0,
}


class DataCleaner:
    """Main data cleaning and validation class"""
    
    def __init__(self, output_dir='data/clean'):
        self.output_dir = output_dir
        self.issues_log = []
        self.cleaning_stats = {}
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/logs', exist_ok=True)
    
    def log_issue(self, table: str, record_id: str, issue_type: str, 
                  issue_detail: str, action_taken: str, original_value: str = '', 
                  corrected_value: str = '', department: str = ''):
        """Log a data quality issue"""
        self.issues_log.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'table': table,
            'record_id': str(record_id),
            'issue_type': issue_type,
            'issue_detail': issue_detail,
            'original_value': str(original_value),
            'corrected_value': str(corrected_value),
            'action_taken': action_taken,
            'department': department
        })
    
    def correct_city_name(self, city: str, record_id: str, table: str) -> str:
        """Correct city name variations"""
        if pd.isna(city):
            self.log_issue(table, record_id, 'MISSING_CITY', 
                          'City value is missing', 'Imputed', '', 'Dubai', 'Sales Operations')
            return 'Dubai'  # Default to Dubai
        
        city_str = str(city).strip()
        city_lower = city_str.lower()
        
        # Check if already valid
        if city_str in VALID_CITIES:
            return city_str
        
        # Try correction mapping
        if city_lower in CITY_CORRECTIONS:
            corrected = CITY_CORRECTIONS[city_lower]
            self.log_issue(table, record_id, 'INVALID_CITY_NAME', 
                          f'City name "{city_str}" is misspelled or has wrong case',
                          'Corrected', city_str, corrected, 'Sales Operations')
            return corrected
        
        # Try to match partial
        for valid_city in VALID_CITIES:
            if valid_city.lower() in city_lower or city_lower in valid_city.lower():
                self.log_issue(table, record_id, 'INVALID_CITY_NAME',
                              f'City name "{city_str}" partially matches {valid_city}',
                              'Corrected', city_str, valid_city, 'Sales Operations')
                return valid_city
        
        # Default to Dubai if unrecognizable
        self.log_issue(table, record_id, 'UNRECOGNIZED_CITY',
                      f'City "{city_str}" not recognized', 
                      'Defaulted', city_str, 'Dubai', 'Sales Operations')
        return 'Dubai'
    
    def correct_category(self, category: str, record_id: str, table: str) -> str:
        """Correct category variations"""
        if pd.isna(category):
            self.log_issue(table, record_id, 'MISSING_CATEGORY',
                          'Category value is missing', 'Imputed', '', 'Electronics', 'Inventory Management')
            return 'Electronics'
        
        cat_str = str(category).strip()
        cat_lower = cat_str.lower()
        
        if cat_str in VALID_CATEGORIES:
            return cat_str
        
        # Try correction mapping
        if cat_lower in CATEGORY_CORRECTIONS:
            corrected = CATEGORY_CORRECTIONS[cat_lower]
            self.log_issue(table, record_id, 'INVALID_CATEGORY',
                          f'Category "{cat_str}" has typo or wrong case',
                          'Corrected', cat_str, corrected, 'Inventory Management')
            return corrected
        
        # Fuzzy match
        for valid_cat in VALID_CATEGORIES:
            if valid_cat.lower().startswith(cat_lower[:3]) or cat_lower.startswith(valid_cat.lower()[:3]):
                self.log_issue(table, record_id, 'INVALID_CATEGORY',
                              f'Category "{cat_str}" fuzzy matched to {valid_cat}',
                              'Corrected', cat_str, valid_cat, 'Inventory Management')
                return valid_cat
        
        self.log_issue(table, record_id, 'UNRECOGNIZED_CATEGORY',
                      f'Category "{cat_str}" not recognized',
                      'Defaulted', cat_str, 'Electronics', 'Inventory Management')
        return 'Electronics'
    
    def correct_year(self, date_str: str, record_id: str, table: str) -> Optional[str]:
        """Correct dates with wrong year (2025 -> 2024)"""
        if pd.isna(date_str) or date_str in ['', 'NULL', 'NaT', 'not_a_time', 'invalid_date']:
            self.log_issue(table, record_id, 'INVALID_TIMESTAMP',
                          f'Timestamp "{date_str}" is invalid or missing',
                          'Dropped', str(date_str), '', 'Sales Operations')
            return None
        
        date_str = str(date_str).strip()
        
        # Try multiple date formats
        date_formats = [
            '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M', '%d/%m/%Y',
            '%Y/%m/%d %H:%M:%S', '%Y/%m/%d', '%m/%d/%Y', '%d-%m-%Y'
        ]
        
        parsed_date = None
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        
        if parsed_date is None:
            self.log_issue(table, record_id, 'CORRUPTED_TIMESTAMP',
                          f'Timestamp "{date_str}" could not be parsed',
                          'Dropped', date_str, '', 'Sales Operations')
            return None
        
        # Check and correct year
        if parsed_date.year == 2025:
            corrected_date = parsed_date.replace(year=2024)
            corrected_str = corrected_date.strftime('%Y-%m-%d %H:%M:%S')
            self.log_issue(table, record_id, 'WRONG_YEAR',
                          f'Date year is 2025, should be 2024 (data timeframe)',
                          'Corrected', date_str, corrected_str, 'Sales Operations')
            return corrected_str
        elif parsed_date.year < 2023 or parsed_date.year > 2024:
            self.log_issue(table, record_id, 'INVALID_YEAR',
                          f'Date year {parsed_date.year} outside valid range (2023-2024)',
                          'Dropped', date_str, '', 'Sales Operations')
            return None
        
        return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
    
    def validate_email(self, email: str, record_id: str, table: str) -> str:
        """Validate and clean email addresses"""
        if pd.isna(email) or email == '':
            self.log_issue(table, record_id, 'MISSING_EMAIL',
                          'Email is missing', 'Flagged', '', '', 'Customer Service')
            return ''
        
        email_str = str(email).strip().lower()
        
        # Basic email validation pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email_str):
            self.log_issue(table, record_id, 'INVALID_EMAIL',
                          f'Email "{email}" is not valid format',
                          'Flagged', email, '', 'Customer Service')
            return email_str  # Keep original but flag
        
        return email_str
    
    def handle_outlier(self, value: float, record_id: str, table: str, 
                       field: str, min_val: float, max_val: float, 
                       cap_strategy: str = 'cap') -> float:
        """Handle outlier values by capping or flagging"""
        if pd.isna(value):
            return value
        
        if value < min_val:
            self.log_issue(table, record_id, 'OUTLIER_LOW',
                          f'{field}={value} below minimum {min_val}',
                          'Capped', str(value), str(min_val), 'Finance')
            return min_val if cap_strategy == 'cap' else value
        
        if value > max_val:
            self.log_issue(table, record_id, 'OUTLIER_HIGH',
                          f'{field}={value} above maximum {max_val}',
                          'Capped', str(value), str(max_val), 'Finance')
            return max_val if cap_strategy == 'cap' else value
        
        return value
    
    def clean_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean products table"""
        print("Cleaning products...")
        df = df.copy()
        original_count = len(df)
        
        # Correct categories
        df['category'] = df.apply(
            lambda row: self.correct_category(row['category'], row['product_id'], 'products'),
            axis=1
        )
        
        # Handle missing unit costs
        for idx, row in df[df['unit_cost_aed'].isna()].iterrows():
            # Impute with category median or 50% of base price
            category = row['category']
            category_median = df[df['category'] == category]['unit_cost_aed'].median()
            
            if pd.notna(category_median):
                imputed_value = round(category_median, 2)
            else:
                imputed_value = round(row['base_price_aed'] * 0.5, 2)
            
            df.at[idx, 'unit_cost_aed'] = imputed_value
            self.log_issue('products', row['product_id'], 'MISSING_UNIT_COST',
                          'Unit cost is missing', 'Imputed',
                          'NaN', str(imputed_value), 'Finance')
        
        # Validate unit_cost <= base_price
        invalid_cost = df[df['unit_cost_aed'] > df['base_price_aed']]
        for idx, row in invalid_cost.iterrows():
            corrected = round(row['base_price_aed'] * 0.6, 2)
            self.log_issue('products', row['product_id'], 'INVALID_COST_PRICE_RATIO',
                          f'Unit cost ({row["unit_cost_aed"]}) > base price ({row["base_price_aed"]})',
                          'Corrected', str(row['unit_cost_aed']), str(corrected), 'Finance')
            df.at[idx, 'unit_cost_aed'] = corrected
        
        # Validate launch_flag
        df['launch_flag'] = df['launch_flag'].apply(
            lambda x: x if x in ['New', 'Regular'] else 'Regular'
        )
        
        self.cleaning_stats['products'] = {
            'original': original_count,
            'cleaned': len(df),
            'removed': 0
        }
        
        return df
    
    def clean_stores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean stores table"""
        print("Cleaning stores...")
        df = df.copy()
        original_count = len(df)
        
        # Correct city names
        df['city'] = df.apply(
            lambda row: self.correct_city_name(row['city'], row['store_id'], 'stores'),
            axis=1
        )
        
        # Validate channels
        invalid_channels = ~df['channel'].isin(VALID_CHANNELS)
        for idx in df[invalid_channels].index:
            self.log_issue('stores', df.at[idx, 'store_id'], 'INVALID_CHANNEL',
                          f'Channel "{df.at[idx, "channel"]}" not recognized',
                          'Corrected', df.at[idx, 'channel'], 'App', 'Sales Operations')
            df.at[idx, 'channel'] = 'App'
        
        # Validate fulfillment
        invalid_fulfill = ~df['fulfillment_type'].isin(VALID_FULFILLMENT)
        for idx in df[invalid_fulfill].index:
            self.log_issue('stores', df.at[idx, 'store_id'], 'INVALID_FULFILLMENT',
                          f'Fulfillment "{df.at[idx, "fulfillment_type"]}" not recognized',
                          'Corrected', df.at[idx, 'fulfillment_type'], 'Own', 'Logistics')
            df.at[idx, 'fulfillment_type'] = 'Own'
        
        self.cleaning_stats['stores'] = {
            'original': original_count,
            'cleaned': len(df),
            'removed': 0
        }
        
        return df
    
    def clean_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean customers table"""
        print("Cleaning customers...")
        df = df.copy()
        original_count = len(df)
        
        # Correct city names
        df['city'] = df.apply(
            lambda row: self.correct_city_name(row['city'], row['customer_id'], 'customers'),
            axis=1
        )
        
        # Validate emails
        df['email'] = df.apply(
            lambda row: self.validate_email(row['email'], row['customer_id'], 'customers'),
            axis=1
        )
        
        # Correct join dates
        df['join_date'] = df.apply(
            lambda row: self.correct_year(row['join_date'], row['customer_id'], 'customers'),
            axis=1
        )
        
        # Validate customer segment
        invalid_segments = ~df['customer_segment'].isin(VALID_SEGMENTS)
        for idx in df[invalid_segments].index:
            self.log_issue('customers', df.at[idx, 'customer_id'], 'INVALID_SEGMENT',
                          f'Segment "{df.at[idx, "customer_segment"]}" not recognized',
                          'Corrected', str(df.at[idx, 'customer_segment']), 'Regular', 'Customer Service')
            df.at[idx, 'customer_segment'] = 'Regular'
        
        # Remove records with invalid dates
        before_drop = len(df)
        df = df.dropna(subset=['join_date'])
        dropped = before_drop - len(df)
        
        self.cleaning_stats['customers'] = {
            'original': original_count,
            'cleaned': len(df),
            'removed': dropped
        }
        
        return df
    
    def clean_sales(self, df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
        """Clean sales table - most complex cleaning"""
        print("Cleaning sales...")
        df = df.copy()
        original_count = len(df)
        
        # 1. Handle corrupted timestamps and wrong years
        df['order_time_clean'] = df.apply(
            lambda row: self.correct_year(row['order_time'], row['order_id'], 'sales'),
            axis=1
        )
        
        # Remove records with unparseable timestamps
        before_drop = len(df)
        df = df.dropna(subset=['order_time_clean'])
        timestamp_dropped = before_drop - len(df)
        
        # 2. Handle duplicate order_ids
        duplicates = df[df.duplicated(subset=['order_id'], keep=False)]
        duplicate_ids = duplicates['order_id'].unique()
        
        for oid in duplicate_ids:
            dup_rows = df[df['order_id'] == oid]
            if len(dup_rows) > 1:
                # Keep the first occurrence
                keep_idx = dup_rows.index[0]
                drop_indices = dup_rows.index[1:]
                
                for idx in drop_indices:
                    self.log_issue('sales', oid, 'DUPLICATE_ORDER',
                                  f'Duplicate order_id found ({len(dup_rows)} occurrences)',
                                  'Dropped', f'Row {idx}', f'Kept row {keep_idx}', 'Sales Operations')
                
                df = df.drop(drop_indices)
        
        duplicates_dropped = len(duplicate_ids)
        
        # 3. Handle missing product_id
        missing_product = df['product_id'].isna()
        for idx in df[missing_product].index:
            self.log_issue('sales', df.at[idx, 'order_id'], 'MISSING_PRODUCT_ID',
                          'Product ID is missing', 'Dropped', 'NaN', '', 'Inventory Management')
        df = df.dropna(subset=['product_id'])
        
        # 4. Handle missing discount_pct
        missing_discount = df['discount_pct'].isna()
        channel_discount_median = df.groupby('store_id')['discount_pct'].transform('median')
        
        for idx in df[missing_discount].index:
            # Get store's median discount or default to 0
            store_median = channel_discount_median.get(idx, 0)
            imputed = store_median if pd.notna(store_median) else 0
            self.log_issue('sales', df.at[idx, 'order_id'], 'MISSING_DISCOUNT',
                          'Discount percentage is missing', 'Imputed',
                          'NaN', str(imputed), 'Marketing')
            df.at[idx, 'discount_pct'] = imputed
        
        df['discount_pct'] = df['discount_pct'].fillna(0)
        
        # 5. Handle quantity outliers
        df['qty'] = df.apply(
            lambda row: self.handle_outlier(
                row['qty'], row['order_id'], 'sales', 'qty',
                OUTLIER_CONFIG['qty_min'], OUTLIER_CONFIG['qty_max']
            ),
            axis=1
        )
        
        # 6. Handle price outliers
        # Merge with products to get base prices
        df = df.merge(
            products_df[['product_id', 'base_price_aed']].rename(columns={'base_price_aed': 'expected_base'}),
            on='product_id',
            how='left'
        )
        
        for idx, row in df.iterrows():
            if pd.notna(row['expected_base']):
                min_price = row['expected_base'] * OUTLIER_CONFIG['price_multiplier_min']
                max_price = row['expected_base'] * OUTLIER_CONFIG['price_multiplier_max']
                
                if row['selling_price_aed'] < min_price or row['selling_price_aed'] > max_price:
                    expected_price = row['expected_base'] * (1 - row['discount_pct'] / 100)
                    self.log_issue('sales', row['order_id'], 'OUTLIER_PRICE',
                                  f'Price {row["selling_price_aed"]} outside expected range [{min_price:.2f}, {max_price:.2f}]',
                                  'Corrected', str(row['selling_price_aed']), str(round(expected_price, 2)), 'Finance')
                    df.at[idx, 'selling_price_aed'] = round(expected_price, 2)
        
        df = df.drop(columns=['expected_base'])
        
        # 7. Validate payment status
        invalid_status = ~df['payment_status'].isin(VALID_PAYMENT_STATUS)
        for idx in df[invalid_status].index:
            self.log_issue('sales', df.at[idx, 'order_id'], 'INVALID_PAYMENT_STATUS',
                          f'Status "{df.at[idx, "payment_status"]}" not recognized',
                          'Corrected', df.at[idx, 'payment_status'], 'Paid', 'Finance')
            df.at[idx, 'payment_status'] = 'Paid'
        
        # Rename clean column
        df['order_time'] = df['order_time_clean']
        df = df.drop(columns=['order_time_clean'])
        
        self.cleaning_stats['sales'] = {
            'original': original_count,
            'cleaned': len(df),
            'removed': original_count - len(df),
            'timestamps_fixed': timestamp_dropped,
            'duplicates_removed': duplicates_dropped
        }
        
        return df
    
    def clean_inventory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean inventory snapshot table"""
        print("Cleaning inventory...")
        df = df.copy()
        original_count = len(df)
        
        # Correct snapshot dates
        df['snapshot_date'] = df.apply(
            lambda row: self.correct_year(
                row['snapshot_date'], 
                f"{row['product_id']}_{row['store_id']}_{row['snapshot_date']}", 
                'inventory'
            ),
            axis=1
        )
        
        # Remove invalid dates
        df = df.dropna(subset=['snapshot_date'])
        
        # Handle negative and extreme stock values
        for idx, row in df.iterrows():
            record_id = f"{row['product_id']}_{row['store_id']}"
            
            if row['stock_on_hand'] < 0:
                self.log_issue('inventory', record_id, 'NEGATIVE_STOCK',
                              f'Stock on hand is negative ({row["stock_on_hand"]})',
                              'Corrected', str(row['stock_on_hand']), '0', 'Inventory Management')
                df.at[idx, 'stock_on_hand'] = 0
            
            elif row['stock_on_hand'] > OUTLIER_CONFIG['stock_max']:
                capped_value = OUTLIER_CONFIG['stock_max']
                self.log_issue('inventory', record_id, 'EXTREME_STOCK',
                              f'Stock ({row["stock_on_hand"]}) exceeds maximum threshold',
                              'Capped', str(row['stock_on_hand']), str(capped_value), 'Inventory Management')
                df.at[idx, 'stock_on_hand'] = capped_value
        
        # Validate lead_time_days
        df['lead_time_days'] = df['lead_time_days'].apply(
            lambda x: max(1, min(30, x)) if pd.notna(x) else 7
        )
        
        self.cleaning_stats['inventory'] = {
            'original': original_count,
            'cleaned': len(df),
            'removed': original_count - len(df)
        }
        
        return df
    
    def clean_campaigns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean campaign plan table"""
        print("Cleaning campaigns...")
        df = df.copy()
        
        # Correct dates
        df['start_date'] = df.apply(
            lambda row: self.correct_year(row['start_date'], row['campaign_id'], 'campaigns'),
            axis=1
        )
        df['end_date'] = df.apply(
            lambda row: self.correct_year(row['end_date'], row['campaign_id'], 'campaigns'),
            axis=1
        )
        
        # Validate discount range
        df['discount_pct'] = df['discount_pct'].clip(0, 50)
        
        # Validate budget
        df['promo_budget_aed'] = df['promo_budget_aed'].clip(1000, 10000000)
        
        self.cleaning_stats['campaigns'] = {
            'original': len(df),
            'cleaned': len(df),
            'removed': 0
        }
        
        return df
    
    def generate_issues_report(self) -> pd.DataFrame:
        """Generate comprehensive issues report"""
        issues_df = pd.DataFrame(self.issues_log)
        
        if len(issues_df) > 0:
            # Sort by timestamp and issue type
            issues_df = issues_df.sort_values(['timestamp', 'issue_type'])
            
        return issues_df
    
    def generate_summary_report(self) -> Dict:
        """Generate cleaning summary statistics"""
        issues_df = self.generate_issues_report()
        
        summary = {
            'total_issues': len(issues_df),
            'issues_by_type': issues_df['issue_type'].value_counts().to_dict() if len(issues_df) > 0 else {},
            'issues_by_table': issues_df['table'].value_counts().to_dict() if len(issues_df) > 0 else {},
            'issues_by_action': issues_df['action_taken'].value_counts().to_dict() if len(issues_df) > 0 else {},
            'issues_by_department': issues_df['department'].value_counts().to_dict() if len(issues_df) > 0 else {},
            'cleaning_stats': self.cleaning_stats
        }
        
        # Calculate outlier percentage
        if len(issues_df) > 0:
            outlier_issues = issues_df[issues_df['issue_type'].str.contains('OUTLIER', na=False)]
            total_records = sum([s.get('original', 0) for s in self.cleaning_stats.values()])
            summary['outlier_percentage'] = round(len(outlier_issues) / max(total_records, 1) * 100, 2)
        else:
            summary['outlier_percentage'] = 0
        
        return summary
    
    def clean_all_data(self, raw_dir='data/raw') -> Dict[str, pd.DataFrame]:
        """Clean all datasets and return cleaned versions"""
        print("=" * 60)
        print("PROMO PULSE - DATA CLEANING PROCESS")
        print("=" * 60)
        
        # Load raw data
        print("\nLoading raw data...")
        products_raw = pd.read_csv(f'{raw_dir}/products_raw.csv')
        stores_raw = pd.read_csv(f'{raw_dir}/stores_raw.csv')
        customers_raw = pd.read_csv(f'{raw_dir}/customers_raw.csv')
        sales_raw = pd.read_csv(f'{raw_dir}/sales_raw.csv')
        inventory_raw = pd.read_csv(f'{raw_dir}/inventory_raw.csv')
        campaigns_raw = pd.read_csv(f'{raw_dir}/campaigns_raw.csv')
        
        print("\nCleaning data...")
        print("-" * 40)
        
        # Clean in order of dependencies
        products_clean = self.clean_products(products_raw)
        stores_clean = self.clean_stores(stores_raw)
        customers_clean = self.clean_customers(customers_raw)
        sales_clean = self.clean_sales(sales_raw, products_clean)
        inventory_clean = self.clean_inventory(inventory_raw)
        campaigns_clean = self.clean_campaigns(campaigns_raw)
        
        # Save cleaned data
        print("\nSaving cleaned data...")
        products_clean.to_csv(f'{self.output_dir}/products_clean.csv', index=False)
        stores_clean.to_csv(f'{self.output_dir}/stores_clean.csv', index=False)
        customers_clean.to_csv(f'{self.output_dir}/customers_clean.csv', index=False)
        sales_clean.to_csv(f'{self.output_dir}/sales_clean.csv', index=False)
        inventory_clean.to_csv(f'{self.output_dir}/inventory_clean.csv', index=False)
        campaigns_clean.to_csv(f'{self.output_dir}/campaigns_clean.csv', index=False)
        
        # Generate and save issues log
        issues_df = self.generate_issues_report()
        issues_df.to_csv(f'{self.output_dir}/logs/issues_log.csv', index=False)
        
        # Generate summary
        summary = self.generate_summary_report()
        
        print("\n" + "=" * 60)
        print("CLEANING SUMMARY")
        print("=" * 60)
        print(f"Total issues found and logged: {summary['total_issues']}")
        print(f"Outlier percentage: {summary['outlier_percentage']:.2f}%")
        print("\nIssues by type:")
        for issue_type, count in sorted(summary['issues_by_type'].items(), key=lambda x: -x[1])[:10]:
            print(f"  {issue_type}: {count}")
        print("\nIssues by department:")
        for dept, count in sorted(summary['issues_by_department'].items(), key=lambda x: -x[1]):
            print(f"  {dept}: {count}")
        
        return {
            'products': products_clean,
            'stores': stores_clean,
            'customers': customers_clean,
            'sales': sales_clean,
            'inventory': inventory_clean,
            'campaigns': campaigns_clean,
            'issues': issues_df,
            'summary': summary,
            'raw': {
                'products': products_raw,
                'stores': stores_raw,
                'customers': customers_raw,
                'sales': sales_raw,
                'inventory': inventory_raw,
                'campaigns': campaigns_raw
            }
        }


def clean_uploaded_data(uploaded_files: Dict[str, pd.DataFrame], column_mapping: Dict[str, Dict[str, str]]) -> Dict:
    """Clean user-uploaded data with custom column mapping"""
    cleaner = DataCleaner(output_dir='data/uploaded_clean')
    
    # Apply column mappings and clean
    cleaned_data = {}
    
    for table_name, df in uploaded_files.items():
        if table_name in column_mapping:
            # Rename columns according to mapping
            mapping = column_mapping[table_name]
            df = df.rename(columns=mapping)
        
        # Apply appropriate cleaning based on table type
        if 'product' in table_name.lower():
            cleaned_data[table_name] = cleaner.clean_products(df)
        elif 'store' in table_name.lower():
            cleaned_data[table_name] = cleaner.clean_stores(df)
        elif 'customer' in table_name.lower():
            cleaned_data[table_name] = cleaner.clean_customers(df)
        elif 'sale' in table_name.lower():
            # Need products for sales cleaning
            products_df = cleaned_data.get('products', uploaded_files.get('products', pd.DataFrame()))
            cleaned_data[table_name] = cleaner.clean_sales(df, products_df)
        elif 'inventory' in table_name.lower():
            cleaned_data[table_name] = cleaner.clean_inventory(df)
        elif 'campaign' in table_name.lower():
            cleaned_data[table_name] = cleaner.clean_campaigns(df)
        else:
            cleaned_data[table_name] = df
    
    cleaned_data['issues'] = cleaner.generate_issues_report()
    cleaned_data['summary'] = cleaner.generate_summary_report()
    
    return cleaned_data


if __name__ == '__main__':
    cleaner = DataCleaner()
    result = cleaner.clean_all_data()
