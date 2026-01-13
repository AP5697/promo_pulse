"""
Promo Pulse Dashboard - Synthetic Data Generator
Generates realistic UAE retail data with intentional errors for cleaning demonstration
Year: 2024 (any 2025 dates are errors to be corrected)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============== CONFIGURATION ==============
VALID_CITIES = ['Dubai', 'Abu Dhabi', 'Sharjah']
VALID_CHANNELS = ['App', 'Web', 'Marketplace']
VALID_FULFILLMENT = ['Own', '3PL']
VALID_PAYMENT_STATUS = ['Paid', 'Failed', 'Refunded']
VALID_CATEGORIES = ['Electronics', 'Fashion', 'Grocery', 'Home & Garden', 'Beauty', 'Sports', 'Toys', 'Books']
VALID_BRANDS = ['Samsung', 'Apple', 'Nike', 'Adidas', 'Zara', 'H&M', 'IKEA', 'Carrefour', 'Noon', 'Amazon', 'LG', 'Sony', 'Puma', 'Levi\'s', 'L\'Oreal']

# Error injection rates
ERROR_RATES = {
    'city_typos': 0.08,  # 8% city name errors
    'missing_unit_cost': 0.015,  # 1.5% missing unit costs
    'missing_discount': 0.03,  # 3% missing discounts
    'duplicate_orders': 0.008,  # 0.8% duplicates
    'corrupted_timestamps': 0.015,  # 1.5% bad timestamps
    'outlier_qty': 0.004,  # 0.4% quantity outliers
    'outlier_price': 0.003,  # 0.3% price outliers
    'negative_stock': 0.005,  # 0.5% negative inventory
    'extreme_stock': 0.003,  # 0.3% extreme stock values
    'wrong_year': 0.02,  # 2% dates with wrong year (2025 instead of 2024)
    'invalid_category': 0.02,  # 2% invalid categories
    'missing_product_id': 0.01,  # 1% missing product IDs
    'invalid_email': 0.03,  # 3% invalid customer emails
}

# City typo variations
CITY_TYPOS = {
    'Dubai': ['Dubaai', 'Dubay', 'Dubaii', 'DUBAI', 'dubai', 'Dxb', 'Dubayy', 'Dibai'],
    'Abu Dhabi': ['Abu dhabi', 'ABU DHABI', 'Abudhabi', 'Abu-Dhabi', 'AbuDhabi', 'Abu Dabi', 'Abudhaby'],
    'Sharjah': ['SHARJAH', 'sharjah', 'Sharjha', 'Sharja', 'Sharjh', 'Al Sharjah']
}


def generate_products(n=300):
    """Generate products table with intentional errors"""
    products = []
    
    for i in range(n):
        product_id = f"PROD-{str(i+1).zfill(5)}"
        category = random.choice(VALID_CATEGORIES)
        brand = random.choice(VALID_BRANDS)
        
        # Base price varies by category
        price_ranges = {
            'Electronics': (200, 5000),
            'Fashion': (50, 800),
            'Grocery': (5, 200),
            'Home & Garden': (100, 2000),
            'Beauty': (30, 500),
            'Sports': (80, 1500),
            'Toys': (20, 400),
            'Books': (15, 150)
        }
        
        min_price, max_price = price_ranges.get(category, (50, 500))
        base_price = round(random.uniform(min_price, max_price), 2)
        
        # Unit cost is typically 40-70% of base price
        cost_ratio = random.uniform(0.4, 0.7)
        unit_cost = round(base_price * cost_ratio, 2)
        
        # Inject missing unit cost error
        if random.random() < ERROR_RATES['missing_unit_cost']:
            unit_cost = np.nan
        
        tax_rate = 0.05  # 5% VAT in UAE
        launch_flag = random.choices(['New', 'Regular'], weights=[0.15, 0.85])[0]
        
        products.append({
            'product_id': product_id,
            'category': category,
            'brand': brand,
            'base_price_aed': base_price,
            'unit_cost_aed': unit_cost,
            'tax_rate': tax_rate,
            'launch_flag': launch_flag
        })
    
    # Inject invalid category errors
    df = pd.DataFrame(products)
    invalid_indices = df.sample(frac=ERROR_RATES['invalid_category']).index
    invalid_categories = ['Electronicss', 'fashion', 'GROCERY', 'Home&Garden', 'Beautty', 'Sport', 'Toyz']
    df.loc[invalid_indices, 'category'] = [random.choice(invalid_categories) for _ in invalid_indices]
    
    return df


def generate_stores(n_per_combo=1):
    """Generate stores table"""
    stores = []
    store_counter = 1
    
    for city in VALID_CITIES:
        for channel in VALID_CHANNELS:
            for fulfillment in VALID_FULFILLMENT:
                for _ in range(n_per_combo):
                    store_id = f"STORE-{str(store_counter).zfill(3)}"
                    
                    # Apply city typo errors
                    display_city = city
                    if random.random() < ERROR_RATES['city_typos']:
                        display_city = random.choice(CITY_TYPOS[city])
                    
                    stores.append({
                        'store_id': store_id,
                        'city': display_city,
                        'channel': channel,
                        'fulfillment_type': fulfillment
                    })
                    store_counter += 1
    
    return pd.DataFrame(stores)


def generate_customers(n=5000):
    """Generate customers table with intentional errors"""
    customers = []
    
    first_names = ['Ahmed', 'Mohammed', 'Ali', 'Omar', 'Fatima', 'Aisha', 'Sara', 'Layla', 'Hassan', 'Ibrahim',
                   'Khalid', 'Yusuf', 'Maryam', 'Zainab', 'Noura', 'Reem', 'Dana', 'Hana', 'Tariq', 'Saeed']
    last_names = ['Al-Maktoum', 'Al-Nahyan', 'Al-Qasimi', 'Al-Nuaimi', 'Al-Sharqi', 'Al-Mualla', 'Al-Rashid',
                  'Al-Ahmad', 'Al-Hassan', 'Al-Hussein', 'Al-Farsi', 'Al-Balushi', 'Al-Zaabi', 'Al-Mazrouei']
    
    domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'icloud.com']
    invalid_emails = ['invalid@', 'test@test', 'noatsign.com', '', 'spaces in@email.com', '@nodomain.com']
    
    for i in range(n):
        customer_id = f"CUST-{str(i+1).zfill(6)}"
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        
        # Generate email
        if random.random() < ERROR_RATES['invalid_email']:
            email = random.choice(invalid_emails)
        else:
            email = f"{first_name.lower()}.{last_name.lower().replace('-', '')}_{random.randint(1,999)}@{random.choice(domains)}"
        
        city = random.choice(VALID_CITIES)
        # Apply city typo
        if random.random() < ERROR_RATES['city_typos']:
            city = random.choice(CITY_TYPOS.get(city, [city]))
        
        segment = random.choices(['Premium', 'Regular', 'New'], weights=[0.15, 0.65, 0.20])[0]
        join_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 364))
        
        # Inject wrong year error (handle leap year edge case)
        if random.random() < ERROR_RATES['wrong_year']:
            try:
                join_date = join_date.replace(year=2025)
            except ValueError:
                # Handle Feb 29 -> Feb 28 for non-leap year
                join_date = join_date.replace(year=2025, day=28)
        
        customers.append({
            'customer_id': customer_id,
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'city': city,
            'customer_segment': segment,
            'join_date': join_date.strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(customers)


def generate_sales_raw(products_df, stores_df, customers_df, n=35000):
    """Generate raw sales data with intentional errors"""
    sales = []
    
    # Historical period: 120 days in 2024
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    product_ids = products_df['product_id'].tolist()
    store_ids = stores_df['store_id'].tolist()
    customer_ids = customers_df['customer_id'].tolist()
    
    for i in range(n):
        order_id = f"ORD-{str(i+1).zfill(7)}"
        
        # Generate order timestamp
        order_date = start_date + timedelta(days=random.randint(0, 364))
        order_time = datetime.combine(order_date, datetime.min.time()) + timedelta(
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Format timestamp with potential errors
        timestamp_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%d/%m/%Y %H:%M',
            '%Y/%m/%d %H:%M:%S',
        ]
        
        if random.random() < ERROR_RATES['corrupted_timestamps']:
            # Create corrupted timestamp
            corrupted_options = ['not_a_time', 'NULL', '0000-00-00', '2024-13-45 25:99:99', 
                                 'invalid_date', '', 'NaT', '99/99/9999']
            order_time_str = random.choice(corrupted_options)
        elif random.random() < ERROR_RATES['wrong_year']:
            # Wrong year (2025 instead of 2024)
            try:
                wrong_date = order_time.replace(year=2025)
            except ValueError:
                wrong_date = order_time.replace(year=2025, day=28)
            order_time_str = wrong_date.strftime(random.choice(timestamp_formats))
        else:
            order_time_str = order_time.strftime(random.choice(timestamp_formats))
        
        # Select product and get its base price
        product_id = random.choice(product_ids)
        if random.random() < ERROR_RATES['missing_product_id']:
            product_id = np.nan
        
        product_row = products_df[products_df['product_id'] == product_id]
        if not product_row.empty:
            base_price = product_row['base_price_aed'].values[0]
        else:
            base_price = random.uniform(50, 500)
        
        store_id = random.choice(store_ids)
        customer_id = random.choice(customer_ids)
        
        # Quantity with potential outliers
        if random.random() < ERROR_RATES['outlier_qty']:
            qty = random.choice([50, 100, 200, 500])  # Outlier quantities
        else:
            qty = random.choices([1, 2, 3, 4, 5], weights=[0.5, 0.25, 0.15, 0.07, 0.03])[0]
        
        # Discount percentage
        if random.random() < ERROR_RATES['missing_discount']:
            discount_pct = np.nan
        else:
            discount_pct = random.choices([0, 5, 10, 15, 20, 25, 30], 
                                          weights=[0.3, 0.2, 0.2, 0.15, 0.1, 0.03, 0.02])[0]
        
        # Selling price with potential outliers
        if random.random() < ERROR_RATES['outlier_price']:
            selling_price = base_price * random.choice([10, 15, 0.1, 0.05])  # Outlier prices
        else:
            if pd.notna(discount_pct):
                selling_price = round(base_price * (1 - discount_pct / 100), 2)
            else:
                selling_price = base_price
        
        payment_status = random.choices(VALID_PAYMENT_STATUS, weights=[0.85, 0.08, 0.07])[0]
        return_flag = random.choices([True, False], weights=[0.05, 0.95])[0]
        
        sales.append({
            'order_id': order_id,
            'order_time': order_time_str,
            'product_id': product_id,
            'store_id': store_id,
            'customer_id': customer_id,
            'qty': qty,
            'selling_price_aed': round(selling_price, 2),
            'discount_pct': discount_pct,
            'payment_status': payment_status,
            'return_flag': return_flag
        })
    
    df = pd.DataFrame(sales)
    
    # Inject duplicate orders
    n_duplicates = int(len(df) * ERROR_RATES['duplicate_orders'])
    duplicate_rows = df.sample(n=n_duplicates).copy()
    # Slightly modify some fields in duplicates
    duplicate_rows['qty'] = duplicate_rows['qty'].apply(lambda x: x + random.choice([-1, 0, 1]))
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    return df


def generate_inventory_snapshot(products_df, stores_df, days=30):
    """Generate inventory snapshot data with intentional errors"""
    inventory = []
    
    # Last 30 days of 2024
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=days-1)
    
    product_ids = products_df['product_id'].tolist()
    store_ids = stores_df['store_id'].tolist()
    
    # For efficiency, only track top 50 products per store
    sampled_products = random.sample(product_ids, min(100, len(product_ids)))
    
    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        snapshot_date = current_date.strftime('%Y-%m-%d')
        
        # Inject wrong year error
        if random.random() < ERROR_RATES['wrong_year']:
            try:
                snapshot_date = current_date.replace(year=2025).strftime('%Y-%m-%d')
            except ValueError:
                snapshot_date = current_date.replace(year=2025, day=28).strftime('%Y-%m-%d')
        
        for store_id in store_ids:
            for product_id in sampled_products:
                # Base stock level
                base_stock = random.randint(10, 200)
                
                # Inject errors
                if random.random() < ERROR_RATES['negative_stock']:
                    stock_on_hand = random.randint(-50, -1)
                elif random.random() < ERROR_RATES['extreme_stock']:
                    stock_on_hand = random.choice([9999, 99999, 0])
                else:
                    # Simulate stock fluctuation
                    stock_on_hand = max(0, base_stock + random.randint(-20, 20))
                
                reorder_point = random.randint(10, 30)
                lead_time_days = random.choice([3, 5, 7, 10, 14])
                
                inventory.append({
                    'snapshot_date': snapshot_date,
                    'product_id': product_id,
                    'store_id': store_id,
                    'stock_on_hand': stock_on_hand,
                    'reorder_point': reorder_point,
                    'lead_time_days': lead_time_days
                })
    
    return pd.DataFrame(inventory)


def generate_campaign_plan():
    """Generate campaign plan scenarios"""
    campaigns = []
    
    campaign_configs = [
        {'name': 'Ramadan Sale', 'discount': 25, 'budget': 500000, 'city': 'All', 'channel': 'All', 'category': 'All'},
        {'name': 'Summer Clearance', 'discount': 30, 'budget': 300000, 'city': 'Dubai', 'channel': 'App', 'category': 'Fashion'},
        {'name': 'Electronics Week', 'discount': 15, 'budget': 400000, 'city': 'All', 'channel': 'Marketplace', 'category': 'Electronics'},
        {'name': 'Beauty Bonanza', 'discount': 20, 'budget': 150000, 'city': 'Abu Dhabi', 'channel': 'Web', 'category': 'Beauty'},
        {'name': 'Sports Festival', 'discount': 20, 'budget': 200000, 'city': 'Sharjah', 'channel': 'All', 'category': 'Sports'},
        {'name': 'Back to School', 'discount': 15, 'budget': 250000, 'city': 'All', 'channel': 'App', 'category': 'Books'},
        {'name': 'Home Makeover', 'discount': 25, 'budget': 350000, 'city': 'Dubai', 'channel': 'Web', 'category': 'Home & Garden'},
        {'name': 'Flash Sale', 'discount': 35, 'budget': 100000, 'city': 'All', 'channel': 'Marketplace', 'category': 'All'},
        {'name': 'Weekend Special', 'discount': 10, 'budget': 75000, 'city': 'Dubai', 'channel': 'App', 'category': 'Grocery'},
        {'name': 'Year End Blowout', 'discount': 40, 'budget': 600000, 'city': 'All', 'channel': 'All', 'category': 'All'},
    ]
    
    base_date = datetime(2024, 12, 15)
    
    for i, config in enumerate(campaign_configs):
        campaign_id = f"CAMP-{str(i+1).zfill(3)}"
        start_date = base_date + timedelta(days=i*14)
        end_date = start_date + timedelta(days=13)
        
        campaigns.append({
            'campaign_id': campaign_id,
            'campaign_name': config['name'],
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'city': config['city'],
            'channel': config['channel'],
            'category': config['category'],
            'discount_pct': config['discount'],
            'promo_budget_aed': config['budget']
        })
    
    return pd.DataFrame(campaigns)


def generate_departments():
    """Generate department data for error tracking"""
    departments = [
        {'dept_id': 'DEPT-001', 'dept_name': 'Sales Operations', 'manager': 'Ahmed Al-Maktoum', 'region': 'Dubai'},
        {'dept_id': 'DEPT-002', 'dept_name': 'Inventory Management', 'manager': 'Fatima Al-Nahyan', 'region': 'Abu Dhabi'},
        {'dept_id': 'DEPT-003', 'dept_name': 'Customer Service', 'manager': 'Omar Al-Qasimi', 'region': 'Sharjah'},
        {'dept_id': 'DEPT-004', 'dept_name': 'Marketing', 'manager': 'Sara Al-Rashid', 'region': 'Dubai'},
        {'dept_id': 'DEPT-005', 'dept_name': 'Finance', 'manager': 'Hassan Al-Farsi', 'region': 'Dubai'},
        {'dept_id': 'DEPT-006', 'dept_name': 'Logistics', 'manager': 'Khalid Al-Balushi', 'region': 'Abu Dhabi'},
    ]
    return pd.DataFrame(departments)


def generate_all_data(output_dir='data/raw'):
    """Generate all synthetic datasets"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating synthetic data with intentional errors...")
    print("=" * 60)
    
    # Generate data
    print("1. Generating products...")
    products = generate_products(300)
    products.to_csv(f'{output_dir}/products_raw.csv', index=False)
    print(f"   Created {len(products)} products")
    
    print("2. Generating stores...")
    stores = generate_stores(1)
    stores.to_csv(f'{output_dir}/stores_raw.csv', index=False)
    print(f"   Created {len(stores)} stores")
    
    print("3. Generating customers...")
    customers = generate_customers(5000)
    customers.to_csv(f'{output_dir}/customers_raw.csv', index=False)
    print(f"   Created {len(customers)} customers")
    
    print("4. Generating sales transactions...")
    sales = generate_sales_raw(products, stores, customers, 35000)
    sales.to_csv(f'{output_dir}/sales_raw.csv', index=False)
    print(f"   Created {len(sales)} sales records")
    
    print("5. Generating inventory snapshots...")
    inventory = generate_inventory_snapshot(products, stores, 30)
    inventory.to_csv(f'{output_dir}/inventory_raw.csv', index=False)
    print(f"   Created {len(inventory)} inventory records")
    
    print("6. Generating campaign plans...")
    campaigns = generate_campaign_plan()
    campaigns.to_csv(f'{output_dir}/campaigns_raw.csv', index=False)
    print(f"   Created {len(campaigns)} campaign plans")
    
    print("7. Generating departments...")
    departments = generate_departments()
    departments.to_csv(f'{output_dir}/departments.csv', index=False)
    print(f"   Created {len(departments)} departments")
    
    print("=" * 60)
    print("Data generation complete!")
    print(f"Files saved to: {output_dir}/")
    
    # Summary of injected errors
    print("\nInjected Error Summary:")
    print("-" * 40)
    for error_type, rate in ERROR_RATES.items():
        print(f"  {error_type}: ~{rate*100:.1f}%")
    
    return {
        'products': products,
        'stores': stores,
        'customers': customers,
        'sales': sales,
        'inventory': inventory,
        'campaigns': campaigns,
        'departments': departments
    }


if __name__ == '__main__':
    generate_all_data()
