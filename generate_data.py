"""
generate_data.py

This script generates a synthetic retail transaction dataset inspired by the
popular Superstore dataset. The generated data includes order details such
as order dates, shipping dates, customer information, geographical regions,
product categories, sub‑categories, sales figures, discounts and profit.
The purpose of this script is to create a realistic dataset for business
analysis without relying on proprietary or sensitive information. The
parameters controlling the data generation can be configured through the
accompanying `config.yaml` file.

Usage:
    python generate_data.py [--config CONFIG_PATH] [--output OUTPUT_DIR]

The default configuration will be loaded from `config.yaml` located in
the same directory as this script. The generated CSV and Excel files
will be written to the specified output directory (default: current
working directory).

Dependencies:
    - pandas
    - numpy
    - PyYAML
    - faker
    These are listed in requirements.txt for convenience.

"""

import argparse
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yaml
# We avoid using the faker package to reduce external dependencies. Instead,
# define simple lists of first and last names to construct customer names.
FIRST_NAMES = [
    'James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda',
    'William', 'Elizabeth', 'David', 'Barbara', 'Richard', 'Susan', 'Joseph',
    'Jessica', 'Thomas', 'Sarah', 'Charles', 'Karen', 'Christopher', 'Nancy',
    'Daniel', 'Lisa', 'Matthew', 'Betty', 'Anthony', 'Sandra', 'Donald', 'Ashley'
]
LAST_NAMES = [
    'Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson',
    'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin',
    'Thompson', 'Garcia', 'Martinez', 'Robinson', 'Clark', 'Rodriguez', 'Lewis',
    'Lee', 'Walker', 'Hall', 'Allen', 'Young', 'King', 'Wright'
]


def load_config(config_path: str) -> dict:
    """Load YAML configuration from the given path."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_dataset(config: dict) -> pd.DataFrame:
    """Generate a synthetic dataset based on the provided configuration.

    Args:
        config: Dictionary containing configuration parameters such as
            number of rows, date range, categories, segments, regions and
            ship modes.

    Returns:
        Pandas DataFrame containing the synthetic dataset.
    """
    num_rows = config.get('num_rows', 1000)
    start_date = datetime.strptime(config.get('start_date', '2021-01-01'), '%Y-%m-%d')
    end_date = datetime.strptime(config.get('end_date', '2025-12-31'), '%Y-%m-%d')
    categories = config.get('categories', {})
    segments = config.get('segments', ['Consumer', 'Corporate', 'Home Office'])
    regions = config.get('regions', ['East', 'West', 'Central', 'South'])
    ship_modes = config.get('ship_modes', ['Standard Class', 'Second Class', 'First Class', 'Same Day'])

    # Flatten categories into a list of tuples (category, sub_category)
    cat_sub_list = []
    for cat, details in categories.items():
        for sub in details.get('subcategories', []):
            cat_sub_list.append((cat, sub))

    # Initialize random seeds for reproducibility
    np.random.seed(42)

    records = []
    for i in range(num_rows):
        order_date = start_date + timedelta(days=int(np.random.randint(0, (end_date - start_date).days + 1)))
        # Ship date is 1–10 days after order date
        ship_date = order_date + timedelta(days=int(np.random.randint(1, 11)))

        # Unique identifiers
        order_id = f"ORD-{100000 + i}"
        product_id = f"PROD-{np.random.randint(1000, 9999)}"
        customer_id = f"CUST-{np.random.randint(1000, 9999)}"
        # Customer name via faker
        # Construct a random customer name by combining a first and last name
        customer_name = f"{np.random.choice(FIRST_NAMES)} {np.random.choice(LAST_NAMES)}"

        region = np.random.choice(regions)
        segment = np.random.choice(segments)
        ship_mode = np.random.choice(ship_modes)

        category, sub_category = cat_sub_list[np.random.randint(0, len(cat_sub_list))]

        # Generate sales amount using a lognormal distribution to simulate skewed sales
        sales = float(np.random.lognormal(mean=3.0, sigma=0.5))  # typical values around exp(mean)
        # Round sales to 2 decimal places
        sales = round(sales, 2)

        quantity = int(np.random.randint(1, 11))
        # Discount rarely above 0.3; heavy discount events are rare
        discount = float(np.round(np.random.choice([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.05, 0.03, 0.015, 0.005]), 2))

        # Profit margin baseline
        profit_margin = float(np.round(np.random.uniform(0.05, 0.3), 2))
        # Compute profit: profit = sales * (profit_margin - discount)
        profit = round(sales * (profit_margin - discount), 2)

        # Shipping cost roughly 5–15% of sales
        shipping_cost = round(sales * np.random.uniform(0.05, 0.15), 2)

        records.append({
            'OrderID': order_id,
            'OrderDate': order_date,
            'ShipDate': ship_date,
            'ShipMode': ship_mode,
            'CustomerID': customer_id,
            'CustomerName': customer_name,
            'Segment': segment,
            'Region': region,
            'Category': category,
            'SubCategory': sub_category,
            'ProductID': product_id,
            'ProductName': f"{sub_category} {product_id}",
            'Sales': sales,
            'Quantity': quantity,
            'Discount': discount,
            'Profit': profit,
            'ShippingCost': shipping_cost
        })

    df = pd.DataFrame.from_records(records)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic retail transaction dataset.")
    parser.add_argument('--config', type=str, default=None, help='Path to YAML configuration file.')
    parser.add_argument('--output', type=str, default='.', help='Directory to store generated files.')
    args = parser.parse_args()

    # Determine config path: use provided path or default relative to script
    if args.config:
        config_path = args.config
    else:
        default_config = os.path.join(os.path.dirname(__file__), 'config.yaml')
        config_path = default_config

    config = load_config(config_path)
    df = generate_dataset(config)

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'data.csv')
    excel_path = os.path.join(output_dir, 'data.xlsx')

    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False, engine='openpyxl')

    print(f"Synthetic dataset generated with {len(df)} rows.\nCSV file: {csv_path}\nExcel file: {excel_path}")


if __name__ == '__main__':
    main()