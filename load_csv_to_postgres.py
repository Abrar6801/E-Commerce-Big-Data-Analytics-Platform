import pandas as pd
from sqlalchemy import create_engine

# Database connection
engine = create_engine('postgresql+psycopg2://postgres:postgres123@localhost:5432/ecommerce_data')

# Define CSV file paths
customer_details_path = 'C:/Users/moham/ecommerce-analytics/data/customer_details.csv'
product_details_path = 'C:/Users/moham/ecommerce-analytics/data/product_details.csv'
sales_data_path = 'C:/Users/moham/ecommerce-analytics/data/E-commerece sales data 2024.csv'

# Read CSV files into Pandas DataFrames
customer_df = pd.read_csv(customer_details_path)
product_df = pd.read_csv(product_details_path)
sales_df = pd.read_csv(sales_data_path)

# Load DataFrames into PostgreSQL (this will create the tables if they don't exist)
customer_df.to_sql('customer_details', engine, if_exists='replace', index=False)
product_df.to_sql('product_details', engine, if_exists='replace', index=False)
sales_df.to_sql('sales_data', engine, if_exists='replace', index=False)

print("Tables created and data inserted successfully!")
