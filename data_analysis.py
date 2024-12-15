from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import col
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from flask import Flask, jsonify, request, send_file
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from io import BytesIO
from flask_cors import CORS
import os


# Set a custom temp directory for Spark
app = Flask(__name__)
CORS(app)
matplotlib.use('Agg')
temp_dir = r"C:\Users\moham\ecommerce-analytics\custom_spark_temp"
os.makedirs(temp_dir, exist_ok=True)  # Make sure the temp directory exists


# Initialize SparkSession
spark = SparkSession.builder \
    .appName("E-commerce Data Analysis") \
    .config("spark.jars", r"C:\Users\moham\ecommerce-analytics\postgresql-42.7.4.jar") \
    .config("spark.local.dir", temp_dir) \
    .config("spark.hadoop.fs.trash.interval", 360) \
    .getOrCreate()

# PostgreSQL database connection properties
db_url = "jdbc:postgresql://localhost:5432/ecommerce_data"
db_properties = {
    "user": "postgres",  # PostgreSQL username
    "password": "postgres123",  # PostgreSQL password
    "driver": "org.postgresql.Driver"
}

# Load customer details from PostgreSQL into Spark DataFrame
customer_df = spark.read.jdbc(url=db_url, table="customer_details", properties=db_properties)
# Load product details from PostgreSQL into Spark DataFrame
product_df = spark.read.jdbc(url=db_url, table="product_details", properties=db_properties)
# Load sales data from PostgreSQL into Spark DataFrame
sales_df = spark.read.jdbc(url=db_url, table="sales_data", properties=db_properties)

# Show the schemas to confirm data is loaded correctly
customer_df.printSchema()
product_df.printSchema()
sales_df.printSchema()

# Convert Spark DataFrames to Pandas DataFrames for visualization
customer_pd = customer_df.toPandas()
product_pd = product_df.toPandas()
sales_pd = sales_df.toPandas()


# Perform data analysis and visualization
# 1. Total sales per product
# Helper function to create horizontal bar chart
def create_horizontal_bar_chart(data, title, x_label, y_label):
    plt.figure(figsize=(10, len(data) * 0.5))  
    plt.barh(data['Product Name'], data['total_quantity_sold'], color='skyblue')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

@app.route('/api/total-sales', methods=["GET"])
def total_sales():
    try:
        # Get the page parameter from the request (default to 1)
        page = int(request.args.get('page', 1))
        items_per_page = 20  # Number of items per page
        offset = (page - 1) * items_per_page

        sales_per_product = sales_df.groupBy("product id").agg(F.count("product id").alias("total_quantity_sold"))

        # Join with product details to get product names
        sales_per_product = sales_per_product.join(product_df, sales_per_product["product id"] == product_df["Uniqe Id"], how="inner")

        # Select relevant columns and sort by total sales
        sales_per_product = sales_per_product.select(product_df["Product Name"], "total_quantity_sold").orderBy(F.desc("total_quantity_sold"))

        # Convert to Pandas DataFrame
        sales_per_product_pd = sales_per_product.toPandas()

        # Handle case when page is out of range
        if offset >= len(sales_per_product_pd):
            return jsonify({"error": f"No data available for page {page}"}), 404

        # Get the relevant slice of data for the current page
        paginated_data = sales_per_product_pd.iloc[offset:offset + items_per_page]

        # Plot total sales per product for the current page
        if not paginated_data.empty:
            create_horizontal_bar_chart(
                paginated_data,
                title=f"Total Sales per Product (Page {page})",
                x_label="Total Quantity Sold",
                y_label="Product Name"
            )

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return send_file(img, mimetype='image/png')

    except Exception as e:
        print(f"Error in /api/total-sales: {e}")
        return jsonify({"error": str(e)}), 500


# 2. Average order value by customer
@app.route('/api/avg-order-by-value', methods=["GET"])
def avg_order_by_value():
    try:
        # Get 'page' and 'page_size' from query parameters, default to 1 and 20
        page = int(request.args.get('page', 1))
        page_size = 20
        offset = (page - 1) * page_size

        # Fetch data and sort
        average_order_value = customer_df.groupBy("Customer ID").agg(F.avg("Purchase Amount (USD)").alias("average_order_value"))
        average_order_value = average_order_value.select("Customer ID", "average_order_value").orderBy(F.desc("average_order_value"))

        # Convert to Pandas DataFrame and apply pagination
        average_order_value_pd = average_order_value.toPandas()

        # Handle case when page is out of range
        if offset >= len(average_order_value_pd):
            return jsonify({"error": f"No data available for page {page}"}), 404

        paginated_data = average_order_value_pd.iloc[offset:offset + page_size]

        # Plot average order value by customer
        if not paginated_data.empty:
            plt.figure(figsize=(15, 8))
            plt.bar(paginated_data['Customer ID'].astype(str), paginated_data['average_order_value'], color="skyblue")
            plt.title(f'Average Order Value by Customer (Page {page})')
            plt.xlabel('Customer ID')
            plt.ylabel('Average Order Value (USD)')
            plt.xticks(rotation=90)
            plt.tight_layout()

        # Save the image to a buffer
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return send_file(img, mimetype='image/png')

    except Exception as e:
        print(f"Error in /api/avg-order-by-value: {e}")
        return jsonify({"error": str(e)}), 500
    
# 3. Customer Segmentation Analysis
@app.route('/api/customer-segmentation',methods=["GET"])
def customer_segmentation():
    try:
        # Convert "Frequency of Purchases" to numeric values (e.g., mapping categories to integers)
        customer_df = spark.read.jdbc(url=db_url, table="customer_details", properties=db_properties)
        customer_df = customer_df.withColumn("Frequency of Purchases Numeric", 
                                            F.when(customer_df["Frequency of Purchases"] == "Daily", 3)
                                            .when(customer_df["Frequency of Purchases"] == "Weekly", 2)
                                            .when(customer_df["Frequency of Purchases"] == "Monthly", 1)
                                            .otherwise(0))

        assembler = VectorAssembler(inputCols=["Purchase Amount (USD)", "Frequency of Purchases Numeric"], outputCol="features")
        customer_features = assembler.transform(customer_df)

        kmeans = KMeans().setK(3).setSeed(1)  # Setting 3 clusters
        model = kmeans.fit(customer_features)

        clustered_customers = model.transform(customer_features)
        print("Customer Segmentation Analysis:")
        clustered_customers.select("Customer ID", "features", "prediction").show(truncate=False)

        # Convert to Pandas DataFrame for visualization
        clustered_customers_pd = clustered_customers.select("Customer ID", "prediction").toPandas()

        # Plot customer segmentation
        if not clustered_customers_pd.empty:
            plt.figure(figsize=(10, 6))
            labels = {0: 'High Value', 1: 'Medium Value', 2: 'Low Value'}
            clustered_customers_pd['Segment'] = clustered_customers_pd['prediction'].map(labels)
            clustered_customers_pd['Segment'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140)
            plt.title('Customer Segmentation')
            plt.ylabel('')
            plt.tight_layout()
            
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close
        return send_file(img, mimetype='image/png')
    
    except Exception as e:
        return jsonify({"error":str(e)}),500

# 4. Product Performance by Season and Category
@app.route('/api/performance', methods=["GET"])
def performance():
    try:
        product_sales_season = sales_df.alias("sales").join(product_df.alias("product"), sales_df["product id"] == product_df["Uniqe Id"], how="inner") \
                                    .join(customer_df.alias("customer"), sales_df["user id"] == customer_df["Customer ID"], how="inner") \
                                    .groupBy("customer.Season", "product.Category") \
                                    .agg(F.count("product.Category").alias("total_sales")) \
                                    .orderBy(F.desc("total_sales"))

        # Convert to Pandas DataFrame for visualization
        product_sales_season_pd = product_sales_season.toPandas()

        # Plot product performance by season and category
        if not product_sales_season_pd.empty:
            pivot_table = product_sales_season_pd.pivot(index='Season', columns='Category', values='total_sales')
            
            plt.figure(figsize=(20, 10))  # Adjust figure size dynamically
            pivot_table.plot(kind='bar', stacked=True, figsize=(20, 10))
            plt.title('Product Performance by Season and Category', fontsize=16)
            plt.xlabel('Season', fontsize=12)
            plt.ylabel('Total Sales', fontsize=12)
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.tight_layout()

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return send_file(img, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 5. Customer Loyalty Analysis
@app.route('/api/customer-loyalty',methods=["GET"])
def customer_loyalty():
    try:
        customer_loyalty = customer_df.groupBy("Customer ID") \
                                    .agg(F.count("Previous Purchases").alias("total_previous_purchases"),
                                        F.count("Promo Code Used").alias("promo_code_usage")) \
                                    .orderBy(F.desc("total_previous_purchases"))

        # Convert to Pandas DataFrame for visualization
        customer_loyalty_pd = customer_loyalty.toPandas()

        # Plot customer loyalty analysis
        if not customer_loyalty_pd.empty:
            plt.figure(figsize=(15, 8))
            plt.plot(customer_loyalty_pd['Customer ID'], customer_loyalty_pd['total_previous_purchases'], marker='o', linestyle='-', color='blue')
            plt.title('Customer Loyalty Analysis')
            plt.xlabel('Customer ID')
            plt.ylabel('Total Previous Purchases')
            plt.grid(True)
            plt.tight_layout()
            

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close
        return send_file(img, mimetype='image/png')
    
    except Exception as e:
        return jsonify({"error":str(e)}),500


    

# 6. Shipping Type Analysis
@app.route('/api/shipping-analysis',methods=["GET"])
def shipping_analysis():
    try:
        shipping_analysis = customer_df.groupBy("Shipping Type").count()

        # Convert to Pandas DataFrame for visualization
        shipping_analysis_pd = shipping_analysis.toPandas()

        # Plot shipping type analysis
        if not shipping_analysis_pd.empty:
            plt.figure(figsize=(10, 6))
            plt.bar(shipping_analysis_pd['Shipping Type'], shipping_analysis_pd['count'], color='orange')
            plt.title('Shipping Type Analysis')
            plt.xlabel('Shipping Type')
            plt.ylabel('Count')
            plt.tight_layout()
            
        
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close
        
        return send_file(img, mimetype='image/png')
    
    except Exception as e:
        return jsonify({"error":str(e)}),500
    

#sales forecasting
@app.route('/api/forecast',methods=["GET"])
def dales_forecast():
    
    try:
        customer_df = spark.read.jdbc(url=db_url, table="customer_details", properties=db_properties)
        product_df = spark.read.jdbc(url=db_url, table="product_details", properties=db_properties)
        sales_df = spark.read.jdbc(url=db_url, table="sales_data", properties=db_properties)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data from PostgreSQL: {e}")
        raise

    # Convert "Time stamp" column to timestamp format for sales_df
    try:
        sales_df = sales_df.withColumn("timestamp", to_timestamp(col("Time stamp"), "dd/MM/yyyy H:mm"))
        print("Timestamp conversion successful.")
    except Exception as e:
        print(f"Error converting timestamp: {e}")
        raise

    try:
        # Convert "Customer ID" and "Purchase Amount (USD)" to Pandas DataFrame for forecasting
        time_series_df = customer_df.select("Purchase Amount (USD)", "Customer ID").toPandas()

        # Ensure numeric sorting and handle missing values
        time_series_df = time_series_df.sort_values(by="Customer ID").dropna()
        time_series_df.rename(columns={"Purchase Amount (USD)": "sales"}, inplace=True)

        # Fit the Exponential Smoothing Model
        model = ExponentialSmoothing(time_series_df["sales"], seasonal="add", seasonal_periods=12).fit()
        forecast = model.forecast(steps=12)

        # Visualize the Forecast
        plt.figure(figsize=(10, 6))
        plt.plot(time_series_df["sales"], label="Actual Sales")
        plt.plot(range(len(time_series_df["sales"]), len(time_series_df["sales"]) + len(forecast)), forecast, label="Forecast", linestyle="--")
        plt.title("Sales Forecast")
        plt.xlabel("Customer ID")
        plt.ylabel("Sales (USD)")
        plt.legend()

        #saves plot to a BytesIO stream
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close
        return send_file(img, mimetype='image/png')

    except Exception as e:
        print(f"An error occurred during sales forecasting: {e}")
        return jsonify({"error":str(e)}),500

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"An error occured: {e}")
    finally:
        if 'spark' in locals():
            spark.stop()

    
