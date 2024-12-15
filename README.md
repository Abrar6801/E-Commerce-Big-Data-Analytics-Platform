# **E-Commerce Analytics Platform**

## **Overview**
The E-Commerce Analytics Platform is a comprehensive solution designed to process, analyze, and visualize large-scale e-commerce datasets. Leveraging PySpark for distributed data processing, PostgreSQL for structured storage, and React.js for an interactive front end, this project enables businesses to derive actionable insights and make data-driven decisions.

---

## **Features**
- **Data Processing**:
  - Efficient handling of large datasets using PySpark.
  - Optimized queries with PostgreSQL.
- **APIs for Analytics**:
  - Total Sales Analysis
  - Average Order Value
  - Customer Segmentation
  - Product Performance by Season and Category
  - Customer Loyalty Analysis
  - Shipping Type Analysis
  - Sales Forecasting
- **Dynamic Visualizations**:
  - Bar charts, pie charts, line graphs, stacked bar charts, and heatmaps.
  - Pagination for large datasets to improve readability.
- **Predictive Analytics**:
  - Sales forecasting using time-series modeling with Exponential Smoothing.

---

## **Technologies Used**
- **Backend**:
  - PySpark: For distributed data processing.
  - PostgreSQL: For structured data storage and querying.
  - Flask: To expose analytics functionalities via RESTful APIs.
- **Frontend**:
  - React.js: For a dynamic and user-friendly dashboard.
- **Visualization**:
  - Matplotlib & Seaborn: For creating interactive visualizations.
- **Data Handling**:
  - Pandas: For converting and manipulating data efficiently.

---

## **APIs**
The platform offers the following endpoints:

1. **Total Sales**:
   - Endpoint: `/api/total-sales?page=<page_number>`
   - Description: Calculates and visualizes total product sales.

2. **Average Order Value**:
   - Endpoint: `/api/avg-order-by-value?page=<page_number>`
   - Description: Calculates the average purchase amount per customer.

3. **Customer Segmentation**:
   - Endpoint: `/api/customer-segmentation`
   - Description: Categorizes customers into segments (High, Medium, Low Value).

4. **Product Performance**:
   - Endpoint: `/api/performance`
   - Description: Analyzes product performance by season and category.

5. **Customer Loyalty**:
   - Endpoint: `/api/customer-loyalty`
   - Description: Evaluates customer loyalty based on historical data.

6. **Shipping Analysis**:
   - Endpoint: `/api/shipping-analysis`
   - Description: Analyzes and visualizes shipping type distributions.

7. **Sales Forecasting**:
   - Endpoint: `/api/forecast`
   - Description: Provides a forecast of future sales trends.

---

## **Setup Instructions**

### **Backend Setup**
1. Clone the repository:
   git clone https://github.com/<your-repo-name>.git
   cd ecommerce-analytics
2. Create a Python virtual environment and activate it:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install required Python libraries
   pip install -r requirements.txt
4. Update the PostgreSQL connection details in the Flask app.
5. Run the Flask server

### **Frontend Setup**
1. Navigate to the frontend directory
2. Install dependencies
3. Start the React development server
   npm start
4. Access the UI at http://localhost:3000.

---
## **Project Architecture**
1. Data Ingestion: Data is read from PostgreSQL into PySpark for processing.
2. Data Processing: Large datasets are partitioned and transformed using PySpark.
3. API Development: RESTful APIs built with Flask expose analytics endpoints.
4. Frontend Integration: React.js consumes APIs and displays visualizations interactively.

