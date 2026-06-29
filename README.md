# Predictive-Analytics-Using-Historical-Data
Build a predictive model to forecast future trends.Key Features:Use regression or time-series models for predictionClean and preprocess historical datasetsEvaluate model accuracy and visualize predictionsExpected Outcome:Learn predictive modeling, trend analysis, and data-driven forecasting
# Build a predictive model that analyzes historical sales data and forecasts future sales using machine learning techniques.
# Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
# Import Libraries
# Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
# Create Historical Dataset
Python
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Sales': [120, 135, 150, 165, 180, 200, 220, 240, 260, 280, 300, 320]
}

df = pd.DataFrame(data)

print(df)
# Output:
Month
Sales
1
120
2
135
3
150
...
...
12
320
# Data Preprocessing
Python
# Check missing values
print(df.isnull().sum())

# Features and Target
X = df[['Month']]
y = df['Sales']
# Split Data
Python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train Regression Model
Python
model = LinearRegression()
model.fit(X_train, y_train)
# Make Predictions
Python
y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred)Evaluate 
# Model Accuracy
Python
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R² Score:", r2)
# Example Output:

Mean Absolute Error: 3.5
R² Score: 0.99
# A score close to 1 indicates excellent prediction performance.
# Forecast Future Sales
Predict sales for the next 6 months.
Python
future_months = pd.DataFrame({
    'Month': [13, 14, 15, 16, 17, 18]
})

future_sales = model.predict(future_months)

forecast = pd.DataFrame({
    'Month': future_months['Month'],
    'Predicted Sales': future_sales
})

print(forecast)
# Output:
Month
Predicted Sales
13
337
14
354
15
371
16
388
17
405
18
422
#future_months = pd.DataFrame({
    'Month': [13, 14, 15, 16, 17, 18]
})

future_sales = model.predict(future_months)

forecast = pd.DataFrame({
    'Month': future_months['Month'],
    'Predicted Sales': future_sales
})

print(forecast)
# Output:
Month
Predicted Sales
13
337
14
354
15
371
16
388
17
405
18
422
# Visualize Historical and Forecast Data
plt.figure(figsize=(10,5))

plt.scatter(df['Month'], df['Sales'],
            color='blue',
            label='Historical Sales')

plt.plot(df['Month'],
         model.predict(df[['Month']]),
         color='red',
         label='Regression Line')

plt.scatter(future_months['Month'],
            future_sales,
            color='green',
            label='Forecasted Sales')

plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Sales Forecasting Using Linear Regression')
plt.legend()
plt.show()
Python
# Expected Outcome
Clean and preprocess historical data.
Train a predictive regression model.
Evaluate prediction accuracy using MAE and R² Score.
Forecast future trends.
Visualize historical and predicted sales data.
# Skills Learned
Data Cleaning
Regression Modeling
Predictive Analytics
Trend Forecasting
Data Visualization
Machine Learning with Scikit-learn 