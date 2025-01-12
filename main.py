# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 1: Load the dataset
file_path = 'Historical_Product_Demand.csv'
df = pd.read_csv(file_path)

# Step 2: Inspect the dataset
print(df.head())
print("Missing values:\n", df.isnull().sum())

# Step 3: Data Preprocessing
# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Remove rows with invalid dates
df = df.dropna(subset=['Date'])

# Handle missing values for the rest of the dataset (forward fill for continuous values)
df = df.ffill()

# Convert 'Order_Demand' to numeric, and handle any invalid values
df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce')

# Replace any remaining missing or invalid values with the median
df['Order_Demand'].fillna(df['Order_Demand'].median(), inplace=True)

# Step 4: Feature Engineering
# Extract year, month, and day from the 'Date' column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Create a feature for demand elasticity (percentage change in demand over time)
df['Demand_Elasticity'] = df['Order_Demand'].pct_change().fillna(0)

# Handle categorical columns ('Warehouse' and 'Product_Category') by encoding them as dummy variables
df = pd.get_dummies(df, columns=['Warehouse', 'Product_Category'], drop_first=True)

# Step 5: Prepare data for modeling
# Define features (X) and target (y)
X = df.drop(columns=['Date', 'Order_Demand', 'Product_Code'])
y = df['Order_Demand']

# Check for any remaining infinite or NaN values in the features
print("Checking X for infinite or NaN values:\n", X.isnull().sum())
print("Checking for infinities in X:\n", np.isinf(X).sum())

# Replace any remaining infinite values in X with NaN and fill with median
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plot feature importance
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()

# Step 8: Save the model
joblib.dump(model, 'pricing_optimization_model.pkl')
print("Model saved as 'pricing_optimization_model.pkl'")

# Additional Visualizations

# 1. Order Demand Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Order_Demand'], kde=True, color='blue')
plt.title('Distribution of Order Demand')
plt.xlabel('Order Demand')
plt.ylabel('Frequency')
plt.show()

# 2. Time Series Plot of Order Demand
plt.figure(figsize=(10, 6))
sns.lineplot(x=df['Date'], y=df['Order_Demand'])
plt.title('Order Demand over Time')
plt.xlabel('Date')
plt.ylabel('Order Demand')
plt.show()

# 3. Boxplot for Order Demand by Month
plt.figure(figsize=(10, 6))
sns.boxplot(x='Month', y='Order_Demand', data=df)
plt.title('Order Demand Distribution by Month')
plt.xlabel('Month')
plt.ylabel('Order Demand')
plt.show()

# 4. Correlation Heatmap - Exclude non-numeric columns before computing the correlation matrix
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])  # Only select numeric columns
correlation_matrix = numeric_df.corr()  # Compute correlation only for numeric columns
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Features')
plt.show()

# Display the columns (features) used by the model
features = X.columns
print("Features used for training the model:", features)
