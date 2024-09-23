import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Replace 'housing_data.csv' with your dataset path
data = pd.read_csv('D:/datasets/house_price.csv')
print(data.head())
print(data.info())

# Handle Missing Values
if data.isnull().sum().any():
    data.fillna(data.mean(numeric_only=True), inplace=True)  

# Feature Selection
if 'Price' not in data.columns:
    raise KeyError("The target variable 'Price' is not in the dataset.")

X = data.drop('Price', axis=1)  # Features
y = data['Price']                # Target variable

# One-Hot Encoding for categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (only for numeric columns)
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions
y_pred_lin = lin_reg.predict(X_test)

# Evaluation Function
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2


# Evaluate the linear regression model
mse_lin, r2_lin = evaluate_model(y_test, y_pred_lin)

print(f'Linear Regression: MSE={mse_lin}, R²={r2_lin}')

# Visualization
plt.figure(figsize=(18, 6))
plt.subplot(1, 1, 1)  # Changed to a single plot
plt.scatter(y_test, y_pred_lin)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.title('Linear Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()
