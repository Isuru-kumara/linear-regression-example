import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Set a seed
linear_regression_example = 1234
np.random.seed(linear_regression_example)

# Generate synthetic data
num_samples = 250
ad_expenditure = np.random.uniform(0, 250, num_samples)
revenue = 1000 + 10 * ad_expenditure + np.random.normal(0, 100, num_samples)

# Create a DataFrame
data = pd.DataFrame({'AdExpenditure': ad_expenditure, 'Revenue': revenue})

# Save the synthetic dataset to a CSV file
data.to_csv('example_data_set.csv', index=False)

# Display the first 5 rows of the generated data
print(data.head())

# Load Financial Data
data = pd.read_csv('example_data_set.csv')
print(data.head())

# Data Visualization
plt.scatter(data['AdExpenditure'], data['Revenue'])
plt.xlabel('Advertising Expenditure')
plt.ylabel('Revenue')
plt.title('Relationship Between Advertising Expenditure and Revenue')
plt.show()

# Data Splitting
X = data[['AdExpenditure']]
y = data['Revenue']


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
# Create a Linear Regression model.
model = LinearRegression()

# Fit a model to the training data.
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate and display the Mean Squared Error (MSE) for your model's predictions.
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# c. Calculate and display the R-squared (R2) score for your model's performance.
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2) Score: {r2}")

# Data Visualization
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.title('Actual vs. Predicted Revenue on the Testing Set')
plt.show()
