import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset from the CSV file
file_path = "data/global air pollution dataset.csv"
df = pd.read_csv(file_path)

# Get a list of the columns in the dataset
X = df[['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']]
y = df['AQI Value']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals)
    plt.title('Residuals vs. Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.show()


# Plot residuals
plot_residuals(y_test, y_pred)
# Calculate Root Mean Squared Error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Create a scatter plot to visualize actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual AQI Value")
plt.ylabel("Predicted AQI Value")
plt.title("Actual vs. Predicted AQI Value")
plt.grid(True)

# Create a histogram of the prediction errors
errors = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=20)
plt.xlabel("Prediction Errors")
plt.ylabel("Frequency")
plt.title("Histogram of Prediction Errors")
plt.grid(True)

# Display resulting plots
plt.show()
