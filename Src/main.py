import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
file_path = "data/global air pollution dataset.csv"
df = pd.read_csv(file_path)

# Only string values for cities
df['City'] = df['City'].astype(str)

# Randomly sample a subset of the data
sample_df = df.sample(n=100)

# Create a scatter plot of AQI Value vs. PM2.5 AQI Value
plt.figure(figsize=(8, 6))
plt.scatter(sample_df['AQI Value'], sample_df['PM2.5 AQI Value'], alpha=0.5)
plt.xlabel('AQI Value')
plt.ylabel('PM2.5 AQI Value')
plt.title('Scatter Plot: AQI Value vs. PM2.5 AQI Value')
plt.tight_layout()

# Show the scatter plot
plt.show()
