# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load dataset
df = pd.read_csv("data/dataset.csv")

# Step 3: Basic info
print(df.info())
print(df.describe())

# Step 4: Convert DATE column
df['DATE'] = pd.to_datetime(df['DATE'])

# Step 5: Handle missing values
df = df.dropna()

# Step 6: Sort data by date
df = df.sort_values(by='DATE')

# Step 7: Save cleaned data
df.to_csv("outputs/cleaned_data.csv", index=False)

# Step 8: Plot Gold price over time
plt.figure(figsize=(10,5))
plt.plot(df['DATE'], df['GOLD_PRICE'])
plt.title("Gold Price Over Time")
plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.xticks(rotation=45)
plt.show()

# Step 9: Plot Silver price
plt.figure(figsize=(10,5))
plt.plot(df['DATE'], df['SILVER_PRICE'], color='orange')
plt.title("Silver Price Over Time")
plt.show()

# Step 10: Correlation heatmap
plt.figure(figsize=(10,6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# Step 11: Relationship between risk & gold
sns.scatterplot(x=df['GPRD'], y=df['GOLD_PRICE'])
plt.title("Geopolitical Risk vs Gold Price")
plt.show()