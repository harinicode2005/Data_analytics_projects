import pandas as pd

# Load data
df = pd.read_csv("dataset.csv")

# -------------------------------
# 1. Basic Overview
# -------------------------------
print("Shape of dataset:", df.shape)
print("\nColumn names:\n", df.columns)
print("\nData types:\n", df.dtypes)

# -------------------------------
# 2. First & Last Rows
# -------------------------------
print("\nHead:\n", df.head())
print("\nTail:\n", df.tail())

# -------------------------------
# 3. Missing Values
# -------------------------------
print("\nMissing values:\n", df.isnull().sum())

# Percentage missing
print("\nMissing %:\n", (df.isnull().sum() / len(df)) * 100)

# -------------------------------
# 4. Statistical Summary
# -------------------------------
print("\nNumerical summary:\n", df.describe())

# Include categorical
print("\nFull summary:\n", df.describe(include='all'))

# -------------------------------
# 5. Unique Values
# -------------------------------
for col in df.columns:
    print(f"\nUnique values in {col}: {df[col].nunique()}")

# -------------------------------
# 6. Correlation (Very Important)
# -------------------------------
corr = df.corr(numeric_only=True)
print("\nCorrelation matrix:\n", corr)

# Top correlations with GOLD_PRICE
print("\nTop correlations with GOLD_PRICE:\n",
      corr['GOLD_PRICE'].sort_values(ascending=False))

# -------------------------------
# 7. Sorting & Time Handling
# -------------------------------
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df = df.sort_values('DATE')

print("\nDate range:", df['DATE'].min(), "to", df['DATE'].max())

# -------------------------------
# 8. Grouping (Year-wise analysis)
# -------------------------------
df['YEAR'] = df['DATE'].dt.year

yearly_avg = df.groupby('YEAR')[['GOLD_PRICE', 'SILVER_PRICE']].mean()
print("\nYearly average prices:\n", yearly_avg)

# -------------------------------
# 9. Returns (Important for finance)
# -------------------------------
df['GOLD_RETURN'] = df['GOLD_PRICE'].pct_change()
df['SILVER_RETURN'] = df['SILVER_PRICE'].pct_change()

print("\nReturns summary:\n", df[['GOLD_RETURN', 'SILVER_RETURN']].describe())

# -------------------------------
# 10. Event Analysis
# -------------------------------
event_data = df[df['EVENT'].notnull()]
print("\nEvent rows:\n", event_data[['DATE', 'EVENT', 'GOLD_PRICE', 'SILVER_PRICE']])

# -------------------------------
# 11. Duplicate Check
# -------------------------------
print("\nDuplicate rows:", df.duplicated().sum())

# -------------------------------
# 12. Outlier Detection (Simple)
# -------------------------------
Q1 = df['GOLD_PRICE'].quantile(0.25)
Q3 = df['GOLD_PRICE'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['GOLD_PRICE'] < (Q1 - 1.5 * IQR)) | 
              (df['GOLD_PRICE'] > (Q3 + 1.5 * IQR))]

print("\nGold price outliers:\n", outliers[['DATE', 'GOLD_PRICE']])