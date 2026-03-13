import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Find CSV
script_dir = os.path.dirname(__file__)
path = os.path.join(script_dir, 'imd_dataset.csv') if os.path.exists(os.path.join(script_dir, 'imd_dataset.csv')) else 'imd_dataset.csv'
if not os.path.exists(path): print(f"CSV not found at {path}"); sys.exit(1)

df = pd.read_csv(path, sep='\t')
print(df.head(3))
print(df.tail(3))

# Tasks 2-4
print(df.head(3), df.tail(3), sep='\n\n')
df.info()
print("\nNulls:", df.isnull().sum().sum())

# Task 5
subset = df.iloc[40:75]
print(f"\nSubset shape: {subset.shape}")

# Task 6
idx = df['No_of_Votes'].astype(float).idxmax()
print(f"\nTop voted movie:\n{df.loc[idx]}")

# Prepare runtime
runtimes = pd.to_numeric(df['Runtime'], errors='coerce')

# Task 7: Boxplots
plt.figure(figsize=(8,3))
plt.subplot(121); sns.boxplot(y=df['IMDB_Rating'])
plt.subplot(122); sns.boxplot(y=runtimes)
plt.show()

# Task 8: Scatter
plt.scatter(df['IMDB_Rating'], runtimes, alpha=0.6)
plt.xlabel('IMDB_Rating'); plt.ylabel('Runtime')
plt.show()

# Task 9: Histograms
plt.figure(figsize=(8,3))
plt.subplot(121); sns.histplot(df['IMDB_Rating'], kde=True)
plt.subplot(122); sns.histplot(runtimes.dropna(), kde=True)
plt.show()

# Task 10: Count plot
rating_col = 'Certificate' if 'Certificate' in df.columns else 'Rating' if 'Rating' in df.columns else None
if rating_col: sns.countplot(x=df[rating_col]); plt.xticks(rotation=45); plt.show()
else: print('No rating column')