
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('titanic.csv')

# 1. Summary Statistics
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe(include='all'))
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Histograms
df.hist(bins=30, figsize=(12, 10))
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout()
plt.show()

# 3. Boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age Distribution by Survival')
plt.show()

# 4. Pairplot (subset)
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']], hue='Survived')
plt.suptitle('Pairplot of Key Features', y=1.02)
plt.show()

# 5. Correlation Matrix + Heatmap
corr = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
