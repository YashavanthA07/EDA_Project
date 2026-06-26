import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Understanding dataset")

file_name='sales_data.csv'
if not os.path.exists(file_name):
    print(f"error:{file_name}is not found")
    exit()

df=pd.read_csv(file_name)
print("Successfully loaded ")
print(f"Shape of the dataset:Rows:{df.shape[0]},Columns:{df.shape[1]}")

print(df.head())
print(df.tail())
print(df.describe())

print("Handling Missing Values")

print(df.isnull().sum())

median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(median_age)

median_spending=df['Spending'].median()
df['Spending']=df['Spending'].fillna(median_age)
print(median_spending)

mean_age=df['Age'].mean()
df['Age']=df['Age'].fillna(mean_age)
print(mean_age)

mean_spending=df['Spending'].mean()
df['Spending']=df['Spending'].fillna(mean_age)
print(mean_spending)


plt.figure(figsize=(7,4))
df['Spending'].hist(bins=10,color='cyan',edgecolor='black')
plt.title("Distribution of Spending")
plt.xlabel('Spending Amount')
plt.ylabel('Number of Customers')
plt.show()

correlation = df.corr(numeric_only=True)
print(correlation)

print("plotting Correlation Heatmap")
plt.figure(figsize=(7,4))
sns.heatmap(correlation,annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(7,4))
plt.scatter(df['Age'],df['Spending'],color='red',alpha=0.5)
plt.title('Age vs Spending')
plt.xlabel('Age')
plt.ylabel('Spending')
plt.show()

plt.figure(figsize=(7,4))
sns.boxplot(x=df['Age'],color='lightgreen')
plt.title("Boxplot of Customer Age")
plt.xlabel('Age')
plt.show()


print("Find the Outliers in age")
outliers=(df['Age']>100)
print("Found Outliers(s):")
print(outliers)
