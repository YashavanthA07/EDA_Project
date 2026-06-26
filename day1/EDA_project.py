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
