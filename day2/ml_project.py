import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os 

try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders not found")


def main():
    print("loading dataset")
    file_path='train.csv'
    

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df=pd.read_csv(file_path)
    print("Datasets Loaded Successfully.Rows:{df.shape[0]},Features:{df.shape[1]}\n")
    print(df.head())
    print(df.tail())

    print("\nBasic Statistics:")
    print(df.describe())

if __name__ == "__main__":
    main()
