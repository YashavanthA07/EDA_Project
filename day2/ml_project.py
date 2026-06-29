from pandas.core.interchange import dataframe
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

    df = pd.read_csv(file_path)
    print("Datasets Loaded Successfully.Rows:{df.shape[0]},Features:{df.shape[1]}\n")

    print("HANDLING MISSING DATA")
    print("Artificially deleting some 'Hits' (H) data to demonstrate")

    df.loc[0.25, 'H'] = np.nan
    imputer = SimpleImputer(strategy='median')
    df['H'] = imputer.fit_transform(df[['H']])
    print(f"Imputation completed.'Hits' (h)  now has { df['H'].isnull().sum()} null values.\n")

    print("Evaluating the sentence of the Runs(R) Distribution....")


    df["LogRuns"] = np.log1p(df["R"])
    print(f"Log Transformation Applied. New skewness: {df['LogRuns'].skew():.2f} (closer to 0 is perfectly balanced)\n")
    print(df.head())
    df['Team_ID'] = ['Team'+str(np.random.randint(1,150)) for _ in range(len(df))]

    if TargetEncoder is not None:
        print("Applying Target Encoder")

        encoder = TargetEncoder()
        print("Missing values in W:", df['W'].isnull().sum())
        df = df.dropna(subset=['W'])
        df['Team_ID_Encoded'] = encoder.fit_transform(df['Team_ID'], df['W'])

        print(f"\nTarget Encoding Example (First 5 Rows):")
        print(df[['Team_ID', 'Team_ID_Encoded', 'W']].head())

    
    else:
        print("Category Encoders not installed")



    features_to_test=['R','HR','BB','BA']

    X_features = df[features_to_test].fillna(0)
    Y_traget = df['W']

    selector = SelectKBest(score_func = mutual_info_regression,k=2)
        
    winning_features = selector.get_support()
    best_feature = X_features.columns[winning_features].tolist()
    print(f"Best Features:{best_feature}")
if __name__ == "__main__":
    main()
