import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

data = load_breast_cancer(as_frame=True)
df=data.frame
df.head()

X = df.drop("target",axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"trainning Data Size:{X_train.shape}")
print(f"Testing Data Size:{X_test.shape}\n")

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

print("Model trained:Random Forest")

rf_pred = rf.predict(X_test)