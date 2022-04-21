import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression


df = pd.read_csv ("./data/Coca-Cola_stock_history.csv")

# This new field together with volume will measure the volatility of the stock
df["HL_PCT"] = (df["High"] - df["Close"]) / df["Close"] * 100

# This will measure if the stock price goes up or down after a day
df["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100

df = df[["HL_PCT", "PCT_change", "Close", "Volume"]]

forecast_col = "Close"

forecast_out = math.ceil(.001 * len(df))

df["label"] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(["label"], 1))
Y = np.array(df["label"])

# Standardization of X
X = preprocessing.scale(X)

X_test, X_train, Y_test, Y_train = model_selection.train_test_split(X, Y, test_size=.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(accuracy)

clf = svm.SVR()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(accuracy)