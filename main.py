# Author: Lee Taylor, ST Number: 190211479
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np


def printdf(df):
    for col in df: print(col)


# Configure pandas display
pd.set_option('display.max_columns', None)

# > Read datasets
mar_data = pd.read_csv("houses_0.5_MAR.csv")
ful_data = pd.read_csv("houses.csv")
all_data = [mar_data, ful_data]

# > Create correlation matrices
md_corr = mar_data.corr()
fd_corr = ful_data.corr()
al_corr = [md_corr, fd_corr]

# > Fill missing data KNN
imputer        = KNNImputer()  # Default nn=5
mar_filled_knn = imputer.fit_transform(mar_data)

# > Fill missing data MICE
mar_filled_mice = mar_data.__copy__()
lr  = LinearRegression()
imp = IterativeImputer(estimator=lr, verbose=2, max_iter=30, tol=1e-10,
                       imputation_order='roman')
imp.fit(mar_data)
imp.transform(mar_filled_mice)

# > Inspect datasets
# printdf(mar_filled)
# print("\n...\n")
# printdf(mar_data)


if __name__ == '__main__':
    print('\n...\n')
    print(f"data.shape      = {mar_data.shape}")
    print(f"data_knn.shape  = {mar_filled_knn.shape}")
    print(f"data_mice.shape = {mar_filled_mice.shape}")
    pass
