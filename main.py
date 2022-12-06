# Author: Lee Taylor, ST Number: 190211479
from sklearn.impute import KNNImputer
import pandas as pd


def printdf(df):
    for col in df: print(col)

pd.set_option('display.max_columns', None)

mar_data = pd.read_csv("houses_0.5_MAR.csv")
ful_data = pd.read_csv("houses.csv")
all_data = [mar_data, ful_data]

md_corr = mar_data.corr()
fd_corr = ful_data.corr()
al_corr = [md_corr, fd_corr]

imputer    = KNNImputer()  # Default nn=5
mar_filled = imputer.fit_transform(mar_data)
printdf(mar_filled)
print("\n...\n")
printdf(mar_data)

if __name__ == '__main__':
    pass
