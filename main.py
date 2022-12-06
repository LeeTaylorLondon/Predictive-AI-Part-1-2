# Author: Lee Taylor, ST Number: 190211479
from decimal import Decimal
import pandas as pd

pd.set_option('display.max_columns', None)

mar_data = pd.read_csv("houses_0.5_MAR.csv")
ful_data = pd.read_csv("houses.csv")
all_data = [mar_data, ful_data]

for data in all_data: print(data)

md_corr = mar_data.corr()
fd_corr = ful_data.corr()
al_corr = [md_corr, fd_corr]
for corr in al_corr: print(corr)

if __name__ == '__main__':
    pass
