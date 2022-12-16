# Author: Lee Taylor, ST Number: 190211479
import time
import numpy                    as np
import pandas                   as pd
from sklearn.impute             import KNNImputer
from sklearn.linear_model       import LinearRegression
from sklearn.model_selection    import train_test_split, KFold
from sklearn.experimental       import enable_iterative_imputer
from sklearn.impute             import IterativeImputer
from sklearn.metrics            import r2_score,  mean_squared_error


# > Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# > Read datasets
mar_data = pd.read_csv("houses_0.5_MAR.csv")
ful_data = pd.read_csv("houses.csv")
all_data = [mar_data, ful_data]

# print(mar_data.shape)
# print(ful_data.shape)

# > Omit the last 2500 rows from mar_data and ful_data
mar_data_new = mar_data.iloc[:-2500, :]
ful_data_new = ful_data.iloc[:-2500, :]

# > Save the last 2500 rows to another dataframe
mar_data_omitted = mar_data.iloc[-2500:, :]
ful_data_omitted = ful_data.iloc[-2500:, :]

# > Create correlation matrices
md_corr = mar_data.corr()
fd_corr = ful_data.corr()
al_corr = [md_corr, fd_corr]

# > Fill missing data KNN
imputer        = KNNImputer()  # Default nn=5
mar_filled_knn = imputer.fit_transform(mar_data_new.__copy__())
del imputer  # Prevent re-using incorrect imputer in next section

# > Fill missing data MICE
mar_filled_mice = mar_data_new.__copy__()
imputer = IterativeImputer(missing_values=np.nan, add_indicator=False,
                           random_state=0, n_nearest_features=5,
                           sample_posterior=True)
imputer.fit(mar_data_new)
mar_filled_mice = imputer.transform(mar_filled_mice)

# > Container for imputed datasets
imputed_data = [pd.DataFrame(mar_filled_knn, columns=("index", "median_house_value","median_income","housing_median_age",
                                                      "total_rooms","total_bedrooms","population","households",
                                                      "latitude","longitude")),
                pd.DataFrame(mar_filled_mice, columns=("index", "median_house_value","median_income","housing_median_age",
                                                      "total_rooms","total_bedrooms","population","households",
                                                      "latitude","longitude"))]

# # > Debug
# print(f"{type(imputed_data[0])}\n{imputed_data[0].columns}")

def xtraintest(imputed_dataset, target='median_house_value', debug=False):
    """ Passed an imputed dataset, training and testing datasets are returned """
    X_train_ = imputed_dataset.copy().drop([target, 'index'], axis=1)   # (~18000, 8)
    Y_train_ = ful_data_new.copy()[target]                              # (~18000, 1)
    if debug: print(f"X_train.shape={X_train_.shape}, Y_train.shape={Y_train_.shape}")
    return X_train_, Y_train_

# > Original complete dataset into train & test datasets
X_test = ful_data_omitted.copy().drop('median_house_value', axis=1)
Y_test = ful_data_omitted['median_house_value']
print(f"Shapes:\n"
      f"X_test -> {X_test.shape}\n"
      f"Y_test -> {Y_test.shape}")

def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Execution time: {end_time - start_time:.6f} seconds")
        return result
    return wrapper

@print_execution_time
def traintestmodel(tup, iname):
    """ Create, train, and evaluate a regression model """
    X_train, Y_train = tup[0], tup[1]
    print(f"\ntraintestmodel()\n"
          f"X_train.shape {X_train.shape}\n"
          f"Y_train.shape {Y_train.shape}\n")
    # Create reproducible regression model
    np.random.seed(100)
    clf = LinearRegression()
    # Train model
    clf.fit(X_train.values, Y_train.values)
    y_test_pred = clf.predict(X_test.values)
    print(f"-------------------------------")
    print(f"{iname}-Imputed Dataset Results:")
    print("MSE: {0:.3}".format(mean_squared_error(Y_test, y_test_pred)))
    print("R^2: {0:.2}".format(r2_score(Y_test, y_test_pred)))
    print(f"-------------------------------\n")
    return clf


if __name__ == '__main__':
    # # > Debug information
    # print('\n...\n')
    # print(f"data.shape      = {mar_data.shape}")
    # print(f"data_knn.shape  = {mar_filled_knn.shape}")
    # print(f"data_mice.shape = {mar_filled_mice.shape}")

    # # > Debug information
    # # > (KNN-Imputed) Train & Test dataset
    # X_train, Y_train = xtraintest(imputed_data[0])
    # print(f"KNN:\n"
    #       f"x_train -> {X_train.shape}\n"
    #       f"x_test -> {Y_train.shape}")
    #
    # # > (MICE-Imputed) Train & Test dataset
    # X_train, Y_train = xtraintest(imputed_data[1])
    # print(f"MICE:\n"
    #       f"x_train -> {X_train.shape}\n"
    #       f"x_test -> {Y_train.shape}")

    # > Train regression models on the imputed datasets respectively
    # > X_train, Y_train = xtraintest(imputed_data[0])
    m1 = traintestmodel(xtraintest(imputed_data[0]), "KNN")
    m2 = traintestmodel(xtraintest(imputed_data[1]), "MICE")

    pass
