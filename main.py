# Author: Lee Taylor, ST Number: 190211479
import numpy    as np
import pandas   as pd
from sklearn.impute             import KNNImputer
from sklearn.linear_model       import LinearRegression
from sklearn.model_selection    import train_test_split, KFold
from sklearn.experimental       import enable_iterative_imputer
from sklearn.impute             import IterativeImputer
from sklearn.metrics            import r2_score,  mean_squared_error


# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# > Read datasets
mar_data = pd.read_csv("houses_0.5_MAR.csv")
ful_data = pd.read_csv("houses.csv")
all_data = [mar_data, ful_data]

# > Create correlation matrices
# md_corr = mar_data.corr()
# fd_corr = ful_data.corr()
# al_corr = [md_corr, fd_corr]

# > Fill missing data KNN
imputer        = KNNImputer()  # Default nn=5
mar_filled_knn = imputer.fit_transform(mar_data)
del imputer  # Prevent re-using imputer in next section

# > Fill missing data MICE
mar_filled_mice = mar_data.__copy__()
imputer = IterativeImputer(missing_values=np.nan, add_indicator=False,
                           random_state=0, n_nearest_features=5,
                           sample_posterior=True)
imputer.fit(mar_data)
mar_filled_mice = imputer.transform(mar_filled_mice)

# > Container for imputed datasets
imputed_data = [pd.DataFrame(mar_filled_knn, columns=("index", "median_house_value","median_income","housing_median_age",
                                                      "total_rooms","total_bedrooms","population","households",
                                                      "latitude","longitude")),
                pd.DataFrame(mar_filled_mice, columns=("index", "median_house_value","median_income","housing_median_age",
                                                      "total_rooms","total_bedrooms","population","households",
                                                      "latitude","longitude"))]

# > Debug
print(f"{type(imputed_data[0])}\n{imputed_data[0].columns}")

def xtraintest(imputed_dataset, target='median_house_value', debug=False):
    """ Passed an imputed dataset, training and testing datasets are returned """
    X_train_ = imputed_dataset.copy().drop([target, 'index'], axis=1)
    X_test_  = ful_data.copy().drop(target, axis=1)
    if debug: print(f"X_train.shape={X_train_.shape}, X_test.shape={X_test_.shape}")
    return X_train_, X_test_

# > Original complete dataset into train & test datasets
y_train_true = ful_data['median_house_value']
y_test_true  = ful_data['median_house_value']

def traintestmodel(tup, iname):
    """ Create, train, and evaluate a regression model """
    X_train, X_test = tup[0], tup[1]
    # Create reproducible regression model
    np.random.seed(100)
    clf = LinearRegression()
    # Train model
    clf.fit(X_train.values, y_train_true.values)
    y_test_pred = clf.predict(X_test.values)
    print(f"{iname}-Imputed Dataset Results:")
    print("RMSE: {0:.3}".format(mean_squared_error(y_test_true, y_test_pred)))
    print("R^2: {0:.2}".format(r2_score(y_test_true, y_test_pred)))
    print()


if __name__ == '__main__':
    # > Train regression models on the imputed datasets respectively
    traintestmodel(xtraintest(imputed_data[0]), "KNN")
    traintestmodel(xtraintest(imputed_data[1]), "MICE")

    # > Debug information
    # print('\n...\n')
    # print(f"data.shape      = {mar_data.shape}")
    # print(f"data_knn.shape  = {mar_filled_knn.shape}")
    # print(f"data_mice.shape = {mar_filled_mice.shape}")

    # > Debug information
    # > (KNN-Imputed) Train & Test dataset
    # X_train, X_test = xtraintest(imputed_data[0])
    # print(f"{X_train.shape} {X_test.shape}")

    # > (MICE-Imputed) Train & Test dataset
    # X_train, X_test = xtraintest(imputed_data[1])
    # print(f"{X_train.shape} {X_test.shape}")

    # > Debug info about the original complete dataset
    # print(f"{type(ful_data)}\n{ful_data.columns}")
    pass
