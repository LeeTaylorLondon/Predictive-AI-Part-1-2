# CSC3831 Predictive Analytics and A.I. - Data Imputation and Machine Learning

## Brief Overview
The performance of the same linear regression model is
trained on two differently imputed datasets. In summary:

1. The original dataset is split into a training set and a test set. I manually removed the last 2500 rows from 
both the complete dataset and incomplete dataset, which forms the test dataset. The training set
will be used to fit the models, and the test set will be used to evaluate the performance of the models.

2. The incomplete dataset is duplicated where is filled by the KNN-Imputer and the other MICE-Imputer.

3. A linear regression model is trained on the training set for each imputed dataset. 

4. The models are used to predict the target variable on the test set. 

5. The performance metrics for each model, such as the mean squared error (MSE) and the R^2 
score are calculated and displayed.

## Results

-------------------------------
KNN-Imputed Dataset Results:  
MSE: 4.21e+09  
R^2: 0.68  
Execution time: 0.010793 seconds  
-------------------------------
MICE-Imputed Dataset Results:  
MSE: 3.84e+09  
R^2: 0.7  
Execution time: 0.003452 seconds  
-------------------------------

## Questions

### Are there differences in the results?
Yes, there are differences in the results of the two models. The model trained on the MICE-imputed dataset 
has a lower mean squared error (MSE) and a higher R^2 value than the model trained on the KNN-imputed dataset.

The MSE is a measure of the difference between the predicted values and the true values, with lower values
indicating a better fit. The R^2 value is a measure of the amount of variance explained by the model, with
values closer to 1 indicating a better fit.

In this case, the model trained on the MICE-imputed dataset has a lower MSE and a higher R^2 value, indicating
that it is a better fit for the data than the model trained on the KNN-imputed dataset.

It's worth noting that these results are specific to the particular datasets and model being used, and may
not necessarily hold true for all datasets and models.

### Why are there differences?
There are a few reasons why the models trained on differently imputed datasets might provide different results.

First, the imputation methods themselves can affect the results. KNN imputation and MICE imputation are two different
techniques for filling in missing values in a dataset. KNN imputation uses the values of the nearest neighbors to fill
in missing values, while MICE imputation uses multiple imputation to fill in missing values. These two methods can
produce different imputed values for the same missing data, which can in turn lead to different model results.

Second, the quality of the imputed data can affect the model results. If the imputed data is of high quality,
it can lead to more accurate model results. However, if the imputed data is of low quality, it can lead to less
accurate model results. The quality of the imputed data can depend on various factors, such as the amount of missing
data, the nature of the data, and the specific imputation method used.

Finally, the amount of execution time can also affect the model results. In this case, the model trained on the
KNN-imputed dataset took longer to execute than the model trained on the MICE-imputed dataset. This could be
because the KNN imputation method is more computationally intensive than the MICE imputation method. It's also
possible that other factors, such as the complexity of the model or the size of the dataset, could have contributed
to the difference in execution time.

### Which model would I choose?
Based on the results of the two models, it appears that the model trained on the MICE-imputed dataset performed
slightly better than the model trained on the KNN-imputed dataset. This is indicated by the higher R^2 value and
lower MSE of the MICE-imputed model. Therefore, in this case I would choose M2.

However, generally I should also consider the quality of the imputed data. If the MICE-imputed data 
is of higher quality than the KNN-imputed data, that could be a significant factor in the better
performance of the MICE-imputed model. It's also possible that the MICE imputation method may be
more appropriate for the specific dataset and model being used.

Overall, while the MICE-imputed model appears to have slightly better performance based on the 
MSE and R^2 values, it's important to consider all relevant factors when deciding which model
to choose for varying scenarios.

### Author
* Lee Taylor, ST Number: 190211479
