# Credit Risk Resampling in Machine Learning

## Overview of the Analysis

In this Analyis, we will be using Machine Learning Models that will help examine future credit worthiness of individuals by using historical data from a Peer-to-Peer lending services company.  We will be using two Logistic Regression Machine Learning Models to help us develop better predictions: through Accuracy, Precision, and Recall scores.  We will also oversample the data to deal with imabalanced classes.

![Credit Risk Data Used](/Users/CelesteT/My work/02-Homework/12-Supervised-Learning/Instructions/Starter_Code/Credit_Risk_Data.jpg)

The financial information received from the Peer-to-Peer Lending company included data on the individual's loan size, interest rate, income, debt-to-income, number of credit accounts, "healthy" (0) or "risky" (1) status (denoted by "derogatory"), and finally total debt.  When using `value_counts`, the number of balances of target values, which is in "derogatory" column, numbered 77,536 total.  "Healthy" loans totaled 75,036 and "risky" totaled 2500.   With the "healthy" loans far outnumbering the "risky", this imbalances the predictions as the models will impute them on an equal footing when they are not.  

First, a `LogisticRegression` method from sk.learn was used with the original data provided.  A second `LogisticRegression` model was implemented using oversampled data.  By resampling the data, you can create a more balanced training dataset that contains an equal number of samples from each class, or at least a more proportional representation of the minority class. This can help the model learn from both classes equally and improve its ability to predict the minority class.  This can lead to better predictions and risk assesment.  

## Results

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores:
      * Balanced accuracy score = 0.952
      * Precision score = 0.85
      * Recall score = 0.91


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores:
    * Balanced accuracy score = 0.994
    * Precision score = 0.84
    * Recall score = 0.99

## Summary

The balanced accuracy score for the second model is higher than that of the first model, indicating that the second model performs even better in predicting both classes. The precision score for class 1 is 0.84, meaning that when the model predicts a loan as risky, it is correct 84% of the time. The recall score for class 1 is 0.99, indicating that the model correctly identifies 99% of all risky loans in the dataset.

In comparison, the first model has a slightly lower balanced accuracy score and a lower recall score for class 1, suggesting that it may not be as effective in identifying risky loans as the second model. The performance of a machine learning model depends on the problem we are trying to solve and the specific goals of the task. In some cases, it may be more important to predict the minority class (1s in this case), such as in the case of identifying fraudulent credit card transactions or diagnosing rare diseases. In these cases, it is crucial to have a high recall score for the minority class, as missing even one positive case could have serious consequences.

In other cases, it may be more important to predict the majority class (0s in this case), such as in the case of spam detection or identifying non-fraudulent transactions. In these cases, it is more important to have a high precision score for the majority class, as misclassifying a non-spam message or a non-fraudulent transaction as spam or fraudulent could lead to unnecessary inconvenience or harm to customers.

Therefore, it is important to define the specific problem and goals of the task before selecting and evaluating a machine learning model, and to choose appropriate metrics based on these goals.





