# _Credit Risk Analysis_

### Overview

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, in this project we will need to employ different techniques to train and evaluate models with unbalanced classes. I am using imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we will oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, we will use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, we will compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once this is finalized, we will evaluate the performance of these models and make recommendation on whether they should be used to predict credit risk.

### Results of the Analysis

#### Resampling Models to Predict Credit Risk

1. Naive Random Oversampling using RandomOverSampler

#### Results: 

- The balanced accuracy score for this model was 66%
- The precision score for high risk individuals was 1% with a recall score of 74%
- The precision score for low risk individuals was 100% with a recall score of 58%
- The overall average precision score was 99% with a recall score of 58%

see code below

![alt text](https://github.com/Yoditatr/Credit_Risk_Analysis/blob/main/Resources/Naive%20Random%20Oversampling.PNG?raw=true)

2. SMOTE Oversampling 

#### Results:

- The balanced accuracy score for this model was 65%
- The precision score for high risk individuals was 1% with a recall score of 62%
- The precision score for low risk individuals was 100% with a recall score of 68%
- The overall average precision score was 99% with a recall score of 68%

see code below

![alt text](https://github.com/Yoditatr/Credit_Risk_Analysis/blob/main/Resources/Smote%20oversampling.PNG?raw=true)

3. Undersampling using ClusterCentroids

#### Results:

- The balanced accuracy score for this model was 59%
- The precision score for high risk individuals was 1% with a recall score of 64%
- The precision score for low risk individuals was 100% with a recall score of 53%
- The overall average precision score was 99% with a recall score of 53%

see code below

![alt text](https://github.com/Yoditatr/Credit_Risk_Analysis/blob/main/Resources/undersampling.PNG?raw=true)


4. Combinatorial using SMOTEEN

#### Results:

- The balanced accuracy score for this model was 64%
- The precision score for high risk individuals was 1% with a recall score of 72%
- The precision score for low risk individuals was 100% with a recall score of 57%
- The overall average precision score was 99% with a recall score of 57%

see code below

![alt text](https://github.com/Yoditatr/Credit_Risk_Analysis/blob/main/Resources/SmoteANN.PNG?raw=true)

### Ensemble Classifiers to Predict Credit Risk

5. Balanced Random Forest Classifier using BalancedRandomForestClassifier

#### Results:

- The balanced accuracy score for this model was 78%
- The precision score for high risk individuals was 3% with a recall score of 57%
- The precision score for low risk individuals was 100% with a recall score of 88%
- The overall average precision score was 99% with a recall score of 88%


see code below

![alt text](https://github.com/Yoditatr/Credit_Risk_Analysis/blob/main/Resources/Ensemble%20_%20Balance%20Random%20Forest%20Classifier.PNG?raw=true)

6. Easy Ensemble AdaBoost Classifier using EasyEnsembleClassifier

#### Results:

- The balanced accuracy score for this model was 92%
- The precision score for high risk individuals was 9% with a recall score of 89%
- The precision score for low risk individuals was 100% with a recall score of 94%
- The overall average precision score was 99% with a recall score of 94%

![alt text](https://github.com/Yoditatr/Credit_Risk_Analysis/blob/main/Resources/easy%20ensemble%20adaboost%20classifier.PNG?raw=true)
