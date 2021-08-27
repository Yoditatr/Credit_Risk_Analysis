# _Credit Risk Analysis with Supervised Machine Learning_

![alt text](https://github.com/Yoditatr/Credit_Risk_Analysis/blob/main/pic.jpg?raw=true)

### Overview

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, in this project we will need to employ different techniques to train and evaluate models with unbalanced classes. I am using imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we will oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, we will use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, we will compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once this is finalized, we will evaluate the performance of these models and make recommendation on whether they should be used to predict credit risk.

### Results of the Analysis

#### Resampling Models to Predict Credit Risk

1. Naive Random Oversampling using RandomOverSampler

#### Results: 

- The balanced accuracy score for this model was 65%
- The precision score for high risk individuals was 1% with a recall score of 69%
- The precision score for low risk individuals was 100% with a recall score of 60%
- The overall average precision score was 99% with a recall score of 60%

see code below

![alt text](https://github.com/Yoditatr/Credit_Risk_Analysis/blob/main/Resources/Naive%20Random%20Oversampling.PNG?raw=true)

2. SMOTE Oversampling 

#### Results:

- The balanced accuracy score for this model was 66%
- The precision score for high risk individuals was 1% with a recall score of 63%
- The precision score for low risk individuals was 100% with a recall score of 69%
- The overall average precision score was 99% with a recall score of 69%

see code below

![alt text](https://github.com/Yoditatr/Credit_Risk_Analysis/blob/main/Resources/Smote%20oversampling.PNG?raw=true)

3. Undersampling using ClusterCentroids

#### Results:

- The balanced accuracy score for this model was 54%
- The precision score for high risk individuals was 1% with a recall score of 69%
- The precision score for low risk individuals was 100% with a recall score of 40%
- The overall average precision score was 99% with a recall score of 40%

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

- The balanced accuracy score for this model was 77%
- The precision score for high risk individuals was 3% with a recall score of 66%
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


### Summary

From the above analysis, models 1 through 4 had accuracy scores of between 54% and 66%. The accuracy of Naive Random Oversampling, Oversampling, Undersampling, and Combinatorial models were much lower than models 5 and 6. The accuracy scores for Balanced Random Forest Classifier model was 77% and the Easy Ensemble AdaBoost Classifier model was the highest at 92%.

The reason we ran different models is because we want to see which one would have the highest best accuracy and precision in predicting the risk of loan applicants. Since we are looking at different models to assess credit risk, it is important to look specifically at the high_risk variable. We want high precision scores because precision is a measure of how reliable a positive classification is, and we want high sensitivity scores because sensitivity looks at how many loans that actually are high risk were correctly labeled as is. 

As a result, I recommend the Easy Ensemble AdaBoost Classifier for predicting credit risk for applicants. The model had an accuracy score of 92%, the highest of all six models tested. The precision scores were also much higher at 9% and 94%, respectively. These higher scores contributed to a higher overall recall mark for the model as well at 94%.
