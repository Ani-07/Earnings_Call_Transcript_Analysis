# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 18:45:13 2020

@author: Anirudh Raghavan
"""

import pandas as pd
import numpy as np
import os


#from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

#############################################################################
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

######################################################################################

output_loc = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_2_Sentiment Analysis\processed_data"
os.chdir(output_loc)

file = open("train_labels.txt", 'r')
y_train = file.readlines()
y_train  = np.array(y_train)


file = open("test_labels.txt", 'r')
y_test = file.readlines()
y_test  = np.array(y_test)


X_train  = pd.read_csv("train_features.csv")
X_test  = pd.read_csv("test_features.csv")


column_names = ["Name","Trial 1","Trial 2","Trial 3","Trial 4","Trial 5","CV"]
cross_val_summary = pd.DataFrame(columns = column_names)

def cv_append(cv, name,column_names):
    row_dict = {}
    for i in range(len(column_names)):
    
        if i == 0:
            row_dict[column_names[i]] = name
        
        elif i == 6:
            row_dict[column_names[i]] = np.mean(cv)
        
        else:
            row_dict[column_names[i]] = cv[i-1]
    
    return row_dict





##############################################################################
# MODEL BUILDING
##############################################################################
## Model 1 - Gaussian Naive Bayes

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

# In order to measure the performance of our classifer, we shall use a cross
# validation performance score

gnb_cv = cross_val_score(gnb, X_train, y_train, cv=5, scoring="accuracy")

gnb_cv_dict = cv_append(gnb_cv, "Naive Bayes",column_names)
    
cross_val_summary = cross_val_summary.append(gnb_cv_dict, ignore_index = True)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

# Build Confusion Matrix

nb_conf = confusion_matrix(y_test, y_pred)

###############################################################################

# Model 2 - Logistic Regression

# Fit Model

log_reg = LogisticRegression(random_state=0, max_iter = 2000)

log_reg.fit(X_train, y_train)

# Measure general accuracy of model with cross validation

log_cv = cross_val_score(log_reg, X_train, y_train, cv=5, scoring="accuracy")

log_cv_dict = cv_append(log_cv, "Logistic Regression", column_names)
    
cross_val_summary = cross_val_summary.append(log_cv_dict, ignore_index = True)

# Predict the response for test dataset
y_pred_log = log_reg.predict(X_test)


# Build Confusion Matrix
log_conf = confusion_matrix(y_test, y_pred_log)

##############################################################################

# Model 3 - Boosting

# Fit Model

ada_boost = AdaBoostClassifier(n_estimators=1000, random_state=0, 
                               algorithm='SAMME', learning_rate = 0.5)

ada_boost.fit(X_train, y_train)  

# Measure general accuracy of model with cross validation

ad_cv = cross_val_score(ada_boost, X_train, y_train, cv=5, scoring="accuracy")

ad_cv_dict = cv_append(ad_cv, "AdaBoost",column_names)
    
cross_val_summary = cross_val_summary.append(ad_cv_dict, ignore_index = True)


# Predict the response for test dataset

y_pred_ab = ada_boost.predict(X_test)

# Build Confusion Matrix

ab_conf = confusion_matrix(y_test, y_pred_ab)

###########################################################################

SVM_Linear = SVC(kernel = "linear")

SVM_Linear.fit(X_train, y_train)  

svml_cv = cross_val_score(SVM_Linear, X_train, y_train, cv=5, scoring="accuracy")

svml_cv_dict = cv_append(svml_cv, "Linear_SVM",column_names)
    
cross_val_summary = cross_val_summary.append(svml_cv_dict, ignore_index = True)

y_pred_svml = SVM_Linear.predict(X_test)

svml_conf = confusion_matrix(y_test, y_pred_svml)

###########################################################################

SVM_radial = SVC()

SVM_radial.fit(X_train, y_train)  

svmr_cv = cross_val_score(SVM_radial, X_train, y_train, cv=5, scoring="accuracy")

svmr_cv_dict = cv_append(svmr_cv, "Radial_SVM",column_names)
    
cross_val_summary = cross_val_summary.append(svmr_cv_dict, ignore_index = True)

y_pred_svmr = SVM_radial.predict(X_test)

svmr_conf = confusion_matrix(y_test, y_pred_svmr)


#############################################################################

# Comparison of models

# We shall use the following methods to compare the 3 models:
#  1) Precision Rate of predicting class 1 and-1
#  2) Utility function

#############################################################################

# We use precision rate for only class 1 and -1 because we shall make decisions
# only based on those classes and thus we measure the performance of how many
# times we get the classes right

print(classification_report(y_test, y_pred, digits=3))

print(classification_report(y_test, y_pred_ab, digits=3))

print(classification_report(y_test, y_pred_log, digits=3))

print(classification_report(y_test, y_pred_svml, digits=3))

print(classification_report(y_test, y_pred_svmr, digits=3))

#############################################################################

# Utlity Function

# We first create a utility matrix. The utility matrix shall depend on our 
# trading strategy

# We shall assume a trading stragey with stop loss and limit on gains as well

##############################################################################
# Prediction = Class 1

# If our prediction is Class 1, we shall invest money and in case the price 
# increases, we shall sell once the increase exceeds 2% as our prediction is 
# that price will inrease atleast upto 2%

# However, we shall also sell our share if price drops more than 1.5%

# Correct Prediction = +2
# Wrong PRediction (0 or -1) = -1.5

##############################################################################
# Prediction = Class 0

# In such a case, we would take no action and thus we would be left with
# opportunity loss

# Correct Prediction = 0
# Prediction of +1 or -1 = -2

##############################################################################

# Prediction = Class -1

# This would be similar to the first case

# Correct Prediction = +2
# Prediction of +1 or 0 = -1.5

#############################################################################

util_matrix = np.matrix([[2,-1.5,-1.5],[-2,0,-2],[-1.5,-1.5,2]])

print(np.sum(np.multiply(util_matrix, nb_conf)),np.dot(util_matrix[1],nb_conf[1]))

print(np.sum(np.multiply(util_matrix, log_conf)),np.dot(util_matrix[1],log_conf[1]))

print(np.sum(np.multiply(util_matrix, ab_conf)),np.dot(util_matrix[1],ab_conf[1]))

print(np.sum(np.multiply(util_matrix, svml_conf)),np.dot(util_matrix[1],svml_conf[1]))

print(np.sum(np.multiply(util_matrix, svmr_conf)),np.dot(util_matrix[1],svmr_conf[1]))


#column_names = ['1', '0', '-1']

#util_matrix = pd.DataFrame(util_matrix, columns=column_names, index=column_names)
