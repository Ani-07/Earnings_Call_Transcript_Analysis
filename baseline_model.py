# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 18:45:13 2020

@author: Anirudh Raghavan
"""

import pandas as pd
import numpy as np
import os


#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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


X_train  = pd.read_csv("base_train.csv")
X_test  = pd.read_csv("base_test.csv")


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
###############################################################################

# Model 2 - Logistic Regression

# Fit Model

log_reg = LogisticRegression(random_state=5, penalty = "none")

log_reg.fit(X_train, y_train)

# Measure general accuracy of model with cross validation

log_cv = cross_val_score(log_reg, X_train, y_train, cv=5, scoring="accuracy")

log_cv_dict = cv_append(log_cv, "Logistic Regression", column_names)
    
cross_val_summary = cross_val_summary.append(log_cv_dict, ignore_index = True)

# Predict the response for test dataset
y_pred_log = log_reg.predict(X_test)

pd.DataFrame(y_test)[0].value_counts()
pd.DataFrame(y_pred_log)[0].value_counts()

# Build Confusion Matrix
log_conf = confusion_matrix(y_test, y_pred_log)

pd.crosstab(y_test, y_pred_log)

log_reg.coef_

#log_reg.get_params



##############################################################################

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

print(np.sum(np.multiply(util_matrix, log_conf)),np.dot(util_matrix[1],log_conf[1]))

#column_names = ['1', '0', '-1']

#util_matrix = pd.DataFrame(util_matrix, columns=column_names, index=column_names)
