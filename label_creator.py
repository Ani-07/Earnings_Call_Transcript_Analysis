# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:13:41 2020

@author: Anirudh Raghavan
"""
# Objective - Create labels for the target variables - stock price

# We use yahoo finance to download stock prices based on the dates of the Earnings
# Call

# We compute the return from 2 days post the Earnings Call 

# In some cases, yahoo finance may not have the prices available for certain stocks
# for certain dates

# These tickers are tracked in the rows_tbr list and we then remove these rows from
# our merged word frequencies table as well

#############################################################################

import os
import pandas as pd

from pandas_datareader import data
from datetime import datetime,timedelta
from time import sleep

def label_creator(x):
    if x > 0.02:
        label = 1
    elif x < -0.02:
        label = -1
    else:
        label = 0
    
    return label



file_source = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_2_Sentiment Analysis"
data_source = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_2_Sentiment Analysis\Source"
output_loc = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_2_Sentiment Analysis\processed_data"


os.chdir(file_source)

target = pd.read_csv("file_train.csv")

output = []

rows_tbr = []

for i in range(target.shape[0]):
    ticker = target.iloc[i,2]
    
    start = target.iloc[i,1]
    
    start = datetime.strptime(start, '%m/%d/%Y')
    start = start.date()
                 
    
    days_before = (start+timedelta(days=10)).isoformat()
    end = datetime.date(datetime.strptime(days_before, '%Y-%m-%d'))
    
    try:
        price = data.DataReader(ticker, "yahoo", start, end).iloc[:,3]
        two_change = (price.iloc[2] - price.iloc[0])/price.iloc[0]
        sleep(0.25)
        output.append(two_change)
    
    
    except KeyError as err_msg:
        rows_tbr.append(i)

    
 


labels = [label_creator(change) for change in output]


os.chdir(output_loc)

with open("train_labels.txt", 'w') as file:
    for item in labels:
        file.write("%s\n" % item)

target["Return"] = output
target["Label"] = labels

target.to_csv("Train_Labels.csv", index = False)

##########################################################################

os.chdir(output_loc)

EC_merged = pd.read_csv("EC_merged.csv")

EC_filtered = EC_merged.drop(rows_tbr)

EC_filtered.to_csv("EC_filtered.csv", index = False)

#############################################################################

os.chdir(file_source)

target = pd.read_csv("file_test.csv")

output = []

rows_tbr = []

for i in range(target.shape[0]):
    ticker = target.iloc[i,2]
    
    start = target.iloc[i,1]
    
    start = datetime.strptime(start, '%m/%d/%Y')
    start = start.date()
                 
    
    days_before = (start+timedelta(days=10)).isoformat()
    end = datetime.date(datetime.strptime(days_before, '%Y-%m-%d'))
    
    try:
        price = data.DataReader(ticker, "yahoo", start, end).iloc[:,3]
        two_change = (price.iloc[2] - price.iloc[0])/price.iloc[0]
        sleep(0.25)
        output.append(two_change)
    
    
    except KeyError as err_msg:
        rows_tbr.append(i)


labels = [label_creator(change) for change in output]

os.chdir(output_loc)


with open("test_labels.txt", 'w') as file:
    for item in labels:
        file.write("%s\n" % item)


target["Return"] = output
target["Label"] = labels

target.to_csv("Test_Labels.csv", index = False)





