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

output = pd.DataFrame(columns = ["prev_day_1","prev_day_2"])

rows_tbr = []

for i in range(target.shape[0]):
    
    ticker = target.iloc[i,2]
    
    start = target.iloc[i,1]
    
    start = datetime.strptime(start, '%m/%d/%Y')
    start = start.date()
                 
    
    days_before = (start-timedelta(days=10)).isoformat()
    end = datetime.date(datetime.strptime(days_before, '%Y-%m-%d'))
    
    try:
        price = data.DataReader(ticker, "yahoo", end, start).iloc[:,3]
        one_change = (price.iloc[-1] - price.iloc[-2])/price.iloc[-2]
        two_change = (price.iloc[-2] - price.iloc[-3])/price.iloc[-3]
        sleep(0.25)
        tmp = {"prev_day_1":one_change,"prev_day_2": two_change}
        output = output.append(tmp, ignore_index = True)
    
    
    except KeyError as err_msg:
        rows_tbr.append(i)
   



os.chdir(output_loc)

output.to_csv("base_train.csv", index = False)


#############################################################################

os.chdir(file_source)

target = pd.read_csv("file_test.csv")

output = pd.DataFrame(columns = ["prev_day_1","prev_day_2"])

rows_tbr = []

for i in range(target.shape[0]):
    
    ticker = target.iloc[i,2]
    
    start = target.iloc[i,1]
    
    start = datetime.strptime(start, '%m/%d/%Y')
    start = start.date()
                 
    
    days_before = (start-timedelta(days=10)).isoformat()
    end = datetime.date(datetime.strptime(days_before, '%Y-%m-%d'))
    
    try:
        price = data.DataReader(ticker, "yahoo", end, start).iloc[:,3]
        one_change = (price.iloc[-1] - price.iloc[-2])/price.iloc[-2]
        two_change = (price.iloc[-2] - price.iloc[-3])/price.iloc[-3]
        sleep(0.25)
        tmp = {"prev_day_1":one_change,"prev_day_2": two_change}
        output = output.append(tmp, ignore_index = True)
    
    
    except KeyError as err_msg:
        rows_tbr.append(i)



os.chdir(output_loc)

output.to_csv("base_test.csv", index = False)





