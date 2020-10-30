# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:24:23 2020

@author: Anirudh Raghavan
"""
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split


file_source = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_2_Sentiment Analysis"
output_loc = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_2_Sentiment Analysis\processed_data"

os.chdir(file_source)


file = pd.read_csv("Tracker_Files.csv")


X_train, X_test = train_test_split(file, test_size=0.2, random_state=42, stratify = file["Ticker"])

Split = pd.DataFrame([file["Ticker"].value_counts(), X_test["Ticker"].value_counts(), X_train["Ticker"].value_counts()])
Split = Split.T

Split.columns = ["Total","Test","Training"]

train = list(X_train.index)
test = list(X_test.index)

os.chdir(output_loc)


with open("train.txt", 'w') as file:
    for item in train:
        file.write("%s\n" % item)

with open("test.txt", 'w') as file:
    for item in test:
        file.write("%s\n" % item)

os.chdir(file_source)


X_train.to_csv("file_train.csv", index = False)
X_test.to_csv("file_test.csv", index = False)


