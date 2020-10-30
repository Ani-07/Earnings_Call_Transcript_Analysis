# Earnings_Call_Transcript_Analysis

Project Flow

Test_Train_Stratified.py - We first download a list of Earnings Call Transcripts from Nexus Uni and we use stratified sampling to split the documents into train and test

Training and Testing Preprocessing - These scripts help in preprocessing the documents into word count matrices. Since, the vocab increases with each additional document processed, to reduce chances of memory overload we save the data as and when the dataframe increases over a certain size

data_merging.Py -  This script is used to merge the saved word count matrices with the entire vocab

Modelling - Here, we clean the word count matrix and manually convert it into TF-IDF matrix

label_creator - We separately use yahoo reader to download stock prices around the Call dates and create labels for training the models

Model_Building - We build models and measure the performance using utility functions

baseline_model & baseline_data - These scripts are used to prepare the baseline model that we compare performance with,
