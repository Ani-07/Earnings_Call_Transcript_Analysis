# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:58:53 2020

@author: Anirudh Raghavan
"""

# Objective - Process data for applying Machine Learning Models

import os
import pandas as pd
import numpy as np

output_loc = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_2_Sentiment Analysis\processed_data"

#############################################################################

# Using this function, we will remove all words which occur only in 1 document
# as this does not provide as sufficient information to determine relative 
# importance of the words


def words_removal(EC_merged):

    word_total = np.sum(EC_merged, axis = 0)
        
    words_single = np.where(word_total == 1)[0]
    
    words_single  = list(words_single)
    
    cols = EC_merged.columns[words_single]
    
    return cols


################################################################################

# Now, that we have vocabulary ready. We shall use this to convert the word frequencies into
# TF-IDF as this gives us a much better representation of the relative importance of the word


def word_freq(w):
    tmp = sum([1 if i > 0 else 0 for i in w])
    return tmp


def idf_comp(EC_merged):

    word_total = np.apply_along_axis(word_freq, 0, EC_merged)
    
    word_total = EC_merged.shape[0]/(word_total+1)
    
    word_total = np.log(word_total)
    
    return word_total


def tf_idf_fit(EC_merged):

    doc_total = np.sum(EC_merged, axis = 1)
    
    word_total = idf_comp(EC_merged)
    
    EC_merged = EC_merged.T/doc_total
    
    EC_merged = EC_merged.T*word_total
    
    return EC_merged, word_total



def tf_idf_transform(test,word_total):

    doc_total = np.sum(test, axis = 1)
    
    test = test.T/doc_total
    
    test = test.T*word_total
    
    return test

#########################################################################################

# Now lets use the above functions to obtain the features for our training data

os.chdir(output_loc)

EC_filtered = pd.read_csv("EC_filtered.csv")

raw_test = pd.read_csv("Test_Matrix.csv") 

####################################################################

raw_test = raw_test.drop(words_removal(EC_filtered), axis = 1)

EC_filtered = EC_filtered.drop(words_removal(EC_filtered), axis = 1)

##############################################################################

EC_tf = tf_idf_fit(EC_filtered)

EC_converted = EC_tf[0]

word_total = EC_tf[1]

# Now lets use the above functions to obtain the features for our test data
test_converted = tf_idf_transform(raw_test,word_total)


EC_converted.to_csv("train_features.csv", index = False)

test_converted.to_csv("test_features.csv", index = False)

