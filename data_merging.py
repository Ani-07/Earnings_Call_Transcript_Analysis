# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 08:51:32 2020

@author: Anirudh Raghavan
"""
# Objective

# Merge the different word frequencies tables created to form one large word 
# frequencies table

# Key note - Each further table will have an additional set of words from the 
# later documents, thus we will need to update the word frequencies for the earlier 
# documents as 0 for those words 

##############################################################################

import os
import pandas as pd
import numpy as np


# Open each file and merge to the larger dataframe

output_loc = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_2_Sentiment Analysis\processed_data"
os.chdir(output_loc)
        
EC_merged = pd.DataFrame()

#################################################################################

file_count = 7

for i in range(1,file_count+1):
    matrix_name = "EC_Count_Matrix_" + str(i) + ".csv"
    EC_Count = pd.read_csv(matrix_name)

    EC_merged = EC_merged.append(EC_Count, ignore_index = True)
    
# Replace all nan values. These are generally caused by the merging

sum(EC_merged.isnull().sum())

EC_merged = EC_merged.fillna(0)

sum(EC_merged.isnull().sum())

#########################################################################

EC_merged.to_csv("EC_merged.csv", index = False)

