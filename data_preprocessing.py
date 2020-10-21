# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 10:22:38 2020

@author: Anirudh Raghavan
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk import tag
from nltk.util import ngrams
import pandas as pd
import PyPDF2 

import os


# Stop_words_filter - Function to Filter Stop Words

def sw_filter(word_tokens):
    
    stop_words = set(stopwords.words('english')) 
     
    punctuation_List = [",",".","?","!",":",";","(",")","this","that","%","'","i","the","a","--","$","-"]
    
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    
    filtered_sentence = [w for w in filtered_sentence if not w.lower() in punctuation_List] 
    
    
    return filtered_sentence


# Number_filter - Remove numbers appearing as separate tokens

def number_filter(tokens):
    alpha_tokens = []
    for token in tokens:
        try:
            tmp = float(token)
            continue
        except ValueError as err_msg:
            alpha_tokens.append(token)
    
    return alpha_tokens     

# pos_filter - Function to Filter Proper Nouns

def pos_filter(tokens):
    
    tags = tag.pos_tag(tokens)
    pos_words = [tag[0] for tag in tags if tag[1] != "NNP" and tag[1] != "NNPS"]
    
    pos_words = [w.lower() for w in pos_words] 
    
    return pos_words


# n_gram_create - Function to create n_grams

def n_gram_create(pos_words):

    n = [' '.join(grams) for grams in ngrams(pos_words, 2)]

    for i in n:
        pos_words.append(i)
        
    return pos_words

# Combine the 


def text_prep(text):

    text = text.replace("\n"," ")
    text = text.replace("'","")
    
    word_tokens = word_tokenize(text) 
    
    tokens = sw_filter(word_tokens)
    
    alpha_tokens = number_filter(tokens)
        
    pos_words = pos_filter(alpha_tokens)
    
    final = n_gram_create(pos_words)
    
    return final
    
##############################################################################

def text_reader(pdfFileObj):
    
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        
        #Discerning the number of pages will allow us to parse through all the pages.
        
        num_pages = pdfReader.numPages
        count = 0
        text = ''
        
        #The while loop will read each page.
        while count < num_pages:
            pageObj = pdfReader.getPage(count)
            count +=1
            text += pageObj.extractText()
        #This if statement exists to check if the above library returned words. It's done because PyPDF2 cannot read scanned files.
        if text != "":
           text = text
        #If the above returns as False, we run the OCR library textract to #convert scanned/image based PDF files into text.
        else:
           text = 'No Text'
        return text
    

#########################################################################


def count_vec(word_list, tokens):

    count_vec = {}
    for word in word_list:
        count_vec[word] = tokens.count(word)
    
    return count_vec


###########################################################################

# Open list of filenames

file_source = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_2_Sentiment Analysis"

os.chdir(file_source)

file = pd.read_csv("Tracker_Files.csv")
file_names = file["Name"]


# Open metadata tracker where we track transcript file name and matrix file name
tracker = pd.read_csv("Tracker.csv")
#i = tracker.iloc[-1,0]
file_count = tracker.iloc[-1,0]

#Open the relevant matrix file
matrix_name = "EC_Count_Matrix_" + str(file_count) + ".csv"
#EC_Count = pd.read_csv(matrix_name)
EC_Count = pd.DataFrame()

data_source = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_2_Sentiment Analysis\Source"
output_loc = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_2_Sentiment Analysis\processed_data"


os.chdir(data_source)


for j in range(file_names.shape[0]):
    
    name = file_names[j] + ".pdf"
    name = name.replace(":", "_")
    name = name.replace("/", "_")
    print(name)
    os.chdir(data_source)
    pdfFileObj = open(name,'rb')
    
    
    text = text_reader(pdfFileObj)
    
    if text == "No Text":
        print(name)
        continue
    
    tokens = text_prep(text)
    
    for token in tokens:
        if token not in EC_Count.columns:
            
            #token = "word"
            #EC_Vocab[token] = vocab_n
            #vocab_n = vocab_n+1
            
            n = EC_Count.shape[0]
            
            EC_Count[token] = [0]*n
    
    vec_file = count_vec(EC_Count.columns, tokens)
    
    EC_Count = EC_Count.append(vec_file, ignore_index = True)
    
    if EC_Count.shape[0]*EC_Count.shape[1] > 1250000:
        os.chdir(output_loc)
        EC_Count.to_csv(matrix_name, index = False)
        
        file_count = file_count + 1
        matrix_name = "EC_Count_Matrix_" + str(file_count) + ".csv"
        
        EC_Count = pd.DataFrame(columns = EC_Count.columns)
        

EC_Count.to_csv(matrix_name, index = False)

track_dict = {"Matrix":file_count}      

tracker = tracker.append(track_dict, ignore_index = True)

tracker.to_csv("Tracker.csv", index = False)

EC_Count.columns     
    
