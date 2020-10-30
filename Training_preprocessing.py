# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 10:22:38 2020

@author: Anirudh Raghavan
"""
# Objective

# We process our list of documents to create a word frequencies table. We shall maintain 
# a limit on the size of the dataframe in memory in order to ensure faster processing 

# Therefore, our output shall periodically be written to file and a new dataframe shall be
# created for further files. 


# It is key to note that we are creating a word frequencies table, which means there is an 
# increasing vocab size with each new file processed. Thus, we would need to keep trach of 
# this increasing vocab as well. 

# We shall track the vocab through column names and thus ensure that each new file carries 
# over the vocab

############################################################################################

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk import tag
from nltk.util import ngrams
import pandas as pd
import PyPDF2 
import os

##########################################################################################

# We first create the functions neccessary to process the documents

# Text_reader - Thus function converts the pdf document into a string and if there is no
# text, then it returns "No Text"

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
    

##############################################################################

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

# pos_filter - Function to Filter Proper Nouns from the tokens

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

############################################################################### 

# We now combine all the above functions into a single text processing function

def text_prep(text):

    text = text.replace("\n"," ")
    text = text.replace("'","")
    
    word_tokens = word_tokenize(text) 
    
    tokens = sw_filter(word_tokens)
    
    alpha_tokens = number_filter(tokens)
        
    pos_words = pos_filter(alpha_tokens)
    
    #final = n_gram_create(pos_words)
    
    return pos_words
    
##############################################################################

# Our first step was to take in a text document and break it down into tokens

# We shall now move to creating word frequencies. We create a function which
# computes the occurrances of words from a words_list (vocab) in the given set 
# of tokens


def count_vec(word_list, tokens):

    count_vec = {}
    for word in word_list:
        count_vec[word] = tokens.count(word)
    
    return count_vec


###########################################################################

# Process the data

# We insert the following checks while processing the data:

# If pdf conversion outputs "No Text" we move to the next document

# If text processing outputs tokens less than 100 we move to the next document
# as Earnings Call Transcripts generally have more than 2000 words and this document
# may be a summary

# We track the above documents in the errors list in order to update our file list
# This, will be useful while downloading the stock prices later as we have to track
# which transcripts have been processed
 
#####################################################################################

# Open list of filenames

file_source = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_2_Sentiment Analysis"

os.chdir(file_source)

file = pd.read_csv("file_train.csv")
file_names = file["Name"]

file_count = 1

#Open the relevant matrix file
matrix_name = "EC_Count_Matrix_" + str(file_count) + ".csv"

EC_Count = pd.DataFrame()

data_source = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_2_Sentiment Analysis\Source"
output_loc = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_2_Sentiment Analysis\processed_data"

os.chdir(data_source)

errors = []

for j in range(file_names.shape[0]):
    
    name = file_names[j] + ".pdf"
    name = name.replace(":", "_")
    name = name.replace("/", "_")
    print(j,name)
    os.chdir(data_source)
    pdfFileObj = open(name,'rb')
    
    
    text = text_reader(pdfFileObj)
    
    if text == "No Text":
        print("No Text", name)
        errors.append(name)
        continue
    
    tokens = text_prep(text)
    
    if(len(tokens) <= 100):
        print("No Text", name)
        errors.append(j)
        continue
    
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
        print(file_count)
        EC_Count = pd.DataFrame(columns = EC_Count.columns)
        
os.chdir(output_loc)

EC_Count.to_csv(matrix_name, index = False)

#new = file.drop(errors)
#new.to_csv("Train_Updated.csv", index = False)

###################################################################################












