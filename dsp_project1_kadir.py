#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:37:57 2021

@author: kadir
"""
from pyspark import SparkContext, SparkConf
import json, sys, os, math
from google.cloud import storage
import numpy as np

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/kadir/Documents/Spyder/DSP_Spring21/google_cloud_key.json'
# If you don't specify credentials when constructing the client, the
# client library will look for credentials in the environment.
client = storage.Client()
# Initialize Spark context
conf = SparkConf().setMaster("local[*]")
sc = SparkContext(conf = conf)

# # trying to get the files 
# # THIS SHOULD BE DONE BY PYSPARK TALKING DIRECTLY TO GOOGLE CLOUD BUT FOR NOW THIS IS TO TEST GOOGLE STORAGE
# hash_files = []
# data_files = []
# def get_gc_files():
#     # bucket = client.get_bucket('uga-dsp') # gives access error if you try to reach to bucket but next one works
#     for blob in client.list_blobs('uga-dsp', prefix='project1/files'):
#         hash_files.append(blob)
#         # print(str(blob))
#     for blob in client.list_blobs('uga-dsp', prefix='project1/data/bytes'):
#         data_files.append(blob)
    
# get_gc_files()

# # get X_small_train.txt
# # parallelized = sc.parallelize(hash_files[1].download_as_string().decode('utf-8').split())
# # 'project1/data/bytes/' + parallelized.first() + '.bytes'
# hash_files
# # x small train
# x_small_train = hash_files[1].download_as_string().decode('utf-8').split()
# # y small train
# y_small_train = hash_files[5].download_as_string().decode('utf-8').split()


# 'project1/data/bytes/' + x_small_train + '.bytes'
# data_files[1].download_as_string().decode('utf-8').split()[:50]

# hash_files[1].download_as_string().decode('utf-8').split()[1]

# [blob.name for blob in data_files if ('project1/data/bytes/' + x_small_train[1] + '.bytes') in blob.name]


################### PRACTICE ROUTINE ###############
####################################################

# Gather document names (x) and related classes (y) (LOCAL NOW FOR PRACTICE) 
# May need to broadcast this

def join_x_and_y(x_train_file, y_train_file):
    x_small_train = sc.textFile(x_train_file).zipWithIndex().map(lambda x: (x[1], x[0]))
    y_small_train = sc.textFile(y_train_file).zipWithIndex().map(lambda x: (x[1], x[0]))
    x_y_train = y_small_train.join(x_small_train).map(lambda x: x[1])
    return x_y_train

# label-doc(y-x) pairs
doc_labels = join_x_and_y('/home/kadir/Documents/Spyder/DSP_Spring21/project1/X_small_train.txt', 
              '/home/kadir/Documents/Spyder/DSP_Spring21/project1/y_small_train.txt' )#.filter(lambda x: x[0] == str(1)).collect()
# np.array(doc_labels).reshape(-1, 2)[:,1][1]

# filter x based on y (might be better with RDDs)
small_train_dir = "/home/kadir/Documents/Spyder/DSP_Spring21/project1/small_train/"
def filter_file_names(class_index):
    y_filtered_x = doc_labels.filter(lambda x: x[0] == str(class_index)).map(lambda x: x[1]).collect()
    addresses = ""
    for i in range(len(y_filtered_x)):    
           addresses = addresses + small_train_dir + y_filtered_x[i] +'.bytes,'
    addresses = addresses[:-1]
    return addresses 

# RDD containing class 1 documents
class1_docs = sc.textFile(filter_file_names(1))

# # class filtered documents RDD
# class1 = sc.textFile("/home/kadir/Documents/Spyder/DSP_Spring21/project1/testing/2F6ZfVCQRi3vrwcj4zxL.bytes,/home/kadir/Documents/Spyder/DSP_Spring21/project1/testing/JnHGRI2v5NuB9lpsEOCS.bytes")
# class1.count()

# class3 = sc.textFile("/home/kadir/Documents/Spyder/DSP_Spring21/project1/testing/5QpgRV2cqU9wvjBist1a.bytes")
# class3.count()

################ TRAINING ######################
################################################

### processing documents
# calculate prior probabilities P(y_k) with log transformation
def prior_prob(doc_label_pair, doc_class_no):
    p_prob = doc_label_pair.filter(lambda x: x[0] == str(doc_class_no)).map(lambda x: x[1]).count()/doc_label_pair.count()
    return math.log10(p_prob)

# all training documents RDD
all_train = sc.textFile("/home/kadir/Documents/Spyder/DSP_Spring21/project1/small_train/")

# all test documents RDD
all_test = sc.textFile("/home/kadir/Documents/Spyder/DSP_Spring21/project1/small_test/")

# comprehensive vocabulary to avoid zero probability issue by introducing words that does not appear in certain classes (from training and test data)
def vocabulary_of_zeros(comprehensive_training_data, comprehensive_test_data):
    voc_train = comprehensive_training_data.map(lambda x: x[9:]).flatMap(lambda x: x.strip().split()).distinct().map(lambda x: (x, 0))
    voc_test = comprehensive_test_data.map(lambda x: x[9:]).flatMap(lambda x: x.strip().split()).distinct().map(lambda x: (x, 0))
    return voc_train.union(voc_test)
    
# calculate term frequencies including zeros for words that appear in documents but not in every class
def term_freq(document_set, vocabulary):
    processed = document_set.map(lambda x: x[9:]).flatMap(lambda x: x.strip().split()).map(lambda x: (x, 1)).union(vocabulary).reduceByKey(lambda x, y: x + y)
    return processed

# term_freq(class1_docs, vocabulary_of_zeros(all_train, all_test)).sortBy(lambda x: x[1]).take(10)

comp_vocabulary = vocabulary_of_zeros(all_train, all_test)

no_unique_words = comp_vocabulary.count()
    
### ALTERNATING CONDITIONAL PROBABILITY CALCULATION: ADDED NO OF UNIQUE WORDS(WHOLE VOCABULARY) TO DENOMINATOR
### HOWEVER, NO OF WORD X IN ALL DOCUMENTS STAYS THE SAME (PER PROJECT MANUAL)
### MIGHT NEED TO CHANGE THAT TO NO OF ALL WORDS IN DOC Y AS IN GENERAL APPLICATIONS

# calculate cond probabilities P(x_i, y_k) including laplace smoothing and log transformation
# def cond_prob(class_filtered, unfiltered_train):
#     probs = term_freq(class_filtered, comp_vocabulary).join(term_freq(unfiltered_train, comp_vocabulary)).map(lambda x: (x[0], math.log10( (x[1][0] + 1)/(x[1][1] + no_unique_words) )))
#     return probs

def cond_prob(class_filtered, unfiltered_train):
    tf_class = term_freq(class_filtered, comp_vocabulary)
    total_words_in_class = tf_class.values().sum()
    # tf_all = term_freq(unfiltered_train, comp_vocabulary)
    # probs = tf_class.join(tf_all).map(lambda x: (x[0], math.log10( (x[1][0] + 1)/(x[1][1] + no_unique_words) )))
    probs = tf_class.map(lambda x: (x[0], math.log10( (x[1] + 1)/(total_words_in_class + no_unique_words) )))
    return probs

# tst = cond_prob(class1_docs, all_train)

# loop to get probability information for all classes (and potentially broadcast)
# get prior and conditional
prior_prob_list = []
cond_prob_list = []
for i in range(1,10):
    prior_prob_list.append(prior_prob(doc_labels, i))
    cond_prob_list.append(sc.broadcast(cond_prob(sc.textFile(filter_file_names(i)),all_train).collectAsMap()).value)
    # print(cond_prob_list[i-1].take(5))

# prior_prob(doc_labels, 1)
# bla = term_freq(class1_docs, comp_vocabulary)
# bla.values().sum()
# cond_probs_1 = cond_prob(class1_docs,all_train) #.sortBy(lambda x: -x[1])
# cond_probs_1.take(5)

################ PREDICTION ######################
##################################################

# get documents to be classified
x_test_list = sc.broadcast(sc.textFile('/home/kadir/Documents/Spyder/DSP_Spring21/project1/X_small_test.txt').collect()).value
y_test_list = sc.broadcast(sc.textFile('/home/kadir/Documents/Spyder/DSP_Spring21/project1/y_small_test.txt').collect()).value
doc_classification = sc.parallelize([])
for doc_i in range(len(x_test_list)):
    # class_scores = np.zeros((9,2))
    class_scores = []
    for class_i in range (1,10):
        filename = x_test_list[doc_i]
        document1 = sc.textFile(("/home/kadir/Documents/Spyder/DSP_Spring21/project1/small_test/" + filename + ".bytes")).map(lambda x: x[9:]).flatMap(lambda x: x.strip().split()).map(lambda x: (x, cond_prob_list[class_i-1][x]))
        class_scores.append([filename, class_i ,document1.values().sum() + prior_prob_list[class_i-1], y_test_list[doc_i]])
        # class_scores[class_i-1,:] = [class_i ,document1.values().sum() + prior_prob_list[class_i-1]]
    doc_classification = doc_classification.union(sc.parallelize([k for k in class_scores if k[2] == max(l[2] for l in class_scores)]))

doc_classification.take(10)

# class_scores[class_scores[:,1] == max(class_scores[:,1]),:]

################ ACCURACY CHECK###################
##################################################

doc_classification.map(lambda x: x[1] == int(x[3])).sum()/doc_classification.count()























