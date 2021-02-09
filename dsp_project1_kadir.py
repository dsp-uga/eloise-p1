#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:37:57 2021

@author: kadir
"""
from pyspark import SparkContext, SparkConf
import json, sys, os, math
from google.cloud import storage

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
    y_filtered_x = doc_labels.filter(lambda x: x[0] == str(1)).map(lambda x: x[1]).collect()
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
# calculate prior probabilities P(y_k)
def prior_prob(doc_label_pair, doc_class_no):
    p_prob = doc_label_pair.filter(lambda x: x[0] == str(doc_class_no)).map(lambda x: x[1]).count()/doc_label_pair.count()
    return p_prob

prior_prob(doc_labels, 1) 

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

# calculate cond probabilities P(x_i, y_k) including laplace smoothing and log transformation
def cond_prob(class_filtered, unfiltered_train):
    probs = term_freq(class_filtered, comp_vocabulary).join(term_freq(unfiltered_train, comp_vocabulary)).map(lambda x: (x[0], math.log10( (x[1][0] + 1)/x[1][1] )))
    return probs

# loop to get probability information for all classes

cond_probs_1 = cond_prob(class1_docs,all_train).sortBy(lambda x: -x[1])
cond_probs_1.take(5)



################ PREDICTION ######################
##################################################

