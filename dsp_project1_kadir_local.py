#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:37:57 2021

@author: kadir
"""
from pyspark import SparkContext, SparkConf
import os, math, sys
# from google.cloud import storage

# GC credentials, I do not know if this is necessary when using from GC command line
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/kadir/Documents/Spyder/DSP_Spring21/google_cloud_key.json'

# If you don't specify credentials when constructing the client, the
# client library will look for credentials in the environment.
# client = storage.Client()

# Change working directory to scripts directory
# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

################### PRACTICE ROUTINE ###############
####################################################
# Trying to connect GC using commandline but we can do that with Clint's approach as well
input_data_path = "/home/kadir/Documents/Spyder/DSP_Spring21/project1/data/" #sys.argv[1]
input_xtrain_file = '/home/kadir/Documents/Spyder/DSP_Spring21/project1/X_small_train.txt' #sys.argv[2]
input_ytrain_file = '/home/kadir/Documents/Spyder/DSP_Spring21/project1/y_small_train.txt' #sys.argv[3]
input_xtest_file = '/home/kadir/Documents/Spyder/DSP_Spring21/project1/X_small_test.txt' #sys.argv[4]
input_output_dir =  '/home/kadir/Documents/Spyder/DSP_Spring21/DSP_Spring21_github/eloise-p1/'
input_ytest_file = '/home/kadir/Documents/Spyder/DSP_Spring21/project1/y_small_test.txt' #sys.argv[5]

# if len(sys.argv) < 6:
#   raise Exception("At least 5 arguments are required: <inputDataDir> <x_train_file> <y_train_file> <x_test_file> <output_dir> (optional) <y_test_file>")

# input_data_path = sys.argv[1] #"/home/kadir/Documents/Spyder/DSP_Spring21/project1/data/" #sys.argv[1]
# input_xtrain_file = sys.argv[2] #'/home/kadir/Documents/Spyder/DSP_Spring21/project1/X_small_train.txt' #sys.argv[2]
# input_ytrain_file = sys.argv[3] #'/home/kadir/Documents/Spyder/DSP_Spring21/project1/y_small_train.txt' #sys.argv[3]
# input_xtest_file = sys.argv[4] #'/home/kadir/Documents/Spyder/DSP_Spring21/project1/X_small_test.txt' #sys.argv[4]
# input_output_dir =  sys.argv[5] #'/home/kadir/Documents/Spyder/DSP_Spring21/DSP_Spring21_github/eloise-p1/'

# if len(sys.argv) == 7:
#     input_ytest_file = sys.argv[6] #'/home/kadir/Documents/Spyder/DSP_Spring21/project1/y_small_test.txt' #sys.argv[5]
# else:
#     input_ytest_file = ''
    

# Initialize Spark context
# conf = SparkConf().setMaster("local[*]")
# sc = SparkContext(conf = conf)
conf = SparkConf().setAppName('ilkproje')
sc = SparkContext(conf = conf)
print('Setting up spark context...')

# Gather document names (x) and related classes (y) (LOCAL NOW FOR PRACTICE) 
# May need to broadcast this
def join_x_and_y(x_train_file, y_train_file):
    x_small_train = sc.textFile(x_train_file).zipWithIndex().map(lambda x: (x[1], x[0]))
    y_small_train = sc.textFile(y_train_file).zipWithIndex().map(lambda x: (x[1], x[0]))
    x_y_train = y_small_train.join(x_small_train).map(lambda x: x[1])
    return x_y_train

# label-doc(y-x) pairs
doc_labels = join_x_and_y(input_xtrain_file, input_ytrain_file )#.filter(lambda x: x[0] == str(1)).collect()
print('Collecting training x and y data together...')

# filter x based on y (might be better with RDDs)
# small_train_dir = "/home/kadir/Documents/Spyder/DSP_Spring21/project1/small_train/"
def filter_file_names(class_index):
    y_filtered_x = doc_labels.filter(lambda x: x[0] in class_index).map(lambda x: x[1]).collect()
    return y_filtered_x

# get subset of data files by collecting addresses of each (there should be a smarter way)
def get_file_addresses(y_filtered_x):
    addresses = ""
    for i in range(len(y_filtered_x)):    
           addresses = addresses + input_data_path + y_filtered_x[i] +'.bytes,'
    addresses = addresses[:-1]
    return addresses 

# len(filter_file_names(['1']))
# doc_labels.filter(lambda x: x[0] in '1').map(lambda x: x[1]).collect()

################ TRAINING ######################
################################################

### processing documents
# calculate prior probabilities P(y_k) with log transformation
def prior_prob(doc_label_pair, doc_class_no):
    p_prob = doc_label_pair.filter(lambda x: x[0] == str(doc_class_no)).map(lambda x: x[1]).count()/doc_label_pair.count()
    return math.log10(p_prob)

# print(doc_labels.filter(lambda x: x[0] == str(1)).map(lambda x: x[1]).count())
# print(doc_labels.count())
# print('PRIOR PROB FOR CLASS = ', prior_prob(doc_labels, 1))

print('Forming comprehensive RDDs for training and testing...')
# all training documents RDD
all_train = sc.textFile(get_file_addresses(sc.textFile(input_xtrain_file).collect()))
# all test documents RDD
all_test = sc.textFile(get_file_addresses(sc.textFile(input_xtest_file).collect()))

############################ WORDS #########################################################
# comprehensive vocabulary to avoid zero probability issue by introducing words that does not appear in certain classes (from training and test data)
def vocabulary_of_zeros(comprehensive_training_data, comprehensive_test_data):
    voc_train = comprehensive_training_data.map(lambda x: x[9:]).flatMap(lambda x: x.strip().split()).distinct().map(lambda x: (x, 0))
    voc_test = comprehensive_test_data.map(lambda x: x[9:]).flatMap(lambda x: x.strip().split()).distinct().map(lambda x: (x, 0))
    return voc_train.union(voc_test)
    
# calculate term frequencies including zeros for words that appear in documents but not in every class
def term_freq(document_set, vocabulary):
    processed = document_set.map(lambda x: x[9:]).flatMap(lambda x: x.strip().split()) \
        .map(lambda x: (x, 1)).union(vocabulary).reduceByKey(lambda x, y: x + y)
    return processed

print('Gathering vocabularies and no of unique words...')
comp_vocabulary = vocabulary_of_zeros(all_train, all_test)

no_unique_words = comp_vocabulary.count()
print('Done')

# calculate cond probabilities P(x_i, y_k) including laplace smoothing and log transformation
# ALTERNATING CONDITIONAL PROBABILITY CALCULATION: ADDED NO OF UNIQUE WORDS(WHOLE VOCABULARY) TO DENOMINATOR
# HOWEVER, NO OF WORD X IN ALL DOCUMENTS STAYS THE SAME (PER PROJECT MANUAL)
# MIGHT NEED TO CHANGE THAT TO NO OF ALL WORDS IN DOC Y AS IN GENERAL APPLICATIONS

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

print('Putting together class-dependent conditional probs for each word...')
# loop to get probability information for all classes and pass to prediction
# get prior and conditional
prior_prob_list = []
cond_prob_list = []
for i in range(1,10):
    prior_prob_list.append(prior_prob(doc_labels, i))
    cond_prob_list.append(sc.broadcast(cond_prob(sc.textFile(get_file_addresses( filter_file_names(str(i)) )),all_train).collectAsMap()).value)

############################ BIGRAMS #########################################################
# def bigrams_of_zeros(comprehensive_training_data, comprehensive_test_data):
#     bigrams_train = comprehensive_training_data.map(lambda x: x[9:]).map(lambda line: line.strip().split(" ")) \
#                     .flatMap(lambda xs: (tuple(x) for x in zip(xs, xs[1:]))) \
#                         .map(lambda x: (str(x[0]) + ' ' + str(x[1]))).distinct().map(lambda x: (x, 0))
#     bigrams_test = comprehensive_test_data.map(lambda x: x[9:]).map(lambda line: line.strip().split(" ")) \
#                     .flatMap(lambda xs: (tuple(x) for x in zip(xs, xs[1:]))) \
#                         .map(lambda x: (str(x[0]) + ' ' + str(x[1]))).distinct().map(lambda x: (x, 0))
#     return bigrams_train.union(bigrams_test)

# def n_gram_term_freq(document_set, ngram_vocabulary):
#     # bla = sc.textFile(get_file_addresses( filter_file_names(str(1)) )).map(lambda x: x[9:])
#     bigrams = document_set.map(lambda line: line.strip().split(" ")) \
#                     .flatMap(lambda xs: (tuple(x) for x in zip(xs, xs[1:]))) \
#                         .map(lambda x: (str(x[0]) + ' ' + str(x[1]))).map(lambda x: (x, 1)) \
#                             .union(ngram_vocabulary).reduceByKey(lambda x,y: x + y)
#     return bigrams

# comp_bigrams = bigrams_of_zeros(all_train, all_test)

# no_unique_bigrams = comp_bigrams.count()
    
# # bigram1 = n_gram_term_freq(sc.textFile(get_file_addresses( filter_file_names(str(1)) )).map(lambda x: x[9:]), comp_bigrams)
# # bigram1.saveAsTextFile('/home/kadir/Documents/Spyder/DSP_Spring21/DSP_Spring21_github/eloise-p1/bigram')

# def ngram_cond_prob(class_filtered, unfiltered_train):
#     bigram_tf_class = n_gram_term_freq(class_filtered, comp_bigrams)
#     total_bigrams_in_class = bigram_tf_class.values().sum()
#     # tf_all = term_freq(unfiltered_train, comp_vocabulary)
#     # probs = tf_class.join(tf_all).map(lambda x: (x[0], math.log10( (x[1][0] + 1)/(x[1][1] + no_unique_words) )))
#     probs = bigram_tf_class.map(lambda x: (x[0], math.log10( (x[1] + 1)/(total_bigrams_in_class + no_unique_bigrams) )))
#     return probs

# # bigramcondprob1 = ngram_cond_prob(sc.textFile(get_file_addresses( filter_file_names(str(1)) )).map(lambda x: x[9:]),all_train)
# # bigramcondprob1.take(10)
# # bigramcondprob1.collectAsMap()
# # bigram1broad = sc.broadcast(bigramcondprob1.collectAsMap()).value

# prior_prob_list = []
# cond_prob_list = []
# for i in range(1,10):
#     prior_prob_list.append(prior_prob(doc_labels, i))
#     cond_prob_list.append(sc.broadcast(ngram_cond_prob(sc.textFile(get_file_addresses( filter_file_names(str(i)) )).map(lambda x: x[9:]),all_train).collectAsMap()).value)

################ PREDICTION ######################
##################################################

# get documents to be classified
x_test_list = sc.broadcast(sc.textFile(input_xtest_file).collect()).value
# get labels of each document for comparison (not needed for later)

############################ WORDS #########################################################
# predict classes (I attached labels(y) to the RDD as well for test purposes but we do not need it for later)
def predict_classes(list_of_files):
    doc_classification = sc.parallelize([])
    for doc_i in range(len(list_of_files)):
        class_scores = []
        for class_i in range (1,10):
            filename = list_of_files[doc_i]
            document1 = sc.textFile((input_data_path + filename + ".bytes")).map(lambda x: x[9:]).flatMap(lambda x: x.strip().split()).map(lambda x: (x, cond_prob_list[class_i-1][x]))
            class_scores.append([filename, class_i ,document1.values().sum() + prior_prob_list[class_i-1]])
        doc_classification = doc_classification.union(sc.parallelize([k for k in class_scores if k[2] == max(l[2] for l in class_scores)]))
    return doc_classification

doc_classification = predict_classes(x_test_list)

## document1 = sc.textFile((input_data_path + x_test_list[1] + ".bytes")).map(lambda x: x[9:]).map(lambda line: line.strip().split(" ")) \
##                     .flatMap(lambda xs: (tuple(x) for x in zip(xs, xs[1:]))) \
##                         .map(lambda x: (str(x[0]) + ' ' + str(x[1])))
## document1.take(10) 

############################ BIGRAMS #########################################################
# def predict_classes(list_of_files):
#     doc_classification = sc.parallelize([])
#     for doc_i in range(len(list_of_files)):
#         class_scores = []
#         for class_i in range (1,10):
#             filename = list_of_files[doc_i]
#             document1 = sc.textFile((input_data_path + filename + ".bytes")).map(lambda x: x[9:]).map(lambda line: line.strip().split(" ")) \
#                     .flatMap(lambda xs: (tuple(x) for x in zip(xs, xs[1:]))) \
#                         .map(lambda x: (str(x[0]) + ' ' + str(x[1]))).map(lambda x: (x, cond_prob_list[class_i-1][x]))
#             class_scores.append([filename, class_i ,document1.values().sum() + prior_prob_list[class_i-1]])
#         doc_classification = doc_classification.union(sc.parallelize([k for k in class_scores if k[2] == max(l[2] for l in class_scores)]))
#     return doc_classification

# doc_classification = predict_classes(x_test_list)
###############

# yler = sc.textFile('/home/kadir/Documents/Spyder/DSP_Spring21/project1/y_small_test.txt').zipWithIndex().map(lambda x: (x[1], x[0]))
# doc_classification.zipWithIndex().map(lambda x: (x[1], x[0])).join(yler).map(lambda x: x[1][0][1] == int(x[1][1])).sum()
# doc_classification.join(sc.textFile('/home/kadir/Documents/Spyder/DSP_Spring21/project1/y_small_test.txt').zipWithIndex()).take(10)

# doc_classification.take(10)

################ ACCURACY CHECK###################
##################################################
def accuracy_check(classifications, ytest_file):  
    y_test_list = sc.textFile(ytest_file).zipWithIndex().map(lambda x: (x[1], x[0]))
    compare_classes = classifications.zipWithIndex().map(lambda x: (x[1], x[0])).join(y_test_list)
    print('Accuracy of the classification: ' + str(format(100*compare_classes.map(lambda x: x[1][0][1] == int(x[1][1])).sum()/classifications.count(), '.2f')) + '%')

if(len(input_ytest_file) > 0):
    accuracy_check(doc_classification,input_ytest_file)
else:
    print('no y data given so accuracy is not calculated')


# # save as txt
# with open(input_output_dir + 'classification.txt', 'w') as txt_file:
#   txt_file.writelines("%s\n" % p for p in doc_classification.map(lambda x: x[1]).collect())
# print('output has been written to txt file')

doc_classification.map(lambda x: x[1]).saveAsTextFile(input_output_dir + 'output_large/')
















