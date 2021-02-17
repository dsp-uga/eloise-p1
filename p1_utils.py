import pyspark
from pyspark.sql import SparkSession
import numpy as np
from operator import add
from __future__ import division
import math



# used for final calculation of scores
def val_predict(predictions):
    correct_predict_count, all_predict_count = 0, 0
    for pred_y, y_true in predictions.items():
        pred_y = pred_y.split()[1]
        if pred_y == y_true:
            correct_predict_count += 1
        all_predict_count += 1
    return correct_predict_count/all_predict_count



# --- step 1 --- #
# get dict key: byte file path, value: class label
def map_datasets2labels(sc, train_name, test_name):
    X_train_rdd=sc.textFile('gs://uga-dsp/project1/files/'+train_name).map(lambda x: 'gs://uga-dsp/project1/data/bytes/'+x+'.bytes')
    bytes = X_train_rdd.reduce(lambda x, y: x + "," + y)
    # read class file and make a dict of key=file_path value=malware_class_num
    y_ = sc.textFile('gs://uga-dsp/project1/files/'+test_name).reduce(lambda x, y: x + "," + y)
    mapper = dict(zip(bytes.split(','),(y_).split(',')))
    return mapper # mapper is a dict of key:filepath, value:class label


# --- step 2 --- #
# read byte file's count words per class type return a dict of key: class, value: list of word count's for that class
def rdd_fix(rdd):
    rdd = rdd.flatMap(lambda x: x.split()).filter(lambda x: clean(x))
#     rdd = rdd.map(lambda x: (x, 1))
    return rdd

def bigram(rdd): # input rdd file loaded
    print('g')
    rdd = rdd.map(lambda x: x[9:]).map(lambda line: line.strip().split(" ")).flatMap(lambda xs: (tuple(x) for x in zip(xs, xs[1:]))).map(lambda x: (str(x[0]) + ' ' + str(x[1])))
    return rdd

# removes "words" of len greater than 2 and "?"
def clean(x):
    if (len(x)>2) or ("?" in x):
        pass
    else:
        return x

# generates two diffent set, if you run trainset=true you get a dict of RDD's formatted key:'class', value: list of word count's for that class
# if trainset=false key:'path', value: words E.G. ("00", "BG" .... "01")
def generate_count_rdds(sc, mapper, trainset=True):
    files_rdds = {}
    class_count = {}
    word_perClass = {}
    for k, v in mapper.items():
        if (v in files_rdds.keys()) and (trainset):
            class_count[v] += 1            
            files_rdds[v]= rdd_fix(sc.textFile(k)).union(files_rdds[v])
        else:
            if trainset:
                class_count[v] = 1
                files_rdds[v]= rdd_fix(sc.textFile(k))
            else:
                class_count[v] = 1
                files_rdds[str(k)+' '+str(v)]= rdd_fix(sc.textFile(k))
            
    for k, v in files_rdds.items():
        if trainset:
            word_perClass[k] = files_rdds[k].count()
            files_rdds[k] = files_rdds[k].map(lambda x: (x, 1)).reduceByKey(add) 
            
    return files_rdds, class_count, word_perClass


def generate_count_rdds_bigram(sc, mapper, trainset=True):
    files_rdds = {}
    class_count = {}
    word_perClass = {}
    for k, v in mapper.items():
        print(k)
        if (v in files_rdds.keys()) and (trainset):
            class_count[v] += 1            
            files_rdds[v]= bigram(sc.textFile(k)).union(files_rdds[v])
        else:
            if trainset:
                class_count[v] = 1
                files_rdds[v]= bigram(sc.textFile(k))
            else:
                class_count[v] = 1
                files_rdds[str(k)+' '+str(v)]= bigram(sc.textFile(k))   
    for k, v in files_rdds.items():
        print(k)
        if trainset:
            word_perClass[k] = files_rdds[k].count()
            files_rdds[k] = files_rdds[k].map(lambda x: (x, 1)).reduceByKey(add) 
            
    return files_rdds, class_count, word_perClass


# --- step 3 --- #
# get total count info
def total_train_info(files_rdds):
    # collectAsMap for total "word" counts
    rdd_names = list(files_rdds.keys())
    rdd1 = rdd_names.pop(-1)
    v2 = files_rdds[rdd1]
    for v in rdd_names:
        v2 = v2.union(files_rdds[v])# union keeps RDD form
    total_count_map=v2.reduceByKey(add).collectAsMap()
    return len(total_count_map), total_count_map # len(total_count_map) = count of all unique words = 256, total_count_map = dict 


# --- step 4 --- #
def P_xi_given_yk(word,count):
    # (("01" in class 1) + (1/vocab size)) / ((# words in class 1) + 1)
    prob = (float(count)+(1/float(total_count)))/(float(current_word_perClass))
    return word, prob



def score_calc_fast(word, count):
    if word in word2prob.keys():
        prob = float(math.log10(float(word2prob[word])))*count
    else:
        _, prob = P_xi_given_yk(word,0)
        prob = prob*count
    return prob
  