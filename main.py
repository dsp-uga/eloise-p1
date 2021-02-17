import pyspark
from pyspark.sql import SparkSession
import numpy as np
from operator import add
from __future__ import division
import math

"""
Current method!!!!!!!!!!
"""

# init spark
spark = SparkSession.builder.appName("P1_team").getOrCreate()       
sc=spark.sparkContext


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
def map_datasets2labels(train_name, test_name):
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
    rdd = rdd.map(lambda x: x[9:]).map(lambda line: line.strip().split(" ")) \
                    .flatMap(lambda xs: (tuple(x) for x in zip(xs, xs[1:]))) \
        .map(lambda x: (str(x[0]) + ' ' + str(x[1]))) # .distinct().map(lambda x: (x, 0))
    return rdd


# removes "words" of len greater than 2 and "?"
def clean(x):
    if (len(x)>2) or ("?" in x):
        pass
    else:
        return x

# generates two diffent set, if you run trainset=true you get a dict of RDD's formatted key:'class', value: list of word count's for that class
# if trainset=false key:'path', value: words E.G. ("00", "BG" .... "01")
def generate_count_rdds(mapper, trainset=True):
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


# --- step 3 --- #
# get total count info
def total_train_info(files_rdds):
    # collectAsMap for total "word" counts
    rdd_names = list(files_rdds.keys())
    rdd1 = list(files_rdds.keys()).pop(-1)
    v2 = files_rdds[rdd1]
    for v in rdd_names:
        v2 = v2.union(files_rdds[v])# union keeps RDD form
    total_count_map=v2.reduceByKey(add).collectAsMap()
    return len(total_count_map), total_count_map # len(total_count_map) = count of all unique words = 256, total_count_map = dict 



# --- step 4 --- #
def P_xi_given_yk(word,count):
    # (("01" in class 1) + (1/vocab size)) / ((# words in class 1) + 1)
    prod = (float(count)+(1/float(total_count)))/(float(current_word_perClass))
    return word, prob


# --- step 1 --- #
mapper = map_datasets2labels('X_small_train.txt', 'y_small_train.txt')  
# --- step 2 --- #
files_rdds, class_count, word_perClass = generate_count_rdds(mapper)
# files_rdds = dict of RDD's formatted key:'class', value: list of word count's for that class
# word_perClass = dict key:class, val:total word count
# class_count = dict key:class val: number of files of that class
# --- step 3 --- #
total_count, total_count_map = total_train_info(files_rdds)
# --- step 4 --- #
train_prob = {}                           
for k in files_rdds.keys():
    current_word_perClass = word_perClass[k] # total wordcount for class[k]
    print(current_word_perClass)
    probs = {} # dict formatted key: class val: (dict key:word val: P(xi|yk))
    for word, count in files_rdds[k].collectAsMap().items():
        word, prob = P_xi_given_yk(word,count)
        probs[word] = prob
    train_prob[k] = probs

    
    
    
    
    
# testing starts here
###############################################################################################
# load test data
# --- step 1 --- #
mapper_test = map_datasets2labels('X_small_test.txt', 'y_small_test.txt')  
# --- step 2 --- #
files_rdds_test, class_count_test, _VOID = generate_count_rdds(mapper_test, trainset=False)

def score_calc(word):
    if word in word2prob.keys():
        prob = float(math.log10(float(word2prob[word])))
    else:
        _ret, prob = P_xi_given_yk(word,0)
    return prob
  


total_labels = 0 
for label, numclass in class_count.items(): #class_count dict key: class val: number of that class in training set
    total_labels += numclass # total files in training set
for k in class_count.keys(): #class_count dict key: class val: number of that class in training set
    class_count[k] = math.log10(float(class_count[k]/total_labels)) # total files in training set

predictions = {}
for rdd_key in files_rdds_test.keys(): # test rdd's formatted key:'path', value: words E.G. ("00", "BG" .... "01")
    scores = {} # dict that will hold key: class, val: score for that class at a single test byte file
    for k, v in class_count.items():
        current_word_perClass = word_perClass[k]
        word2prob = train_prob[k]
        scores[k] = files_rdds_test[rdd_key].map(lambda x: score_calc(x)).reduce(lambda x, y: x +y)
        scores[k] = scores[k] + class_count[k]
    max_score = -100000000
    best_class = None
    print(scores)
    for label, score in scores.items():
        if score > max_score:
            max_score = score
            best_class = label
    print(best_class)
    predictions[rdd_key] = best_class
    
    

