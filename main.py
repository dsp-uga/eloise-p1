import pyspark
from pyspark.sql import SparkSession
import numpy as np
from operator import add
from __future__ import division

"""
Current method!!!!!!!!!!
"""

# init spark
spark = SparkSession.builder.appName("P1_team").getOrCreate()       
sc=spark.sparkContext


# --- step 1 --- #
# get dict key: byte file path, value: class label
def map_datasets2labels(train_name, test_name):
    X_train_rdd=sc.textFile('gs://uga-dsp/project1/files/'+train_name).map(lambda x: 'gs://uga-dsp/project1/data/bytes/'+x+'.bytes')
    bytes = X_train_rdd.reduce(lambda x, y: x + "," + y)
    # read class file and make a dict of key=file_path value=malware_class_num
    y_ = sc.textFile('gs://uga-dsp/project1/files/'+test_name).reduce(lambda x, y: x + "," + y)
    mapper = dict(zip(bytes.split(','),(y_).split(',')))
    return mapper


# --- step 2 --- #
# read byte file's count words per class type return a dict of key: class, value: list of word count's for that class
def rdd_fix(rdd):
    rdd = rdd.flatMap(lambda x: x.split()).filter(lambda x: clean(x))
#     rdd = rdd.map(lambda x: (x, 1))
    return rdd
# removes "words" of len greater than 2 and "?"
def clean(x):
    if (len(x)>2) or ("?" in x):
        pass
    else:
        return x

def generate_count_rdds(mapper, trainset=True):
    files_rdds = {}
    class_count = {}
    word_perClass = {}
    for k, v in mapper.items():
        if (v in files_rdds.keys()) and (trainset):
            class_count[v] += 1            
            files_rdds[v]= rdd_fix(sc.textFile(k)).union(files_rdds[v])
        else:
            class_count[v] = 1
            files_rdds[v]= rdd_fix(sc.textFile(k))
            
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
    return len(total_count_map), total_count_map



# --- step 4 --- #
# P(xi/yk)
# oo = [(number of oo in A+B) + 1 ] / [(number of all words in A+B) + no of unique words in A+B+C+D+E]
def P_xi_given_yk(word,count):
    if word in total_count_map.keys():
        count = (float(count)+float(1))/(float(current_word_perClass)+float(total_count))
    else:
        count = (float(1))/(float(current_word_perClass)+float(total_count))
    return word, count


# --- step 1 --- #
mapper = map_datasets2labels('X_small_train.txt', 'y_small_train.txt')  
# --- step 2 --- #
files_rdds, class_count, word_perClass = generate_count_rdds(mapper)
# --- step 3 --- #
total_count, total_count_map = total_train_info(files_rdds)
# --- step 4 --- #
train_prob = {}                           
for k in files_rdds.keys():
    current_word_perClass = word_perClass[k]
    train_prob[k] = files_rdds[k].map(lambda x: P_xi_given_yk(x[0],x[1]))

    
    
    
    
    
# testing starts here
###############################################################################################
# load test data
# --- step 1 --- #
mapper_test = map_datasets2labels('X_small_test.txt', 'y_small_test.txt')  
# --- step 2 --- #
files_rdds_test, class_count_test, _VOID = generate_count_rdds(mapper_test, trainset=False)

def score_calc(word,count):
    count = float(word2prob[word])*float(count)
    return word, count
                           
for rdd_key in files_rdds_test.keys():
    scores = {}
    total_labels = 0
    for label, numclass in class_count.items():
        total_labels += numclass
    for k, v in class_count.items():
        print(v)
        print(total_labels)
        word2prob = train_prob[k].collectAsMap()
        # issues HERE cannont .reduceByKey(multiply) we need another way to do that
        scores[k] = files_rdds_test[rdd_key].map(lambda x: score_calc(x[0],x[1])).map(lambda x: float(x[1])).reduce(lambda x, y: x*y)
        p_yk = float(v/total_labels)
        scores[k] = scores[k]*p_yk
    
    

