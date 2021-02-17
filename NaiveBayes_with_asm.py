from __future__ import division
import pyspark
from pyspark.sql import SparkSession
import numpy as np
from operator import add
import math

"""
Current method!!!!!!!!!!
"""

# init spark
spark = SparkSession.builder.appName("P1_team").getOrCreate()       
sc=spark.sparkContext



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
    X_train_rdd=sc.textFile('gs://uga-dsp/project1/files/'+train_name)
    bytes = X_train_rdd.reduce(lambda x, y: x + "," + y)
    # read class file and make a dict of key=file_path value=malware_class_num
    y_ = sc.textFile('gs://uga-dsp/project1/files/'+test_name).reduce(lambda x, y: x + "," + y)
    mapper = dict(zip(bytes.split(','),(y_).split(',')))
    return mapper


# --- step 2 --- #
# read byte file's count words per class type return a dict of key: class, value: list of word count's for that class
def rdd_fix(rdd, asm):
    rdd1 = rdd.flatMap(lambda x: x.split()).filter(lambda x: clean(x))
    ''' Two seperate methods for integrating asm files into NB. 
    1.) Filter for all 'Get' and 'Set' system calls. Similiar malware should make 
    similiar calls (i.e. GetKeyboardInfo, SetCalendar, etc.)
    2.) Filter for all assembly code memory shifts (i.e. [ebx+2] -> ebx+1') '''
    rdd2 = asm.flatMap(lambda x: x.split()).filter(lambda x: ('Get' in x) or ('Set' in x)).filter(lambda x: '(' in x).map(lambda x: x[:x.index("(")])
    # rdd2 = asm.flatMap(lambda x: x.split()).filter(lambda x: ('[' in x) and (']' in x)).map(lambda x: x[x.index("[")+1:x.index("]")])
    rdd1 = rdd1.zipWithIndex().map(lambda x: (x[1], x[0]))
    rdd2 = rdd2.zipWithIndex().map(lambda x: (x[1], x[0]))
    combinedRdd = rdd1.leftOuterJoin(rdd2)
    combinedRdd = combinedRdd.values()
    rdd = combinedRdd.map(lambda x: [x[0]].append([x[1]])) # combine hexadecimal and asm data into same vocabulary
#     rdd = rdd.map(lambda x: (x, 1))
    return rdd
# removes "words" of len greater than 2 and "?"
def clean(x):
    if (len(x)<2) or ("?" in x):
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
            files_rdds[v]= rdd_fix(sc.textFile('gs://uga-dsp/project1/data/bytes/'+k+'.bytes'), sc.textFile('gs://uga-dsp/project1/data/asm/'+k+'.asm')).union(files_rdds[v])
        else:
            if trainset:
                class_count[v] = 1
                files_rdds[v]= rdd_fix(sc.textFile('gs://uga-dsp/project1/data/bytes/'+k+'.bytes'), sc.textFile('gs://uga-dsp/project1/data/asm/'+k+'.asm'))
            else:
                class_count[v] = 1
                files_rdds[str(k)+' '+str(v)]= rdd_fix(sc.textFile('gs://uga-dsp/project1/data/bytes/'+k+'.bytes'), sc.textFile('gs://uga-dsp/project1/data/asm/'+k+'.asm'))
            
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
def P_xi_given_yk(word,count):
    count = (float(count)+(1/float(total_count)))/(float(current_word_perClass))
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
    print(current_word_perClass)
    probs = {}
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
        __ret, prob = P_xi_given_yk(word,0)
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
    print("Pred: "+str(best_class))
    print("Actual: "+str(rdd_key))
    predictions[rdd_key] = best_class

