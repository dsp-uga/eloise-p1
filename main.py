import pyspark
from pyspark.sql import SparkSession
import numpy as np
from operator import add

# init spark
spark = SparkSession.builder.appName("P1_team").getOrCreate()       
sc=spark.sparkContext

def grouper(x0,x1):
    matched = []
    for x in x1.split():
        if (len(x) > 3) or ("?" in x):
            pass
        else:
            matched.append((x,1))
    return matched
    
# read X_train file
X_train_rdd=sc.textFile("gs://uga-dsp/project1/files/X_small_train.txt").map(lambda x: 'gs://uga-dsp/project1/data/bytes/'+x+'.bytes')
bytes = X_train_rdd.reduce(lambda x, y: x + "," + y)
# read class file and make a dict of key=file_path value=malware_class_num
y_ = sc.textFile("gs://uga-dsp/project1/files/y_small_train.txt").reduce(lambda x, y: x + "," + y)
mapper = dict(zip(bytes.split(','),(y_).split(',')))

# read byte file in the form bellow
def rdd_fix(rdd):
    rdd = rdd.flatMap(lambda x: x.split()).filter(lambda x: clean(x))
    rdd = rdd.map(lambda x: (x, 1)).reduceByKey(add)
    return rdd

# removes "words" of len greater than 2 and "?"
def clean(x):
    if (len(x)>2) or ("?" in x):
        pass
    else:
        return x
    
# unoins ddf of the same class    
# collects a dict keys=malware_class val = byte_file_name
files_rdds = {}
for k, v in mapper.items():
    files_rdds[v] = rdd_fix(sc.textFile(k)).union(files_rdds[v]).reduceByKey(add) if v in files_rdds.keys() else rdd_fix(sc.textFile(k))

# collectAsMap for total "word" counts
rdd_names = list(files_rdds.keys())
rdd1 = list(files_rdds.keys()).pop(-1)
v2 = files_rdds[rdd1]
for v in files_rdds.keys():
    v2 = v2.union(files_rdds[v])# union keeps RDD form
total_count_map=v2.reduceByKey(add).collectAsMap()

# P(xi/yk)
def P_xi_given_yk(word,count,):
    if word in total_count_map.keys():
        count = float(count)/float((total_count_map[word]))
    else:
        count = float(count)*-1
    return word,count


# returns P(xi/yk) for each class
for k, v in files_rdds.items():
    files_rdds[k] = files_rdds[k].map(lambda x: P_xi_given_yk(x[0],x[1]))
