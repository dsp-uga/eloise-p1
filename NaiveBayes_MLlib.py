import pyspark
from pyspark.sql import SparkSession
import numpy as np
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes 
from pyspark.mllib.tree import RandomForest

# init spark
spark = SparkSession.builder.appName("P1_team").getOrCreate()       
sc=spark.sparkContext

# read X_train file
X_train_rdd=sc.textFile("gs://uga-dsp/project1/files/X_small_train.txt").map(lambda x: 'gs://uga-dsp/project1/data/bytes/'+x+'.bytes')
bytes = X_train_rdd.flatMap(lambda x: x.split())
# read class file
labels = sc.textFile("gs://uga-dsp/project1/files/y_small_train.txt").flatMap(lambda x: x.split())

# # removes "words" of len greater than 2 and "?"
def clean(x):
    temp = []
    for word in x:
        if (len(word)>2) or ("?" in word):
            pass
        else:
            temp += [word]
    return temp

# mapper = dict(zip(bytes.split(','),(y_).split(','))) what clint did before 
list_of_all_paths = bytes.collect() # only collects hash file names

# gather all byte files under one RDD; get rid on name values, only keep text; clean out byte text for just hexadecimals
all_text = sc.wholeTextFiles(','.join(list_of_all_paths)).map(lambda x: x[1]).map(lambda x: x.split()).map(lambda x: clean(x))

# encode text
hashingTF = HashingTF()
hash_values = hashingTF.transform(all_text)
idf = IDF().fit(hash_values)
tfidf = idf.transform(hash_values)

# Combine using zip workaround
temp_labels = labels.zipWithIndex().map(lambda x: (x[1], x[0]))
temp_tfidf = tfidf.zipWithIndex().map(lambda x: (x[1], x[0]))
training = temp_labels.leftOuterJoin(temp_tfidf)
raw_label_and_values = training.values()
raw_label_and_values = raw_label_and_values.map(lambda x: (x[0], x[1]))

# labeled point must be used for built in Naive Bayes
temp_labeled_points = raw_label_and_values.map(lambda x: (x[0],LabeledPoint(x[0], x[1])) )
labeled_points = temp_labeled_points.values()

# Train and check
model = NaiveBayes.train(labeled_points)
test = raw_label_and_values.collect()
correct = 0
total = 0
for pair in test:
    pred = int(model.predict(pair[1]))
    label = int(pair[0])
    if label == pred:
        correct += 1
    total += 1
    # print("Actual: "+str(label)+" Pred: "+str(pred))

print("Correct:")
print(correct)
print("Total:")
print(total)

spark.stop()
