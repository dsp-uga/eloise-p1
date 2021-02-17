import pyspark
from pyspark.sql import SparkSession
import numpy as np
from operator import add
from __future__ import division
import math
import p1_utils
from absl import app, flags
from absl.flags import FLAGS
from string import punctuation

flags.DEFINE_string('X_train', './X_train.txt', 'path to trainX byte file(s)')
flags.DEFINE_string('y_train', './y_train.txt', 'path to trainY file(s)')
flags.DEFINE_string('X_test', './X_test.txt', 'path to testX byte file(s)')
flags.DEFINE_string('y_test', '0', 'path to testY byte file(s)') # we can still use it the same way if we replace this value with a y_test
flags.DEFINE_string('bucket', 'gs://bucket/', 'bucket for saving text file')
flags.DEFINE_boolean('bigram', False, 'use bigram')

def main(_argv):
    
    f= open(f'{FLAGS.bucket}output.txt","w+"') # init text
    # init spark
    spark = SparkSession.builder.appName("P1_team").getOrCreate()       
    sc=spark.sparkContext

    # --- step 1 --- #
    mapper = p1_utils.map_datasets2labels(sc, FLAGS.X_train, FLAGS.y_train)  
    # --- step 2 --- #
    if FLAGS.bigram:
        files_rdds_test, class_count_test, _VOID = p1_utils.generate_count_rdds_bigram(sc, mapper, trainset=True)
    else:
        files_rdds_test, class_count_test, _VOID = p1_utils.generate_count_rdds(sc, mapper, trainset=True)


    # files_rdds = dict of RDD's formatted key:'class', value: list of word count's for that class
    # word_perClass = dict key:class, val:total word count
    # class_count = dict key:class val: number of files of that class
    # --- step 3 --- #
    total_count, total_count_map = p1_utils.total_train_info(files_rdds)
    # --- step 4 --- #
    train_prob = {}                           
    for k in files_rdds.keys():
        current_word_perClass = word_perClass[k] # total wordcount for class[k]
        probs = {} # dict formatted key: class val: (dict key:word val: P(xi|yk))
        for word, count in files_rdds[k].collectAsMap().items():
            word, prob = p1_utils.P_xi_given_yk(word,count)
            probs[word] = prob
        train_prob[k] = probs

        
    # load test data
    # --- step 1B --- #
    mapper_test = p1_utils.map_datasets2labels(sc, FLAGS.X_test, FLAGS.y_test)
    # --- step 2B --- #
    if FLAGS.bigram:
        files_rdds_test, class_count_test, _VOID = p1_utils.generate_count_rdds_bigram(sc, mapper_test, trainset=False)
    else:
        files_rdds_test, class_count_test, _VOID = p1_utils.generate_count_rdds(sc, mapper_test, trainset=False)

    
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
            scores[k] = files_rdds_test[rdd_key].map(lambda x: (x, 1)).reduceByKey(add).map(lambda x: p1_utils.score_calc_fast(x[0], x[1])).reduce(lambda x, y: x +y)
            scores[k] = scores[k] + class_count[k]
        max_score = -100000000
        best_class = None
        print(scores)
        for label, score in scores.items():
            if score > max_score:
                max_score = score
                best_class = label
        print(best_class)
        f.write(str(best_class)+"/n") # write to text
        predictions[rdd_key] = best_class
        
      f.close()

      # not needed for auto lab
#     final_score = p1_utils.val_predict(predictions)
#     print(colored('Final Model Score: ', None), colored(str(final_score*100), 'red')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
