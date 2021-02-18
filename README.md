# eloise-p1

Kadir - Initial small test results were around 63% with main naive bayes with no improvements
dsp_project1_kadir_gc.py - takes 5 arguments:
1 - data dir
2 - x train file
3 - y train file
4 - x test file
5 - y test file (for accuracy, optional)

Introduction:
-----------------
The goal of this project was to develop at Malware Classification Algorithm.
This project was outlined to start from a simple Naive Bayes model, then increase complexity to obtain improved results.
Training Malware files came in both bytes and asm files.


Technologies Used:
-----------------
- Python 2.7
- Apache Spark

How to Implement The Models
------------------

This project was segmented into two main files:

p1_main.py and a helper file titled p1_utils.py


Examples of how to solve run a simple Naive Bayes model from the terminal are listed below:

```
python p1_main.py --X_train ./X_small_train.txt --y_train ./y_small_train.txt --X_test ./X_small_test.txt --y_test ./y_small_test.txt
```

To run this model on the larger set run the command line below: 

```
python p1_main.py
```

This code can also produce predictions from bigrams of "words" from the bytes files:

```
python p1_main.py --brigram True
```




The files are dynamic enough to allow you to specify the following information:


Authors
--------------

   [kadirbice](https://github.com/kbice)
   
   [Clint Morris](https://github.com/clint-kristopher-morris)
   
   [Gianni](https://github.com/Gianni2437)
