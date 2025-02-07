# eloise-p1

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

NaiveBayes_from_Scratch2.py - takes 6 arguments:
1 - data dir
2 - x train file
3 - y train file
4 - x test file
5 - output dir
6 - y test file (for accuracy, optional)

Note: Code also includes commented out bigram implementation which works however, due to time limitations, we were not able to produce an output.

NaiveBayes_with_asm.py experiments with associating asm file features with the hexadecimals in the 'bag of words'. The first included methodologiy scrapes 'Get' and 'Set' method names used within the assembly script in order to associate specific actions like 'GetKeyboardInput' and 'SetCalendarInfo' with malware classes. The second method includes assembly memory actions (i.e. [esx+14H]) within the script with malware classes for ease of classification. 

This portion of the repository did not lead to improvement in accuracy and is under active development.

Issues
--------------
Builtin PySpark MLlib classification models such as Naive Bayes and Random Forest offered no significant improvement in accuracy over the "from scratch" Naive Bayes implementations.

Use of asm files leading to overprediction of specific malware classes such a 3 and 7.

Authors
--------------

   [kadirbice](https://github.com/kbice)
   
   [Clint Morris](https://github.com/clint-kristopher-morris)
   
   [Gianni](https://github.com/Gianni2437)
