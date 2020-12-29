#!/usr/bin/env python
# coding: utf-8

# In[131]:


from __future__ import print_function

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
import random

from time import time
import sys
import re
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn import metrics
from pyspark.sql import SQLContext
import nltk
from nltk.corpus import stopwords

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


# In[103]:


stopwords_set = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be']




# In[104]:


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False


# In[105]:


def buildArray(listOfIndices):
    returnVal = np.zeros(f)

    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1

    mysum = np.sum(returnVal)

    returnVal = np.divide(returnVal, mysum)

    return returnVal


# In[108]:


def predict1(x):
    return int(model1.predict(x))


# In[109]:


sc.stop()


# In[110]:


start = time()

f = 20000
sc = SparkContext(appName="task0")
# Pages = sc.textFile('winemag-data-130k-v2.csv')
spark = SQLContext(sc)


# In[111]:


def tranfer_label(x):
    if x < 88:
        return 0
    else:
        return 1


# In[112]:



df= spark.read.csv(sys.argv[1],header=True)

dfs = pd.DataFrame(df.toPandas())
dfs.dropna(subset=['_c0','description', 'designation','points','region_1','variety'])

data_values=dfs[['_c0','description', 'designation','points','region_1','variety']].values.tolist()
data_coulumns=['_c0','description', 'designation','points','region_1','variety']

#将pandas.DataFrame转为spark.dataFrame，需要转数据和列名
df_spark = spark.createDataFrame(data_values,data_coulumns)

df_rdd = df_spark.rdd

regex = re.compile('[^a-zA-Z]')

# remove all non letter characters
keyAndListOfWords1 = df_rdd.filter(lambda x :is_number(x[0]) is True and x[5] is not None and x[3] is not None
                                  and is_number(x[3]) is True)

keyAndListOfWords2 = keyAndListOfWords1.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split(),x[2],x[3], x[4],x[5]))

keyAndListOfWords = keyAndListOfWords2.map(lambda x : (x[0],[word for word in x[1] if word not in stopwords_set],x[2],
                                                       tranfer_label(float(x[3])),x[4],x[5]))

numberOfDocs = keyAndListOfWords.count()

allWords = keyAndListOfWords.flatMap(lambda x: ((i,1) for i in x[1]))

allCounts = allWords.reduceByKey(lambda x,y: x+y)
allCounts.collect()
topWords = allCounts.top(f,key = lambda x: x[1])

topWordsK = sc.parallelize(range(f))

dictionary = topWordsK.map (lambda x : (topWords[x][0], x))
print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ", dictionary.top(20, lambda x : x[1]))

dictionary.cache()


# In[113]:



allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

allDictionaryWords = dictionary.join(allWordsWithDocID)

justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))

allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()

allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))



zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0], np.where(x[1] > 0, 1, 0)))
dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]

multiplier = np.full(f, numberOfDocs)


idfArray = np.log(np.divide(np.full(f, numberOfDocs), dfArray + 1))

allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))

allDocsAsNumpyArraysTFidf


# In[114]:


labe_rdd = keyAndListOfWords.map(lambda x:(x[0],x[3]))

a = allDocsAsNumpyArraysTFidf.join(labe_rdd)
a.take(1)


# In[118]:



b = a.map(lambda x:(x[1][1],x[1][0],random.randint(1,3)))

train = b.filter(lambda x: x[2] == 1 or x[2] == 2).map(lambda x: LabeledPoint(x[0],x[1]))


stop = time()
time1 = stop - start
print('reading the data costs '+str(time1) + "seconds")


# In[119]:


start = time()

model = LogisticRegressionWithLBFGS.train(train,iterations=300,numClasses = 2)
stop = time()
time2 = stop - start
print('training the data costs '+str(time2) + "seconds")


# In[133]:


start = time()

model1 = LogisticRegressionWithSGD.train(train,iterations=300)
stop = time()
time2 = stop - start
print('training the data costs '+str(time2) + "seconds")


# In[ ]:


start = time()

model1 = SVMWithSGD.train(train,iterations = 200)
stop = time()
time2 = stop - start
print('training the data costs '+str(time2) + "seconds")


# In[121]:



start = time()

model2 = NaiveBayes.train(train)
stop = time()
time2 = stop - start
print('training the data costs '+str(time2) + "seconds")


# In[137]:


start = time()

def predict(x):
    return model.predict(x)

test = b.filter(lambda x: x[2] == 3)

result = test.map(lambda x: (x[0],model.predict(x[1]))).collect()
y=[]
prediction =[]
for i in result:
    y.append(i[0])
    prediction.append(i[1])
# result_rf = test_rf.map(lambda x: (x[0],int(model1.predict(x[1]))))
# result_rf.take(1)
# # In[63]:


f1 = f1_score(y, prediction, average='weighted')
a = confusion_matrix(y, prediction)
print(metrics.classification_report(y,prediction,target_names = ['negative','positive']))
print(f1)
print(a)

stop = time()
time3 = stop - start
print('testing the data costs '+str(time3) + "seconds")


# In[ ]:


start = time()

def predict(x):
    return model1.predict(x)

test = b.filter(lambda x: x[2] == 3)

result = test.map(lambda x: (x[0],model1.predict(x[1]))).collect()
y=[]
prediction =[]
for i in result:
    y.append(i[0])
    prediction.append(i[1])
# result_rf = test_rf.map(lambda x: (x[0],int(model1.predict(x[1]))))
# result_rf.take(1)
# # In[63]:


f1 = f1_score(y, prediction, average='weighted')
a = confusion_matrix(y, prediction)
print(metrics.classification_report(y,prediction,target_names = ['negative','positive']))

print(f1)
print(a)

stop = time()
time3 = stop - start
print('testing the data costs '+str(time3) + "seconds")


# In[138]:


start = time()

def predict(x):
    return model2.predict(x)

test = b.filter(lambda x: x[2] == 3)

result = test.map(lambda x: (x[0],model2.predict(x[1]))).collect()
y=[]
prediction =[]
for i in result:
    y.append(i[0])
    prediction.append(i[1])
# result_rf = test_rf.map(lambda x: (x[0],int(model1.predict(x[1]))))
# result_rf.take(1)
# # In[63]:


f1 = f1_score(y, prediction, average='weighted')
a = confusion_matrix(y, prediction)
print(metrics.classification_report(y,prediction,target_names = ['negative','positive']))

print(f1)
print(a)

stop = time()
time3 = stop - start
print('testing the data costs '+str(time3) + "seconds")


# In[99]:


sc.stop()


# In[ ]:




