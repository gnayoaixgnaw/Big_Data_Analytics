from __future__ import print_function

#!/usr/bin/env python
# coding: utf-8

# In[68]:


from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint

from time import time
import sys
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext
from pyspark.sql import functions as func
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# In[69]:


def buildArray(listOfIndices):
    returnVal = np.zeros(f)

    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1

    mysum = np.sum(returnVal)

    returnVal = np.divide(returnVal, mysum)

    return returnVal


# In[70]:


def get_tf_idf(textFile):
    wikiPages = sc.textFile(textFile)
    # wikiCategoryLinks=sc.textFile(sys.argv[2])

    # wikiCats=wikiCategoryLinks.map(lambda x: x.split(",")).map(lambda x: (x[0].replace('"', ''), x[1].replace('"', '') ))
    numberOfDocs = wikiPages.count()

    # Each entry in validLines will be a line from the text file
    validLines = wikiPages.filter(lambda x: 'id' in x and 'url=' in x)
    # Now, we transform it into a set of (docID, text) pairs
    keyAndText = validLines.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))
    regex = re.compile('[^a-zA-Z]')

    # remove all non letter characters
    keyAndListOfWords = keyAndText.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()))

    # Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
    # ("word1", docID), ("word2", docId), ...

    allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

    # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
    allDictionaryWords = dictionary.join(allWordsWithDocID)

    # Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
    justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))
    # Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
    # The following line this gets us a set of
    # (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    # and converts the dictionary positions to a bag-of-words numpy array...
    allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))

    # Now, create a version of allDocsAsNumpyArrays where, in the array,
    # every entry is either zero or one.
    # A zero means that the word does not occur,
    # and a one means that it does.

    zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0], np.where(x[1] > 0, 1, 0)))
    dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]
    # Create an array of 20,000 entries, each entry with the value numberOfDocs (number of docs)
    multiplier = np.full(f, numberOfDocs)

    # Get the version of dfArray where the i^th entry is the inverse-document frequency for the
    # i^th word in the corpus
    idfArray = np.log(np.divide(np.full(f, numberOfDocs), dfArray + 1))
    # Finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors
    allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))

    return allDocsAsNumpyArrays


# In[71]:


def valiation(textInput, top_ten):
    test_tf_idf = get_tf_idf(textInput)
    result = test_tf_idf.map(
        lambda x: (x[0], 1 if 'AU' in x[0] else 0, 0 if np.dot(x[1], ragulation_parameter) < 0 else 1)).collect()

    num = 0
    y = []
    y_pred = []
    for i in result:
        y.append(i[1])
        y_pred.append(i[2])
        if i[1] == 0 and i[1] != i[2]:
            if num < 3:
                temp = test_tf_idf.filter(lambda x: x[0] == i[0]).collect()
                print(temp[0][0])
                for i in top_ten:
                    print(temp[0][1][i[0]])
                print('\n')
                num += 1
    f1 = f1_score(y, y_pred, average='binary')
    a = confusion_matrix(y, y_pred)
    print(f1)
    print(a)


# In[72]:


start = time()
f = 20000
sc = SparkContext(appName="task2")
wikiPages = sc.textFile(sys.argv[1])


#wikiCategoryLinks=sc.textFile(sys.argv[2])
#wikiCats=wikiCategoryLinks.map(lambda x: x.split(",")).map(lambda x: (x[0].replace('"', ''), x[1].replace('"', '') ))
numberOfDocs = wikiPages.count()

# Each entry in validLines will be a line from the text file
validLines = wikiPages.filter(lambda x : 'id' in x and 'url=' in x)
# Now, we transform it into a set of (docID, text) pairs
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))

keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))

# Now, we split the text in each (docID, text) pair into a list of words
# After this step, we have a data set with
# (docID, ["word1", "word2", "word3", ...])
# We use a regular expression here to make
# sure that the program does not break down on some of the documents

regex = re.compile('[^a-zA-Z]')

# remove all non letter characters
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
# better solution here is to use NLTK tokenizer

# Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: ((i,1) for i in x[1]))

# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey(lambda x,y: x+y)
# Get the top 20,000 words in a local array in a sorted format based on frequency
# If you want to run it on your laptio, it may a longer time for top 20k words.
topWords = allCounts.top(f,key = lambda x: x[1])

topWordsK = sc.parallelize(range(f))

# Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
# ("NextMostCommon", 2), ...
# the number will be the spot in the dictionary used to tell us
# where the word is located
dictionary = topWordsK.map (lambda x : (topWords[x][0], x))
print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ", dictionary.top(20, lambda x : x[1]))
dictionary.cache()


# In[73]:


a = get_tf_idf(sys.argv[1])
stop = time()
time1 = stop - start
print('reading the data costs '+str(time1) + "seconds")


# In[74]:


def convert_1(x):
    temp = []
    if 'AU' in x[0]:
        return(1,
               x[1],
               max(0,(1-(1)*np.dot(x[1],parameter_vector_current))))
    else:
        return(-1,
               x[1],
               max(0,(1-(-1)*np.dot(x[1],parameter_vector_current))))
def convert_2(x):
    temp = []
    if x[0] == -1:
        return(x[2],
               (-1)*x[1] if x[2]!=0 else 0)
    elif x[0] == 1:
        return(x[2],
               (1)*x[1] if x[2]!=0 else 0)


# In[85]:
start = time()


learningRate = 1
c = 50
num_iteration = 0
precision = 0.1
oldCost = 999999
mu, sigma = 0, 0.1 # mean and standard deviation
parameter_vector_current = np.random.normal(mu, sigma, f)
a.cache()
while True:
    num_iteration+=1

    # Calculate the prediction with current regression coefficients.
    # We compute costs just for monitoring
    temp = a.map(convert_1)
    cost_list =temp.map(convert_2).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    cost = (1/numberOfDocs)*cost_list[0]+ (1/numberOfDocs*c*2)*(np.square(parameter_vector_current).sum())
    # calculate gradients.

    if cost > oldCost:
        break
    l_gradient_list = (1/numberOfDocs)*cost_list[1]+(1/numberOfDocs*c)*parameter_vector_current
    prevp = np.sqrt(np.square(parameter_vector_current).sum())
    parameter_vector_current = parameter_vector_current - learningRate *l_gradient_list
    nextp = np.sqrt(np.square(parameter_vector_current).sum())
    if abs(nextp - prevp) < 0.001:
        break
    # update the weights - Regression Coefficients

    # Stop if the cost is not descreasing
    if cost > oldCost:
        learningRate = learningRate * 0.5
        oldCost = cost

    if cost < oldCost:
        learningRate = learningRate * 1.05
        oldCost = cost
    print("Iteration No.=", num_iteration, " Cost=", cost)
    print("parameter : ", parameter_vector_current)

ragulation_parameter = parameter_vector_current

stop = time()
time2 = stop - start
print('training the data costs '+str(time2) + "seconds")


# In[86]:


def valiation(textInput):
    test_tf_idf = get_tf_idf(textInput)
    result = test_tf_idf.map(
        lambda x: (x[0], 1 if 'AU' in x[0] else -1, -1 if np.dot(x[1], ragulation_parameter) < 0 else 1)).collect()

    y = []
    y_pred = []
    for i in result:
        y.append(i[1])
        y_pred.append(i[2])
        
    f1 = f1_score(y, y_pred, average='binary')
    a = confusion_matrix(y, y_pred)
    print(f1)
    print(a)


# In[87]:

start = time()

valiation(sys.argv[2])

stop = time()
time3 = stop - start
print('testing the data costs '+str(time3) + "seconds")
# In[64]:
print('total time cost are '+str(time1+time2+time3) + "seconds")


sc.stop()


# In[ ]:




