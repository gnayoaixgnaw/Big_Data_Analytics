#!/usr/bin/env python
# coding: utf-8

# In[22]:


import sys
import re
import numpy as np
from operator import add

from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext
from pyspark.sql import functions as func
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# In[15]:


def buildArray(listOfIndices):
    returnVal = np.zeros(f)

    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1

    mysum = np.sum(returnVal)

    returnVal = np.divide(returnVal, mysum)

    return returnVal


def build_zero_one_array(listOfIndices):
    returnVal = np.zeros(f)

    for index in listOfIndices:
        if returnVal[index] == 0: returnVal[index] = 1

    return returnVal


# In[16]:


f = 20000
sc = SparkContext(appName="task3")
wikiPages = sc.textFile(sys.argv[1])
# wikiCategoryLinks=sc.textFile(sys.argv[2])

# wikiCats=wikiCategoryLinks.map(lambda x: x.split(",")).map(lambda x: (x[0].replace('"', ''), x[1].replace('"', '') ))
numberOfDocs = wikiPages.count()

# Each entry in validLines will be a line from the text file
validLines = wikiPages.filter(lambda x: 'id' in x and 'url=' in x)
# Now, we transform it into a set of (docID, text) pairs
keyAndText = validLines.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))


# Now, we split the text in each (docID, text) pair into a list of words
# After this step, we have a data set with
# (docID, ["word1", "word2", "word3", ...])
# We use a regular expression here to make
# sure that the program does not break down on some of the documents

regex = re.compile('[^a-zA-Z]')

# remove all non letter characters
keyAndListOfWords = keyAndText.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()))
# better solution here is to use NLTK tokenizer

# Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: ((i, 1) for i in x[1]))

# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey(lambda x, y: x + y)
# Get the top 20,000 words in a local array in a sorted format based on frequency
# If you want to run it on your laptio, it may a longer time for top 20k words.
topWords = allCounts.top(f, key=lambda x: x[1])

#
#print("Top Words in Corpus:", allCounts.top(10, key=lambda x: x[1]))

# We'll create a RDD that has a set of (word, dictNum) pairs
# start by creating an RDD that has the number 0 through 20000
# 20000 is the number of words that will be in our dictionary
topWordsK = sc.parallelize(range(f))

# Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
# ("NextMostCommon", 2), ...
# the number will be the spot in the dictionary used to tell us
# where the word is located
dictionary = topWordsK.map(lambda x: (topWords[x][0], x))
#print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ", dictionary.top(20, lambda x: x[1]))

# In[17]:


dictionary.cache()


# In[32]:


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

    return allDocsAsNumpyArraysTFidf


# In[33]:


def convert_theta(x):
    temp = []
    if 'AU' in x[0]:
        return(1,
               x[1],
               np.dot(x[1],parameter_vector_current))
    else:
        return(0,
               x[1],
               np.dot(x[1],parameter_vector_current))
def convert_final(x):
    temp = []
    if x[0] == 1:
        return(-1*x[0]*x[2],
               np.log(1+np.e**x[2]),
               -1*x[1]*x[0]+x[1]*(np.e**x[2]/(1+np.e**x[2])))
    elif x[0] == 0:
        return(-1*x[0]*x[2],
               np.log(1+np.e**x[2]),
               -1*x[1]*x[0]+x[1]*(np.e**x[2]/(1+np.e**x[2])))


# In[34]:


a = get_tf_idf(sys.argv[1])
a.cache()

# In[35]:


learningRate = 0.1
lambdas = 0.0001
num_iteration = 0
precision = 0.1
oldCost = 0
mu, sigma = 0, 0.1 # mean and standard deviation
parameter_vector_current = np.random.normal(mu, sigma, f)
result_list = []

while True:
    # Calculate the prediction with current regression coefficients.
    # We compute costs just for monitoring
    temp = a.map(convert_theta)
    cost_list =temp.map(convert_final).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]))
    cost = cost_list[0] + cost_list[1]+ lambdas*np.square(parameter_vector_current).sum()
    # calculate gradients.

    l_gradient_list = cost_list[2]+2*lambdas*parameter_vector_current
    prevp = np.sqrt(np.square(parameter_vector_current).sum())
    parameter_vector_current = parameter_vector_current - learningRate *l_gradient_list
    nextp = np.sqrt(np.square(parameter_vector_current).sum())
    if abs(nextp - prevp) < 0.01:
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
    num_iteration+=1

ragulation_parameter = parameter_vector_current



word_dic = dict()
for item in dictionary.collect():
    word_dic[item[1]] = item[0]

dic = dict()
for i in range(len(ragulation_parameter)):
    dic[i] = ragulation_parameter[i]

top_ten =sorted(dic.items(),key=lambda x:x[1],reverse=True)[0:10]
for j in top_ten:
    print(word_dic[j[0]])   # task2

# In[37]:

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


# In[38]:


valiation(sys.argv[2],top_ten)

# In[ ]:




