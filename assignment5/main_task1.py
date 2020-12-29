from __future__ import print_function

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


# In[56]:


def buildArray(listOfIndices):
    returnVal = np.zeros(f)

    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1

    mysum = np.sum(returnVal)

    returnVal = np.divide(returnVal, mysum)

    return returnVal


# In[57]:


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


# In[58]:


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


# In[59]:


start = time()
f = 20000
sc = SparkContext(appName="task1")
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


# In[60]:


a = get_tf_idf(sys.argv[1])
a.cache()
stop = time()
time1 = stop - start
print('reading the data costs '+str(time1) + "seconds")

start = time()
b = a.map(lambda x: LabeledPoint(1 if 'AU' in x[0] else 0,x[1]))


# In[61]:


model = LogisticRegressionWithLBFGS.train(b,iterations=100)
stop = time()
time2 = stop - start
print('training the data costs '+str(time2) + "seconds")


# In[62]:


start = time()
test_tf_idf = get_tf_idf(sys.argv[2])
result = test_tf_idf.map(lambda x: x[1])
y =test_tf_idf.map(lambda x: 1 if 'AU' in x[0] else 0).collect()
prediction = model.predict(result).collect()


# In[63]:


f1 = f1_score(y, prediction, average='binary')
a = confusion_matrix(y, prediction)
print(f1)
print(a)

stop = time()
time3 = stop - start
print('testing the data costs '+str(time3) + "seconds")


# In[64]:


print('total time cost are '+str(time1+time2+time3) + "seconds")


# In[65]:


sc.stop()
