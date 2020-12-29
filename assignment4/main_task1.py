from __future__ import print_function

import sys
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext
from pyspark.sql import functions as func

def buildArray(listOfIndices):
    returnVal = np.zeros(f)

    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1

    mysum = np.sum(returnVal)

    returnVal = np.divide(returnVal, mysum)

    return returnVal


if __name__ == "__main__":
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
    dictionary = topWordsK.map (lambda x : (topWords[x][0], x))
    print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ", dictionary.top(20, lambda x : x[1]))
    
    dic = dictionary.collectAsMap()
    word1_index = dic['applicant']
    word2_index= dic['and']
    word3_index = dic['attack']
    word4_index = dic['protein']
    word5_index = dic['court']
    
    # Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
    # ("word1", docID), ("word2", docId), ...

    allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

    # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
    allDictionaryWords = dictionary.join(allWordsWithDocID)

    # Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
    justDocAndPos = allDictionaryWords.map(lambda x:(x[1][1],x[1][0]))
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

    zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0],np.where(x[1] > 0, 1, 0)))
    dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]
    # Create an array of 20,000 entries, each entry with the value numberOfDocs (number of docs)
    multiplier = np.full(f, numberOfDocs)

    # Get the version of dfArray where the i^th entry is the inverse-document frequency for the
    # i^th word in the corpus
    idfArray = np.log(np.divide(np.full(f, numberOfDocs), dfArray))
    # Finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors
    allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))
    
    all_AU_doc = allDocsAsNumpyArrays.filter(lambda x : 'AU' in x[0] )
    all_Wiki_doc = allDocsAsNumpyArrays.filter(lambda x : 'AU' not in x[0] )
    len_au = all_AU_doc.count()
    len_wike = all_Wiki_doc.count()
    
    zeroOrOne_AU = all_AU_doc.map(lambda x: (x[0],np.where(x[1] > 0, 1, 0)))
    dfArray_AU = zeroOrOne_AU.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]
    zeroOrOne_Wiki = all_Wiki_doc.map(lambda x: (x[0],np.where(x[1] > 0, 1, 0)))
    dfArray_Wiki = zeroOrOne_Wiki.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]
    
    print('average TF value of words applicant, and, attack, protein, court in court doc are :',
     dfArray_AU[346]/len_au,dfArray_AU[2]/len_au,dfArray_AU[502]/len_au,
      dfArray_AU[3014]/len_au,dfArray_AU[149]/len_au)
    print('average TF value of words applicant, and, attack, protein, court in wiki doc are :',
     dfArray_Wiki[346]/len_wike,dfArray_Wiki[2]/len_wike,dfArray_Wiki[502]/len_wike,
      dfArray_Wiki[3014]/len_wike,dfArray_Wiki[149]/len_wike)
    
    result_list = []
    result_list.append(['average TF value of words applicant, and, attack, protein, court in court doc are :',[dfArray_AU[346]/len_au,dfArray_AU[2]/len_au,dfArray_AU[502]/len_au,
      dfArray_AU[3014]/len_au,dfArray_AU[149]/len_au]])
    result_list.append(['average TF value of words applicant, and, attack, protein, court in wiki doc are :',[dfArray_Wiki[346]/len_wike,dfArray_Wiki[2]/len_wike,dfArray_Wiki[502]/len_wike,
      dfArray_Wiki[3014]/len_wike,dfArray_Wiki[149]/len_wike]])
    z=sc.parallelize(result_list).coalesce(1)
    z.collect()
    z.saveAsTextFile(sys.argv[2])
    sc.stop()


    
 
