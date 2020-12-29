from __future__ import print_function

import sys
import re
from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np

from pyspark.sql import SparkSession

def buildArray(listOfIndices):
    # returnVal = np.zeros(20000)
    returnVal = [0] * f

    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1

    # mysum = np.sum(returnVal)
    mysum = 0
    for i in returnVal:
        mysum += i

    # returnVal = np.divide(returnVal, mysum)
    for i in range(len(returnVal)):
        returnVal[i] = returnVal[i] / mysum

    return returnVal


def build_zero_one_array(listOfIndices):
    returnVal = np.zeros(f)

    for index in listOfIndices:
        if returnVal[index] == 0: returnVal[index] = 1

    return returnVal


def stringVector(x):
    returnVal = str(x[0])
    for j in x[1]:
        returnVal += ',' + str(j)
    return returnVal


def cousinSim(x, y):
    normA = np.linalg.norm(x)
    normB = np.linalg.norm(y)
    return np.dot(x, y) / (normA * normB)

def transfer1(x):
    temp = []
    for i in x:
        for j in i:
            temp.append(j)
    return temp
def transfer2(x):
    dic = {}
    list1 = []
    for i in x:
        if i in dic.keys():
            dic[i] +=1
        else:
            dic[i] = 1
    for k ,v in zip(dic.keys(),dic.values()):
        list1.append([k,v])
    dic1 = sorted(list1, key = lambda x:x[1], reverse=True)
    for j in range(f):
        dic1[j] = [dic1[j][0],j]
    return dic1[0:f]
def auto_index1(x):
    global idx
    idx = 0
    if x is not None:
        idx += 1
    return idx - 1

def get_id_text_list1(x):
    id = x[0]
    list_all = [id]
    temp = []
    regex = re.compile('[^a-zA-Z]')
    a = regex.sub(' ', x[1])
    for i in a.lower().split():
        if i in dic2.keys():
            temp.append(dic2[i])
    list_all.append(temp)
    return list_all

def get_array1(x):
    id = x[0]
    list_all = [id]
    list_all.append(buildArray(list(x[1])))
    print(list_all)
    return list_all

def get_zero_one1(x):
    temp = []
    for i in x[1]:
        if i > 0:
            temp.append(1)
        else:
            temp.append(0)
    return temp

def getPrediction(textInput, k):
    # Create an RDD out of the textIput

    input_text = spark.sparkContext.parallelize([textInput])
    input_text.take(1)
    myDoc = input_text.map(lambda x: (x,)).toDF()

    split1 = udf(lambda x: regex.sub(' ', x).lower().split(), ArrayType(StringType(), containsNull=False))
    allWordssplit = myDoc.withColumn('words', split1('_1')).drop('_1')

    def get_id_text_list_single1(x):
        temp = []
        for i in x:
            if i in dic2.keys():
                temp.append(int(dic2[i]))
        return temp

    get_id_text_list_single = udf(lambda x: get_id_text_list_single1(x), ArrayType(IntegerType(), containsNull=False))
    allDictionaryWordsInThatDoc = allWordssplit.withColumn('words_indic', get_id_text_list_single('words')).drop(
        'words')

    myArray = buildArray(allDictionaryWordsInThatDoc.select("words_indic").collect()[0][0])

    myArray = np.multiply(np.array(myArray), idfArray)

    def get_dot1(x):
        temp = np.dot(np.array(x), myArray)
        return temp

    get_dot = udf(lambda x: get_dot1(x).tolist(), FloatType())

    distances = featuresRDD.withColumn('rank', get_dot('TFidf')).drop('TFidf')

    topK = distances.orderBy(desc("rank"), '_c1').limit(k)
    add_one = udf(lambda x: 1, IntegerType())

    docIDRepresented = topK.withColumn('times', add_one('rank')).drop('rank')

    docIDRepresented1 = docIDRepresented.groupBy('_c1').agg(F.sum('times').alias('sum'))

    numTimes = docIDRepresented1.orderBy(desc("sum")).limit(k)
    temp = []
    for item in numTimes.collect():
        temp.append((item[0], item[1]))
    return temp



if __name__ == "__main__":
    spark = SparkSession.builder.appName("DataFrame").getOrCreate()
    f = 20000
    df = spark.read.text(sys.argv[1])
    wikiCategoryLinks=spark.read.csv(sys.argv[2])
    
    validLines = df.filter(df.value.contains('id') & df.value.contains('url='))

    numberOfDocs = df.count()

    get_id = udf(lambda x: x[x.index('id="') + 4 : x.index('" url=')])
    get_text = udf(lambda x: x[x.index('">') + 2:][:-6])
    validLines1 = validLines
    validLines = validLines.withColumn("id", get_id("value"))
    keyAndText = validLines.withColumn("text", get_text("value")).drop('value')
    
    transfer = udf(lambda x:transfer1(x),ArrayType(StringType(), containsNull=False))

    regex = re.compile('[^a-zA-Z]')
    split = udf(lambda x : regex.sub(' ', x).lower().split(),ArrayType(StringType(), containsNull=False))
    keyAndListOfWords = keyAndText.withColumn('words', split('text')).drop('text')

    keyAndListOfWords_drop_id = keyAndListOfWords.drop('id')
    keyAndListOfWords_count = keyAndListOfWords.withColumn('index',lit(1))

    keyAndListOfWords_count = keyAndListOfWords_count.groupby('index').agg(collect_list('words').alias("words"))
    allWords = keyAndListOfWords_count.withColumn('words_list',transfer('words')).drop('words')
    #allWords.show()
    row_list = allWords.select('words_list').collect()
    
    transfer = udf(lambda x:transfer2(x),ArrayType(StringType(), containsNull=False))

    topWords = allWords.withColumn('words_dict',transfer('words_list'))

    #keyAndListOfWords.show()
    word_list=allWords.select('words_list').collect()[0][0]
    from collections import Counter

    counts = Counter(word_list)

    df_dic = spark.createDataFrame(counts.items(), ['words','counts'])
    df_dic=df_dic.orderBy(desc("counts")).limit(f)
    
    auto_index = udf(auto_index1, IntegerType())
    df_dic = df_dic.withColumn('index',auto_index('words')).drop('counts')

    get_id_text = udf(lambda x: [x[x.index('id="') + 4 : x.index('" url=')],x[x.index('">') + 2:][:-6]],ArrayType(StringType(), containsNull=False))

    word_doc = validLines1.withColumn("id", get_id_text("value")).drop('value')

    dic1 = dict(sorted(counts.items(),key=lambda x:x[1],reverse=True) ) #
    dic2 = {}
    num = 0
    for i in dic1.keys():
        dic2[i] = num
        num+=1
        if num == f:
            break
    
    get_id_text_list = udf(lambda x: get_id_text_list1(x), ArrayType(StringType(), containsNull=False))
    word_doc_list = word_doc.withColumn("id_list", get_id_text_list("id")).drop('id')
    
    get_array = udf(lambda x:get_array1(x))
    allDocsAsNumpyArrays = word_doc_list.withColumn("array_list", get_array("id_list")).drop('id_list')
    
    add_index = udf(lambda x : x[0])
    add_list = udf(lambda x : x[1])
    get_zero_one = udf(lambda x:get_zero_one1(x),ArrayType(IntegerType(), containsNull=False))
    temp_list = allDocsAsNumpyArrays.withColumn("index", add_index("array_list"))
    temp_list1 = temp_list.withColumn('list',add_list('array_list')).drop('array_list')
    zeroOrOne = temp_list.withColumn("zero_one_list", get_zero_one("array_list")).drop('array_list')
    
    from pyspark.sql import functions as F
    
    n = len(zeroOrOne.select('zero_one_list').first()[0])

    resultDF = zeroOrOne.agg(F.array(*[F.sum(F.col("zero_one_list")[i]) for i in range(n)]).alias("sum")).select('sum').collect()[0][0]

    multiplier = np.full(f, numberOfDocs)
    idfArray = np.log(np.divide(np.full(f, numberOfDocs), resultDF))

    TFidf = udf(lambda x:np.multiply(x, idfArray).tolist(),ArrayType(FloatType(), containsNull=False))

    allDocsAsNumpyArraysTFidf = temp_list1.withColumn("TFidf", TFidf("list")).drop('list')

    featuresRDD = wikiCategoryLinks.join(allDocsAsNumpyArraysTFidf,wikiCategoryLinks._c0==allDocsAsNumpyArraysTFidf.index).drop('_c0').drop('index')

    print(allDocsAsNumpyArrays.take(3))
    
    print(allDocsAsNumpyArraysTFidf.take(2))
    
    print(getPrediction('Sport Basketball Volleyball Soccer', 10))

    print(getPrediction('What is the capital city of Australia?', 10))

    print(getPrediction('How many goals Vancouver score last year?', 10))





    
    
    
    
    
    
    
 
        
    
