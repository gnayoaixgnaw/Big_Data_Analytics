import sys
from operator import add
from pyspark import SparkContext

sc = SparkContext(appName="task1")

lines = sc.textFile(sys.argv[1])
taxilines = lines.map(lambda X: X.split(','))

def isfloat (value):
    try:
        float(value)
        return True
    except:
        return False
def correctRows(p):
    if(len(p) == 17):
        if(isfloat(p[5])) and isfloat(p[11]):
            if(float(p[5]!=0) and float(p[1]!=0)):
                return p

def counting(line):
    taxi = line[0]
    driver = line[1]
    return (taxi,1)

texilinesCorrected = taxilines.filter(correctRows)
x = texilinesCorrected.map(counting).reduceByKey(add)
y = x.top(10,key = lambda y:y[1])
z=sc.parallelize(y).coalesce(1)
z.collect()
z.saveAsTextFile(sys.argv[2])
sc.stop()

