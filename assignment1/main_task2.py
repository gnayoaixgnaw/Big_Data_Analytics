import sys
from operator import add
from pyspark import SparkContext

sc = SparkContext(appName="task2")

lines = sc.textFile(sys.argv[1])
taxilines = lines.map(lambda X: X.split(','))


def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False


def correctRows(p):
    if (len(p) == 17):
        if (isfloat(p[5])) and isfloat(p[11]):
            if (float(p[5] != 0) and float(p[1] != 0 and float(p[4]) != 0 and float(p[16]) != 0)):
                return p


def counting(line):
    driver = line[1]
    duration = line[4]
    amount = line[16]
    return (driver, (duration, amount))


def calculate(line):
    driver1 = line[0]
    amount1 = float(line[1][1])
    duration1 = float(line[1][0])
    average = float(amount1 / (duration1) * 60)
    return (driver1, average)


texilinesCorrected = taxilines.filter(correctRows)
x = texilinesCorrected.map(counting).reduceByKey(lambda x, y: (float(x[0]) + float(y[0]), float(x[1]) + float(y[1])))
y = x.map(calculate).reduceByKey(lambda x: x[0])
z = y.top(10, key=lambda y: y[1])

k=sc.parallelize(z).coalesce(1)
k.collect()
k.saveAsTextFile(sys.argv[2])
sc.stop()
