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
            if (float(p[5] != 0) and float(p[1] != 0 and float(p[4]) != 0 and float(p[16]) != 0) and float(
                    p[5]) != 0 and float(p[12]) != 0):
                return p


def counting(line):
    driver = line[1]
    duration = line[4]
    amount = line[16]
    return (driver, (duration, amount))


def calculate(line):
    hour = line[0]
    distance = float(line[1][1])
    surcharge = float(line[1][0])
    ratio = float(surcharge) / float(distance)
    return (hour, ratio)


def time(line):
    surcharge = line[12]
    distance = line[5]
    times = line[2].split(' ')[1].split(':')[0]
    ratio = float(surcharge) / float(distance)
    return (times, (surcharge, distance))


texilinesCorrected = taxilines.filter(correctRows)
x = texilinesCorrected.map(time).reduceByKey(lambda x, y: (float(x[0]) + float(y[0]), float(x[1]) + float(y[1])))
y = x.map(calculate).reduceByKey(lambda x: x[0])
z = y.top(10, key=lambda y: y[1])

k=sc.parallelize(z).coalesce(1)
k.collect()
k.saveAsTextFile(sys.argv[2])
sc.stop()
