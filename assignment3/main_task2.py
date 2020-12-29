import sys
import re
import numpy as np
from operator import add

from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext
from pyspark.sql import functions as func


def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False


def correctRows(p):
    if (len(p) == 17):
        if (isfloat(p[5])) and isfloat(p[11]):
            if (float(p[5] != 0) and float(p[1] != 0)):
                if (int(p[4]) >= 120) and (int(p[4]) <= 3600):
                    if (float(p[5]) >= 1) and (float(p[5]) <= 50):
                        if (float(p[11]) >= 3) and (float(p[11]) <= 200):
                            if float(p[15]) >= 3:
                                return p

def get_rdd(p):
    return (p[5], p[11], float(p[5]) ** 2, float(p[5]) * float(p[11]))


def get_y_predict(p):
    return(float(p[11]),(m_current * float(p[5]) + b_current),
           float(p[5]) * float(p[11]),float(p[5]) * (m_current * float(p[5]) + b_current)
           ,(float(p[11]) - (m_current * float(p[5]) + b_current)) ** 2)



# def get_cost(p):
#     return ((float(p[11]) - (m_current * float(p[5]) + b_current)) ** 2)

sc = SparkContext(appName="task2")

lines = sc.textFile(sys.argv[1])
taxilines = lines.map(lambda X: X.split(','))

taxilinesCorrected = taxilines.filter(correctRows)

n = taxilinesCorrected.count()

learningRate = 0.0001
num_iteration = 100
precision = 0.1
oldCost = 0

m_current = 0.1
b_current = 0.1

taxilinesCorrected.cache()

result_list=[]
for i in range(num_iteration):
    # Calculate the prediction with current regression coefficients.
    # We compute costs just for monitoring
    temp_list = taxilinesCorrected.map(get_y_predict).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3]+y[3], x[4]+y[4]))

    #cost_sum = taxilinesCorrected.map(get_cost).reduce(lambda x, y: (x + y))

    cost = (1 / n) * temp_list[4]

    # calculate gradients.
    m_gradient = (-2.0 / n) * (temp_list[2]-temp_list[3])
    b_gradient = (-2.0 / n) * (temp_list[0]-temp_list[1])

    # update the weights - Regression Coefficients
    m_current = m_current - learningRate * m_gradient
    b_current = b_current - learningRate * b_gradient

    # Stop if the cost is not descreasing
    #     if(abs(cost - oldCost) <= precision):
    #         print("Stoped at iteration", i)
    #         break
    if cost > oldCost:
        learningRate = learningRate * 0.5
    if cost < oldCost:
        learningRate = learningRate * 1.05

    oldCost = cost
    #     if(i % 10 ==0):
    result_list.append("Iteration No.="+ str(i) +" Cost=" + str(cost))
    result_list.append("m = "+ str(m_current) +" b=" + str(b_current))
    print("Iteration No.=", i, " Cost=", cost)
    print("m = ", m_current, " b=", b_current)

k=sc.parallelize(result_list).coalesce(1)
k.collect()
k.saveAsTextFile(sys.argv[2])
sc.stop()

