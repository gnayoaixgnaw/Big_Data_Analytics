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
            if (int(p[4]) >= 120) and (int(p[4]) <= 3600):
                if (float(p[11]) >= 3) and (float(p[11]) <= 200):
                    if (float(p[5]) >= 1) and (float(p[5]) <= 50):
                        if float(p[15]) >= 3:
                            return p


def get_rdd(p):
    return (p[5], p[11], float(p[5]) ** 2, float(p[5]) * float(p[11]))


def get_vector(p):
    vector = [float(p[4]), float(p[5]), float(p[11]), float(p[12])]
    a = (float(p[16]) - (np.dot(np.array(vector), parameter_vector_current[1:5]) + parameter_vector_current[0]))
    temp = []
    for i in vector:
        temp.append(a * i)
    b = np.array(temp)
    return (float(a), np.array(vector), b, float(a) ** 2)

sc = SparkContext(appName="task3")

lines = sc.textFile(sys.argv[1])
taxilines = lines.map(lambda X: X.split(','))

texilinesCorrected = taxilines.filter(correctRows)
texilinesCorrected.cache()

n = texilinesCorrected.count()

learningRate = 0.001
num_iteration = 100
precision = 0.1
oldCost = 0
parameter_vector_current = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
result_list = []

for i in range(num_iteration):
    # Calculate the prediction with current regression coefficients.
    # We compute costs just for monitoring
    temp_list = texilinesCorrected.map(get_vector).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]))

    cost = (1 / (2 * n) )* temp_list[3]
    # calculate gradients.

    m_gradient_list = (-1.0 / n) * temp_list[2]
    b_gradient = (-1.0 / n) * temp_list[0]

    # update the weights - Regression Coefficients
    parameter_vector_current[1:5] = parameter_vector_current[1:5] - learningRate * m_gradient_list
    parameter_vector_current[0] = parameter_vector_current[0] - learningRate * b_gradient

    # Stop if the cost is not descreasing
    if cost > oldCost:
        learningRate = learningRate * 0.5
        oldCost = cost

    if cost < oldCost:
        learningRate = learningRate * 1.05
        oldCost = cost
    result_list.append("Iteration No.=" + str(i) + " Cost=" + str(cost))
    result_list.append("parameter : " )
    result_list.append(parameter_vector_current)
    print("Iteration No.=", i, " Cost=", cost)
    print("parameter : ", parameter_vector_current[1:5])

k=sc.parallelize(result_list).coalesce(1)
k.collect()
k.saveAsTextFile(sys.argv[2])
sc.stop()
