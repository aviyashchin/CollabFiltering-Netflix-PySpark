#stolen from http://web.cs.ucla.edu/~zhoudiyu/tutorial/
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans
from numpy import array
from math import sqrt


#CLUSTERING
sc = SparkContext()
 
#4 data points (0.0, 0.0), (1.0, 1.0), (9.0, 8.0) (8.0, 9.0)
data = array([0.0,0.0, 1.0,1.0, 9.0,8.0, 8.0,9.0]).reshape(4,2)
 
#Generate K means
model = KMeans.train(sc.parallelize(data), 2, maxIterations=10, runs=30, initializationMode="random")
 
#Print out the cluster of each data point
print (model.predict(array([0.0, 0.0])))
print (model.predict(array([1.0, 1.0])))
print (model.predict(array([9.0, 8.0])))
print (model.predict(array([8.0, 0.0])))


#Standardizes features by removing the mean and scaling to unit variance using column summary statistics on the samples in the training set.
from pyspark.mllib.feature import Normalizer
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext
from pyspark.mllib.feature import StandardScaler

sc = SparkContext()

vs = [Vectors.dense([-2.0, 2.3, 0]), Vectors.dense([3.8, 0.0, 1.9])]

dataset = sc.parallelize(vs)

#all false, do nothing.
standardizer = StandardScaler(False, False)
model = standardizer.fit(dataset)
result = model.transform(dataset)
for r in result.collect(): print r

print("\n")

#deducts the mean
standardizer = StandardScaler(True, False)
model = standardizer.fit(dataset)
result = model.transform(dataset)
for r in result.collect(): print r

print("\n")

#divides the length of vector
standardizer = StandardScaler(False, True)
model = standardizer.fit(dataset)
result = model.transform(dataset)
for r in result.collect(): print r

print("\n")

#Deducts min first, divides the length of vector later
standardizer = StandardScaler(True, True)
model = standardizer.fit(dataset)
result = model.transform(dataset)
for r in result.collect(): print r

print("\n")


