#!/usr/bin/env python

import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
from numpy import array


def parseRating(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    fields = line.strip().split("::")
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseMovie(line):
    """
    Parses a movie record in MovieLens format movieId::movieTitle .
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1]

def loadRatings(ratingsFile):
    """
    Load ratings from file.
    """
    if not isfile(ratingsFile):
        print "File %s does not exist." % ratingsFile
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print "No ratings provided."
        sys.exit(1)
    else:
        return ratings

def computeRmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    """
# model = bestModel
# data = test
# n = numTest
# Load and parse the data
data = sc.textFile("test.data")
ratings = data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

    #predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    testdata = ratings.map(lambda p: (int(p[0]), int(p[1])))
    predictions = model.predictAll(test.map(lambda r: ((r[0], r[1]), r[2])))

    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))
    
#if __name__ == "__main__":
#    if (len(sys.argv) != 3):
#        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
#          "MovieLensALS.py movieLensDataDir personalRatingsFile"
#        sys.exit(1)

# set up environment
conf = SparkConf() \
  .setAppName("MovieLensALS") \
  .set("spark.executor.memory", "2g")
sc = SparkContext(conf=conf)

# load personal ratings
#myRatings = loadRatings(sys.argv[2])
myRatings = loadRatings("ratings.dat")
myRatingsRDD = sc.parallelize(myRatings, 1)

# load ratings and movie titles

#movieLensHomeDir = sys.argv[1]
movieLensHomeDir = ""

# ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
ratings = sc.textFile(join(movieLensHomeDir, "ratings.dat")).map(parseRating)

def parseMovie(line):
  fields = line.split("::")
  return int(fields[0]), fields[1]

# movies is an RDD of (movieId, movieTitle)
movies = dict(sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseMovie).collect())

numRatings = ratings.count()
numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
numMovies = ratings.values().map(lambda r: r[1]).distinct().count()

print "Got %d ratings from %d users on %d movies." % (numRatings, numUsers, numMovies)


# Splitting training data
# We will use MLlib’s ALS to train a MatrixFactorizationModel, which takes a RDD[Rating] object as input in Scala and RDD[(user, product, rating)] in Python. ALS has training parameters such as rank for matrix factors and regularization constants. To determine a good combination of the training parameters, we split the data into three non-overlapping subsets, named training, test, and validation, based on the last digit of the timestamp, and cache them. We will train multiple models based on the training set, select the best model on the validation set based on RMSE (Root Mean Squared Error), and finally evaluate the best model on the test set. We also add your ratings to the training set to make recommendations for you. We hold the training, validation, and test sets in memory by calling cache because we need to visit them multiple times.

numPartitions = 4
training = ratings.filter(lambda x: x[0] < 6) \
  .values() \
  .union(myRatingsRDD) \
  .repartition(numPartitions) \
  .cache()

validation = ratings.filter(lambda x: x[0] >= 6 and x[0] < 8) \
  .values() \
  .repartition(numPartitions) \
  .cache()

test = ratings.filter(lambda x: x[0] >= 8).values().cache()

numTraining = training.count()
numValidation = validation.count()
numTest = test.count()

print "Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)


# 7. Training using ALS
# In this section, we will use ALS.train to train a bunch of models, and select and evaluate the best. Among the training paramters of ALS, the most important ones are rank, lambda (regularization constant), and number of iterations. The train method of ALS we are going to use is defined as the following:

ranks = [8, 12]
lambdas = [1.0, 10.0]
numIters = [10, 20]
bestModel = None
bestValidationRmse = float("inf")
bestRank = 0
bestLambda = -1.0
bestNumIter = -1

for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
    #model = ALS.train(training, rank, numIter, lmbda)
    model = ALS.train(training, rank, numIter)
    validationRmse = computeRmse(model, validation, numValidation)
    print "RMSE (validation) = %f for the model trained with " % validationRmse + \
          "rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter)
    if (validationRmse < bestValidationRmse):
        bestModel = model
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lmbda
        bestNumIter = numIter

testRmse = computeRmse(bestModel, test, numTest)

# evaluate the best model on the test set
print "The best model was trained with rank = %d and lambda = %.1f, " % (bestRank, bestLambda) \
  + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse)


# clean up
sc.stop()