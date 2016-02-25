# CollabFiltering-Netflix-PySpark
Collaborative Filtering - spark.mllib
Collaborative filtering
Explicit vs. implicit feedback
Scaling of the regularization parameter
Examples
Tutorial
Collaborative filtering
Collaborative filtering is commonly used for recommender systems. These techniques aim to fill in the missing entries of a user-item association matrix. spark.mllib currently supports model-based collaborative filtering, in which users and products are described by a small set of latent factors that can be used to predict missing entries. spark.mllib uses the alternating least squares (ALS) algorithm to learn these latent factors. The implementation in spark.mllib has the following parameters:

numBlocks is the number of blocks used to parallelize computation (set to -1 to auto-configure).
rank is the number of latent factors in the model.
iterations is the number of iterations to run.
lambda specifies the regularization parameter in ALS.
implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data.
alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations.
Explicit vs. implicit feedback
The standard approach to matrix factorization based collaborative filtering treats the entries in the user-item matrix as explicit preferences given by the user to the item.

It is common in many real-world use cases to only have access to implicit feedback (e.g. views, clicks, purchases, likes, shares etc.). The approach used in spark.mllib to deal with such data is taken from Collaborative Filtering for Implicit Feedback Datasets. Essentially instead of trying to model the matrix of ratings directly, this approach treats the data as a combination of binary preferences and confidence values. The ratings are then related to the level of confidence in observed user preferences, rather than explicit ratings given to items. The model then tries to find latent factors that can be used to predict the expected preference of a user for an item.

Scaling of the regularization parameter
Since v1.1, we scale the regularization parameter lambda in solving each least squares problem by the number of ratings the user generated in updating user factors, or the number of ratings the product received in updating product factors. This approach is named “ALS-WR” and discussed in the paper “Large-Scale Parallel Collaborative Filtering for the Netflix Prize”. It makes lambda less dependent on the scale of the dataset. So we can apply the best parameter learned from a sampled subset to the full dataset and expect similar performance.

Examples
Scala
Java
Python
In the following example we load rating data. Each row consists of a user, a product and a rating. We use the default ALS.train() method which assumes ratings are explicit. We evaluate the recommendation by measuring the Mean Squared Error of rating prediction.

Refer to the ALS Python docs for more details on the API.

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# Load and parse the data
data = sc.textFile("data/mllib/als/test.data")
ratings = data.map(lambda l: l.split(','))\
    .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

# Save and load model
model.save(sc, "target/tmp/myCollaborativeFilter")
sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
Find full example code at "examples/src/main/python/mllib/recommendation_example.py" in the Spark repo.
If the rating matrix is derived from other source of information (i.e., it is inferred from other signals), you can use the trainImplicit method to get better results.

# Build the recommendation model using Alternating Least Squares based on implicit ratings
model = ALS.trainImplicit(ratings, rank, numIterations, alpha=0.01)
In order to run the above application, follow the instructions provided in the Self-Contained Applications section of the Spark Quick Start guide. Be sure to also include spark-mllib to your build file as a dependency.
