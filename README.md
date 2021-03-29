## AI - Machine Learning

### 1. Naïve Bayes Classification for Text

In this project, you will implement your own version of a Naïve Bayesian classifier and use it for automatic text categorization. Specifically you will implement the Multinomial Naïve Bayes algorithm. Note that the algorithm consists of two separate components: training and test. The training component learns from the training data by computing the prior and conditional class probabilities associated with different features (in this case terms in the vocabulary). The test component uses the learned model to classify a test instance that was not used for training (to simulate the situation where the learned model is used to classify a new document into one of the existing classes).

### 2. Simple Recommender System using K-Nearest-Neighbor Collaborative Filtering

In this project, you will implement your own version of a collaborative filtering recommender system that will use the K-Nearest-Neighbor (KNN) strategy to predict users' ratings on items. There is no speparate training component to generate a model. Instead all the work is done at prediction time: when we want to generate a prediction for a test instance (in this case a user), at that point, we measure the similarity of that test instance (a new user), x, to every instance in the training data to identify the K most similar users to x. These are the K nearest neighbors. Your implementation should use the Pearson Correlation Coefficient as the similarity function (to compute similarities between the test instance and the training instances).
