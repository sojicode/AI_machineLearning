
import numpy as np
import pandas as pd
import os


trainMatrix = np.loadtxt('./newsgroups/trainMatrix.txt').T
trainClasses = np.loadtxt('./newsgroups/trainClasses.txt')[:,1]
testMatrix = np.loadtxt('./newsgroups/testMatrix.txt').T
testClasses = np.loadtxt('./newsgroups/testClasses.txt')[:,1]


def naiveBayesTrain(X, y):
    prior = []
    n_voc = X.shape[1]
    pwc = np.zeros((n_voc, 2))
    for i in [0, 1]:
        prior.append(np.mean(y == i))
        docs = X[y == i]
        n_i = docs.sum() # total number of words
        pwc[:,i] = (docs.sum(axis=0) + 1)/(n_i + n_voc)
    
    return prior, pwc

def naiveBayesPredict(prior, pwc, X):
    preds = []
    for j in range(X.shape[0]):
        preds.append(1*(np.log(prior[1]/prior[0]) + sum(np.log(pwc[:,1]/pwc[:,0])*X[j]) > 0))
    return np.array(preds)



# train naive bayes model and determine the probabilities
prior, pwc = naiveBayesTrain(trainMatrix, trainClasses)


# predict the class on the test using the trained parameters
y_pred = naiveBayesPredict(prior, pwc, testMatrix)


# Part 1

print()
print('Full test data accuracy:', np.mean(y_pred == testClasses),'\n')


# Part 2

print('Predicted class labels for the first 20 document instances:','\n')
print(pd.DataFrame({'actual': testClasses, 'predicted': y_pred}).astype(int).head(20),'\n')


# Part 3

terms = pd.read_csv('./newsgroups/terms.txt', header=None)[0].values.tolist()
terms[:3]

probs = []
words = ["program", "includ", "match", "game", "plai", "window", "file", "subject", "write"]
for item in words:
    indx = terms.index(item)
    probs.append(pwc[indx,].tolist())

print('Class probabilities for these terms:', '\n')
print(pd.DataFrame(probs, index=words),'\n')

