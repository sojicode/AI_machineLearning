
import pandas as pd
import numpy as np

data = pd.read_excel('knn-csc480-a4.xls', na_values=[' ']).set_index('Unnamed: 0')
cols = data.columns.tolist()

train = data.iloc[:20,:].values
test = data.iloc[21:,:].values

def correlation(x,y):
    x2 = x - x.mean()
    y2 = y - y.mean()
    return np.sum(x2*y2)/np.sqrt(np.sum(x2**2))/np.sqrt(np.sum(y2**2)) 


def predict(train, testrow, item, k):
    train_valid = train[~np.isnan(train[:,item]),:].copy() # get valid training data that item is not nan
    
    testrow = testrow.copy()
    test_excluded = np.delete(testrow, item)               # delete the item column that you predict later
    train_excluded = np.delete(train_valid, item, axis=1)
    
    test_excluded = np.nan_to_num(test_excluded)           # change 'nan' to zero
    train_excluded = np.nan_to_num(train_excluded)
    
    corr = [] 
       
    for i in range(train_excluded.shape[0]):
        corr.append(correlation(test_excluded,train_excluded[i])) # compute the value between train data and test data
        
    corr = np.array(corr)
    indx = np.argsort(corr)[-k:]
    
    pred = np.sum(corr[indx]*train_valid[indx,item])/np.sum(corr[indx]) # formular from lecture slide
    
    return pred


# Part 1

actual = []
predicted = []
k = 3
for i in range(5):
    items = np.where(~np.isnan(test[i]))[0] # using it's not NaN, 
    for item in items:
        actual.append(test[i][item])
        predicted.append(predict(train, test[i], item, k))
print()
print('MAE:', np.mean(abs(np.array(actual) - np.array(predicted)))) # avg of |acutual - predicted|.     


# Part 2

k_list = []
mae_list = []

for k in range(1, 21):
    k_list.append(k)
    actual = []
    predicted = []
    for i in range(5):
        items = np.where(~np.isnan(test[i]))[0]
        for item in items:
            actual.append(test[i][item])
            predicted.append(predict(train, test[i], item, k))
    mae = np.mean(abs(np.array(actual) - np.array(predicted)))
    mae_list.append(mae)

print()
print('{0:<5}  {1:<18} '.format('K', 'MAE'))
print('-'*27)
for i in range(len(mae_list)):
    print('{0:<5}  {1:<18} '.format(i+1, mae_list[i]))

    
# Part 3

k = 3
predicted = []
print()
print('{0:<5}  {1:<18} {2:4}'.format('User', 'Book', 'Score'))
print('-'*34)
for i in range(2):
    items = np.where(np.isnan(test[i]))[0]  # get the items that are not rated 
    for item in items:
        print('NU{0:<5} {1:<18} {2:.4f}'.format(i+1, cols[item], predict(train, test[i], item, k)))

# Part 4

def recommender(train, user, k, m):
    items = np.where(np.isnan(train[user]))[0] # items that the trainUser didn't rating
    train_aug = np.delete(train, user, axis=0) # delete those train user from train data
    scores = []
    for item in items:                         # predict the rating of items here
        scores.append((predict(train_aug, train[user], item, k), cols[item]))
        
    scores = sorted(scores, reverse=True)
    return scores[:m]


# k = 4, Top 3 recommendation using recommender function

k = 4
m = 3
print()
print('{0:<5}  {1:<18} {2:4}'.format('User', 'Book', 'Score'))
print('-'*34)
for i in [1, 4, 12, 19]:
    scores = recommender(train, i, k, m)
    for j in range(len(scores)):
        print('U{0:<5} {1:<18} {2:.4f}'.format(i+1, scores[j][1], scores[j][0]))
print()





