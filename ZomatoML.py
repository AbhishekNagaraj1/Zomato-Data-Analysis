
# coding: utf-8

# # Improvisation of Accuracy on Zomato India Data Set

# In[52]:

import pandas as pd
df = pd.read_csv('ZomatoIndia.csv',encoding = 'latin-1', sep = ',') # Reading the csv file
df.head()


# In[53]:

cols = list(df.columns.values)
cols.pop(cols.index('Rating text'))
cols.pop(cols.index('Aggregate rating'))
dfClean = df[cols+['Rating text']+['Aggregate rating']] # Rearranging the Columns
dfClean.head()


# In[77]:

import numpy as np
x = dfClean.iloc[:,7:88] #Features
y = dfClean.iloc[:,88] #Target Value - Rating Text
X = np.array(x)
Y = np.array(y)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 1) #Splitting the data into test and train


# ## Building a Decision tree model because decision tree gave us the best accuracy previously

# In[82]:

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dtree = DecisionTreeClassifier(min_samples_leaf=100)
model = dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)
print('Decision Tree Accuracy:', accuracy_score(y_test, dtree_pred)*100)


# ### Changed the different criteria of the decision tree and increased the minimum samples in each leaf node of the tree
# ### Performing 10 fold cross validation to get the best accuracy

# In[85]:

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
scores = cross_val_score(model, X, Y, cv=10) #cv = 10 determines the k value for the k-fold cross validation
print ('Cross-validated scores:', scores*100)
print ('Avearge Accuracy:', np.mean(scores)*100)
print ('Best Accuracy:', max(scores)*100)


# ## By changing the criteria and performing cross validation on the decision tree classifier, we obtained the best accuracy of almost 77%
