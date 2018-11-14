#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
import sys 
import warnings


# In[2]:


def load_data(train_path, test_path, test_validate):
    col_names = ["age", "workclass", "education",
                 "marital-status", "occupation", "race",
                 "sex", "hours-per-week",
                 "country", "income"]
    train_data = pd.read_csv(train_path, header=None, names=col_names)
    test_data = pd.read_csv(test_path, header=None, names=col_names)
    dev_data = pd.read_csv(test_validate, header= None, names=col_names)

    return train_data, test_data, dev_data

def standardize_data(train_data, test_data, dev_data):
    # Fit scaler on train data only. Transform training and testing set
    numerical_col = ["age", "hours-per-week"]
    scaler = StandardScaler()
    train_data[numerical_col] = scaler.fit_transform(train_data[numerical_col])
    test_data[numerical_col] = scaler.fit_transform(test_data[numerical_col])
    dev_data[numerical_col] = scaler.fit_transform(dev_data[numerical_col])
    return train_data, test_data, dev_data


def split_data(train_data, test_data, dev_data):

    y_train = train_data["income"]
    X_train = train_data.drop("income", axis=1)

    y_test = test_data['income']
    X_test = test_data.drop("income", axis=1)

    y_dev = dev_data['income']
    X_dev = dev_data.drop("income", axis=1)

    return X_train, y_train, X_test, y_test, X_dev, y_dev
def ohe_data(X_train, y_train, X_test, y_test, X_dev, y_dev):
    """
    One hot encode categorical data.
    Args:
        X_train: Train features as Pandas DataFrame.
        y_train: Train labels as Pandas Series.
        X_test: Test features as Pandas DataFrame.
        y_test: Test labels as Pandas Series.
    Returns:
        X_train_ohe: One-hot encoded training features as Pandas DataFrame.
        y_train_ohe: One-hot encoded training labels as Pandas Series.
        X_test_ohe: One-hot encoded testing features as Pandas DataFrame.
        y_test_ohe: One-hot encoded testing labels as Pandas Series.
    """
    data = pd.concat([X_train, X_test])
    data_1 = pd.concat([X_train, X_dev])

    data_ohe = pd.get_dummies(data)
    data_ohe1 = pd.get_dummies(data_1)
    X_train_ohe = data_ohe[:len(X_train)]
    X_test_ohe = data_ohe[len(X_train):]
    X_dev_ohe = data_ohe1[len(X_train):]
    y_train_ohe = y_train.replace([' <=50K', ' >50K'], [-1, 1])
    y_test_ohe = y_test.replace([' <=50K', ' >50K'], [-1, 1])
    y_dev_ohe = y_dev.replace([' <=50K', ' >50K'], [-1, 1])
    X_train_ohe = np.array(X_train_ohe)
    y_train_ohe = np.array(y_train_ohe)
    X_test_ohe  = np.array(X_test_ohe)
    y_test_ohe  = np.array(y_test_ohe)
    X_dev_ohe = np.array(X_dev_ohe)
    y_dev_ohe= np.array(y_dev_ohe)
    return X_train_ohe, y_train_ohe, X_test_ohe, y_test_ohe, X_dev_ohe, y_dev_ohe
def preprocess_data():
    path_to_train = "income.train.txt"
    path_to_test = "income.test.txt"
    path_to_validate = "income.dev.txt"
    # Load the data
    print("Loading data...")
    train_data, test_data, dev_data = load_data(path_to_train, path_to_test,path_to_validate)
    # Standardize the data
    print("Standardizing the data...")
    train_data, test_data ,dev_data= standardize_data(train_data, test_data,dev_data)
    # Split data into features and labels
    X_train, y_train, X_test, y_test,X_dev, y_dev = split_data(train_data, test_data,dev_data)
    # One-hot encode the data
    X_train, y_train, X_test, y_test, X_dev, y_dev = ohe_data(X_train, y_train, X_test, y_test,X_dev,y_dev)
    
    return X_train, y_train, X_test, y_test, X_dev, y_dev
if __name__ == "__main__":

    X_train, y_train, X_test, y_test, X_dev, y_dev = preprocess_data()
    print("\nData sucessfully loaded.")


# In[3]:


def warn(*args,**kwargs):
    pass
warnings.warn = warn


# In[4]:


dep=[1,2,3,5,10]
n_trees=[10,20,40,60,80,100]
for depth in dep:
    plt_valid=[]
    plt_train=[]
    plt_test=[]
    base = DecisionTreeClassifier(max_depth=depth)
    print("For Depth = " ,depth)
    for trees in n_trees:
        print("For number of trees = ", trees)
        clf=BaggingClassifier(base_estimator = base ,n_estimators=int(trees))
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_train)
        plt_train.append(accuracy_score(y_pred,y_train)*100)
        print ("Accuracy score for training data ",accuracy_score(y_pred,y_train)*100)
        y_pred=clf.predict(X_dev)
        print ("Accuracy score for Validation data ",accuracy_score(y_pred,y_dev)*100)
        plt_valid.append(accuracy_score(y_pred,y_dev)*100)
        y_pred=clf.predict(X_test)
        print ("Accuracy score for Testing data ",accuracy_score(y_pred,y_test)*100)
        plt_test.append(accuracy_score(y_pred,y_test)*100)
        print()
    print()
    plt.xlabel("Size of ensemble")
    plt.ylabel("Accuracy")
    plt.plot(n_trees,plt_train,label = 'Training Accuracy')
    plt.plot(n_trees,plt_test, label = 'Testing Accuracy')
    plt.plot(n_trees,plt_valid, label = 'Validation Accuracy')
    plt.legend()
    plt.show()
    


# In[5]:


dep=[1,2,3]
boost_iter=[10,20,40,60,80,100]
for depth in dep:
    base = DecisionTreeClassifier(max_depth=depth)
    plt_valid=[]
    plt_train=[]
    plt_test=[]
    print("For Depth =" ,depth)
    print()
    print()
    for no_iter in boost_iter:
        print("For number of iterations = ", no_iter)
        clf=AdaBoostClassifier(base_estimator = base ,n_estimators= no_iter)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_train)
        plt_train.append(accuracy_score(y_pred,y_train)*100)
        print ("Accuracy score for training data ",accuracy_score(y_pred,y_train)*100)
        y_pred=clf.predict(X_dev)
        print ("Accuracy score for Validation data ",accuracy_score(y_pred,y_dev)*100)
        plt_valid.append(accuracy_score(y_pred,y_dev)*100)
        y_pred=clf.predict(X_test)
        print ("Accuracy score for Testing data ",accuracy_score(y_pred,y_test)*100)
        plt_test.append(accuracy_score(y_pred,y_test)*100)
        print()
    print()
    plt.xlabel("Boosting iteration")
    plt.ylabel("Accuracy")
    plt.plot(boost_iter,plt_train,label = 'Training Accuracy')
    plt.plot(boost_iter,plt_test, label = 'Testing Accuracy')
    plt.plot(boost_iter,plt_valid, label = 'Validation Accuracy')
    plt.legend()
    plt.show()
    


# In[6]:


def bagging(max_depth,n_estimators):
    base = DecisionTreeClassifier(max_depth=int(max_depth))
    clf=BaggingClassifier(base_estimator = base ,n_estimators=int(n_estimators))
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_dev)
    score =accuracy_score(y_pred,y_dev)*100
    return score
         


# In[ ]:


optimizer = BayesianOptimization(bagging,{"max_depth" :(1,20),
             "n_estimators":(1,100)})
optimizer.maximize(n_iter=50)


# In[92]:


max=0
bagging_hyperparameter=[]
for i in optimizer.Y:
    if max<i:
        bagging_hyperparameter.append(i)
        max=i
print("The number of hyperparameters that bayesian optimization chose was ",len(bagging_hyperparameter))
print("and the values are")
print(bagging_hyperparameter)


# In[93]:



x=list(range(1,56))
plt.xlabel("BO iteration")
plt.ylabel("Performance")
plt.plot(x,optimizer.Y,label = 'Performance')
plt.legend()
plt.show()


# In[94]:


def boosting(max_depth,n_estimators):
    base = DecisionTreeClassifier(max_depth=int(max_depth))
    clf=AdaBoostClassifier(base_estimator = base ,n_estimators=int(n_estimators))
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_dev)
    score =accuracy_score(y_pred,y_dev)*100
    return score


# In[95]:


optimizer = BayesianOptimization(boosting,{"max_depth" :(1,20),
             "n_estimators":(1,100)})
optimizer.maximize(n_iter=50)


# In[96]:


max=0
boosting_hyperparameter=[]
for i in optimizer.Y:
    if max<i:
        boosting_hyperparameter.append(i)
        max=i
print("The number of hyperparameters that bayesian optimization chose was ",len(boosting_hyperparameter))
print("and the values are")
print(boosting_hyperparameter)


# In[97]:


x=list(range(1,56))
plt.xlabel("BO iteration")
plt.ylabel("Performance")
plt.plot(x,optimizer.Y,label = 'Performance')
plt.legend()
plt.show()

