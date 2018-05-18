import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC

in_file = 'drug.csv'

dataset = pd.read_csv(in_file)
dataset = dataset.fillna(method='ffill')

X = dataset.iloc[:, [2, 3, 4,5,6,7,8,9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20, 21, 22, 23, 24, 25]].values
y = dataset.iloc[:, [1]].values

def split():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print y_test.ravel()
    return X_train, X_test, y_train, y_test


def knn_clasify(X_train, X_test, y_train):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train.ravel())

    y_predicted = knn.predict(X_test)
    return y_predicted

def knn_efficiency(y_predicted, y_test):
    y_test = y_test.ravel()
    total = len(y_predicted)
    i = 0
    non_deflect = 0
    while i < total:
        if y_test[i] == y_predicted[i]:
            non_deflect = non_deflect + 1
        i = i + 1
    print "\n \n KNN Efficiency"
    print float(non_deflect) / float(total)

def linear_regression(X_train, X_test, y_train, y_test):
    regr = LinearRegression()
    regr.fit(X_train, y_train.ravel())
    print "\n \n Linear regression Variance"
    print regr.score(X_test, y_test)

def logistic_regression(X_train, X_test, y_train):
    logistic = LogisticRegression()
    logistic.fit(X_train, y_train.ravel())

    y_predicted = logistic.predict(X_test)
    return y_predicted

def logistic_regression_efficiency(y_predicted, y_test):
    y_test = y_test.ravel()
    total = len(y_predicted)
    i = 0
    non_deflect = 0
    while i < total:
        if y_test[i] == y_predicted[i]:
            non_deflect = non_deflect + 1
        i = i + 1
    print "\n \n Logistic Regression Efficiency"
    print float(non_deflect) / float(total)

def svm(X_train, X_test, y_train):
    svc = SVC(kernel='linear')
    svc.fit(X_train, y_train.ravel())

    y_predicted = svc.predict(X_test)
    return y_predicted

def svm_efficiency(y_predicted, y_test):
    y_test = y_test.ravel()
    total = len(y_predicted)
    i = 0
    non_deflect = 0
    while i < total:
        if y_test[i] == y_predicted[i]:
            non_deflect = non_deflect + 1
        i = i + 1
    print "\n \n SVM Efficiency"
    print float(non_deflect) / float(total)

def svm_kernal(X_train, X_test, y_train, k):
    svc = SVC(kernel='poly', degree=k)
    svc.fit(X_train, y_train.ravel())

    y_predicted = svc.predict(X_test)
    non_deflect = 0
    return y_predicted

def svm_kernal_efficiency(y_predicted, y_test, k):
    y_test = y_test.ravel()
    total = len(y_predicted)
    i = 0
    while i < total:
        if y_test[i] == y_predicted[i]:
            non_deflect = non_deflect + 1
        i = i + 1
    print "\n \n SVM Efficiency for " + str(k) + ' degree poly'
    print float(non_deflect) / float(total)

def svm_rbf(X_train, X_test, y_train):
    svc = SVC(kernel='rbf')
    svc.fit(X_train, y_train.ravel())

    y_predicted = svc.predict(X_test)
    return y_predicted

def svm_rbf_efficiency(y_predicted, y_test):
    y_test = y_test.ravel()
    total = len(y_predicted)
    i = 0
    non_deflect = 0
    while i < total:
        if y_test[i] == y_predicted[i]:
            non_deflect = non_deflect + 1
        i = i + 1
    print "\n \n Radial Basis Function Efficiency"
    print float(non_deflect) / float(total)

def main():
    X_train, X_test, y_train, y_test = split()

#    y_predicted = knn_clasify(X_train, X_test, y_train)
#    knn_efficiency(y_predicted, y_test)

    linear_regression(X_train, X_test, y_train, y_test)

    y_predicted = logistic_regression(X_train, X_test, y_train)
    logistic_regression_efficiency(y_predicted, y_test)

    y_predicted = svm(X_train, X_test, y_train)
    svm_efficiency(y_predicted, y_test)

    k = 1
    while k < 10:
        y_predicted = svm_kernal(X_train, X_test, y_train, k)
        svm_kernal_efficiency(y_predicted, y_test, k)
        k = k + 1

    y_predicted = svm_rbf(X_train, X_test, y_train)
    svm_rbf_efficiency(y_predicted, y_test)

if __name__ == '__main__':
    main()
