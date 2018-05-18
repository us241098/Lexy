import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
import pandas as pd
import server



def input_array(filename2,result2):
    df = pd.read_csv(filename2)
    y = df.as_matrix(columns = [result2[0]])
    X = df.as_matrix(columns = [result2[1], result2[2]])
    return X, y



def plot_graph(X, y):
    train_percent = 50.0
    # splitting X and y into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_percent/100,
                                                        random_state=1)

    # create linear regression object
    reg = linear_model.LinearRegression()

    # train the model using the training sets
    reg.fit(X_train, y_train)

    # regression coefficients
    print('Coefficients: \n', reg.coef_)

    # variance score: 1 means perfect prediction
    print('Variance score: {}'.format(reg.score(X_test, y_test)))

    # plot for residual error

    ## setting plot style
    plt.style.use('fivethirtyeight')

    ## plotting residual errors in training data
    plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
                color = "green", s = 10, label = 'Train data')

    ## plotting residual errors in test data
    plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
                color = "blue", s = 10, label = 'Test data')

    ## plotting line for zero residual error
    plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
    ## plotting legend
    plt.legend(loc = 'upper right')
    ## function to show plot
    plt.savefig('./static/pon.png')
    plt.show()



def mainmulti(filename2,result2):
    X, y = input_array(filename2,result2)
    #train_percent = train_percent / 100
    plot_graph(X, y)

if __name__ == '__main__':
    mainmulti()
