import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
data_train = {'X': np.genfromtxt('./data/Q3/data_train_X.csv', delimiter=','),
'y': np.genfromtxt('./data/Q3/data_train_y.csv', delimiter=',')}
data_test = {'X': np.genfromtxt('./data/Q3/data_test_X.csv', delimiter=','),
'y': np.genfromtxt('./data/Q3/data_test_y.csv', delimiter=',')}

print("test size: x: {:d} X {:d} (entry X xi), y: {:d} \n".format(len(data_test['X']), len(data_test['X'][0]), len(data_test['y'])))
print("training size: x: {:d} X {:d} (entry X xi), y: {:d} \n".format(len(data_train['X']),len(data_train['X'][0]), len(data_train['y'])))

# data : (y, X)
def shuffle_data(data):
    random_list = np.random.permutation(len(data[0])) # permuate for all index+1
    # print(random_list)
    X = list(map(lambda x: data[1][x], random_list))
    y = list(map(lambda x: data[0][x], random_list))
    return (y, X)

def split_data(data, num_folds, fold):
    total_len = len(data[0])
    # print(total_len)
    partition = total_len//num_folds
    # print(partition)
    data_fold0 = data[0][(fold-1)*partition: fold*partition]
    data_fold1 = data[1][(fold-1)*partition: fold*partition]
    data_rest0 = data[0][0: (fold-1)*partition] + data[0][fold*partition:]
    data_rest1 = data[1][0: (fold-1)*partition] + data[1][fold*partition:]
    data_fold = (data_fold0, data_fold1)
    data_rest = (data_rest0, data_rest1)
    return data_fold, data_rest


def train_model(data, lambd):
    # Data: (y, X)
    y = np.matrix(data[0]).T
    # print(y)
    X = np.matrix(data[1])
    I = np.matrix(np.identity(X.shape[1]))
    # print(y.shape)
    # print(X.shape)
    # print(I.shape)
    Beta = linalg.inv(X.T * X + lambd * I) * X.T * y
    return Beta

def predict(data, model):
    X = np.matrix(data[1]).T
    return X * model

def loss(data, model):
    #data : (y, X)
    y = np.matrix(data[0]).T
    X = np.matrix(data[1])
    # print(y.shape)
    # print(X.shape)

    a = y - X * model
    return (a.T * a / y.shape[0]).tolist()[0][0]

def cross_validation(data, num_folds, lambd_seq):
    # data = shuffle_data(data)
    cv_error = []
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0.
        for fold in range(num_folds):
            val_cv, train_cv = split_data(data, num_folds, fold+1)
            # print(train_cv)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error.append(cv_loss_lmd / num_folds)
    return cv_error

def answer(training_data, test_data, lambd_seq):
    # original data
    # training data first
    training_data = shuffle_data(training_data)
    test_data = shuffle_data(test_data)
    training_loss = []
    test_loss = []
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        model = train_model(training_data, lambd)
        loss1 = loss(test_data, model)
        loss2 = loss(training_data, model)
        training_loss.append(loss1)
        test_loss.append(loss2)
    cv_5_fold = cross_validation(training_data, 5, lambd_seq)
    cv_10_fold = cross_validation(training_data, 10, lambd_seq)
    plt.plot(lambd_seq, training_loss, 'r', lambd_seq, test_loss, 'b', lambd_seq, cv_5_fold, 'g', lambd_seq, cv_10_fold, 'y')
    plt.legend(('training', 'test', '5 folds CV', '10 folds CV'),loc='upper right')
    plt.title('Linear Regression Data Error v.s Lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.show()
if __name__ == "__main__":
    # data = ([1,2,3,4,5],[2,4,6,8,10])
    # Beta = train_data(data, 0)
    # predict = predict(data, Beta)
    # print(predict)
    # loss = loss(data, Beta)
    # print(loss)
    lambd_seq = np.linspace(0.02, 1.5, num=50)
    data_train = (data_train['y'], data_train['X'])
    data_test = (data_test['y'], data_test['X'])
    # print(data_train)
    # cv_error = cross_validation(data_train, 5, lambd_seq)
    # print(cv_error)
    answer(data_train, data_test, lambd_seq)
