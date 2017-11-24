import os
from subprocess import call
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import time
from keras import backend as K

objects_per_batch = 1000
batch_num = 5
classes_num = 10
image_size = 32
K.set_image_data_format('channels_last')


def check_download_dataset():
    if not os.path.exists("cifar-10-python.tar.gz"):
        print("Downloading CIFAR-10 dataset")
        call("wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", shell=True)
        print("Dataset downloaded")
    else:
        print("Dataset already downloaded")


def check_extract_dataset():
    cifar_python_directory = os.path.abspath("cifar-10-batches-py")
    if not os.path.exists(cifar_python_directory):
        print("Extracting dataset")
        call("tar -zxvf cifar-10-python.tar.gz", shell=True)
        print("Dataset extracted")
    else:
        print("Dataset already extracted")


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def process_dict(datadict):
    data = datadict[b'data'][:objects_per_batch]
    data_norm = data.reshape(objects_per_batch, 3, image_size, image_size).transpose(0, 2, 3, 1).astype("uint8")
    labels = datadict[b'labels'][:objects_per_batch]
    return data_norm, labels


def get_data_from_batch(batch_no):
    datadict = unpickle("cifar-10-batches-py/data_batch_" + str(batch_no))
    return process_dict(datadict)


def get_data_from_test_batch():
    datadict = unpickle("cifar-10-batches-py/test_batch")
    return process_dict(datadict)


def get_label_names():
    labeldict = unpickle("cifar-10-batches-py/batches.meta")
    return labeldict[b'label_names']


def hyperparameter_optimization(clf, params, X, Y):
    print("Optimizing hyperparameters")
    grid = GridSearchCV(clf, params, cv=5)
    start = time.time()
    grid.fit(X, Y)

    # evaluate the best grid searched model on the testing data
    print("[INFO] grid search took {:.2f} seconds".format(
        time.time() - start))
    print("[INFO] grid search best parameters: {}".format(
        grid.best_params_))

    return grid.best_estimator_


def print_accuracy_report(expected, predicted, test=1):
    if test == 1:
        print("====================Test data====================")
    else:
        print("====================Train data====================")
    print(metrics.classification_report(expected, predicted))
    print("Confusion matrix:")
    print(metrics.confusion_matrix(expected, predicted))
    print("Accuracy: " + str(metrics.accuracy_score(expected, predicted)))


def run_svm(X, Y, XT, YT, params=None):
    print("Run SVM")
    clf = svm.LinearSVC()

    if params is not None:
        clf.set_params(params)
        clf.fit(X, Y)
    else:
        params = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}
        clf = hyperparameter_optimization(clf, params, X, Y)

    expected = Y
    predicted = clf.predict(X)
    print_accuracy_report(expected, predicted, test=0)

    expected = YT
    predicted = clf.predict(XT)
    print_accuracy_report(expected, predicted, test=1)
