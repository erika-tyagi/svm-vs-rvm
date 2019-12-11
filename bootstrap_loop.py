import time
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from skrvm import RVC

import svm_rvm_helpers as helpers

# SVM hyperparametes 
PARAM_GRID = {'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000, 10000], 
              'gamma': [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.1, 1, 10], 
              'kernel': ['rbf']}


# split into training and testing 
def prep_data(cancer_df):
    '''
    Takes: cancer dataframe
    Returns: normalized train and test data, ready for modeling
    '''
    
    X = cancer_df.drop(['target'], axis = 1)
    y = cancer_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

    # normalize data
    X_train_scaled = helpers.normalize(X_train)
    X_test_scaled = helpers.normalize(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test


# build simple SVM 
def build_simple_SVM(X_train_scaled, y_train, X_test_scaled, y_test):
    '''
    Takes: normalized training features, training labels, normalized test features, test labels
    Returns: time to train
    '''
    svc_model = SVC(gamma = 'auto')
    print("fitting simple SVM:")
    start = time.time()
    svc_model.fit(X_train_scaled, y_train)
    delta = time.time() - start
    print("seconds to fit simple SVM: ", delta)
    svc_predict = svc_model.predict(X_test_scaled)

    # evaluate simple SVM 
    print(helpers.confusion_matrix(y_test, svc_predict))
    print(classification_report(y_test, svc_predict))

    return delta


def grid_svm(param_grid, X_train_scaled, y_train):
    '''
    Takes: hyperparameter dict, normalized training data, training labels
    Returns: optimized grid, time to fit
    '''
    grid = GridSearchCV(SVC(), param_grid, refit = True, cv = 5)
    print("fitting SVM grid:")
    start = time.time()
    grid.fit(X_train_scaled, y_train)
    delta = time.time() - start
    print("seconds to fit SVM grid: ", delta)

    # print best SVM parameters 
    print("SVM best params:")
    print(grid.best_params_)
    print(grid.best_estimator_)
    print("\n")
    
    return grid, delta


def predict_optimized_SVM(X_test_scaled, y_test, grid):
    '''
    Takes: training data, optimized grid
    Returns: time to predict
    '''
    
    # predict with optimized SVM
    start = time.time()
    grid_predict = grid.predict(X_test_scaled)
    delta = time.time()-start
    print("seconds to predict with optimized SVM: ", delta)
    print(helpers.confusion_matrix(y_test, grid_predict))
    print(classification_report(y_test, grid_predict))
    
    return delta


def build_and_run_rvc(X_train_scaled, y_train, X_test_scaled, y_test):
    '''
    Takes: training and testing data
    Returns: time to fit, time to predict, plus it prints
    '''
        
    # build RVC 
    rvc_model = RVC()
    print("fitting RVM:")
    start = time.time()
    rvc_model.fit(X_train_scaled, y_train)
    delta0 = time.time()-start
    print("time to fit RVM: ", delta0)

    start = time.time()
    rvc_predict = rvc_model.predict(X_test_scaled)
    delta1 = time.time() - start
    print("time to predict with RVM: ", delta1)

    # print parameters
    print("RVM hyperparameters:")
    print(rvc_model.get_params())

    # evaluate RVC
    print(helpers.confusion_matrix(y_test, rvc_predict))
    print(classification_report(y_test, rvc_predict))

    return delta0, delta1


def summarize_runtime(info, delta_list):
    '''
    Takes: string info about run, time deltas
    Returns: dataframe summarizing run
    '''

    df = pd.DataFrame([delta_list])
    df.columns = ['svm_simple_train', 'svm_grid_train', 'svm_predict', 'rvm_train', 'rvm_predict']
    df['run_info'] = info
    cols = ['run_info'] + list(df.columns[:-1])

    return df[cols]


def simple_run(cancer_df, info):
    '''
    Takes: input data frame, string info about run
    Returns: dataframe summarizing runtime
    '''

    X_train_scaled, y_train, X_test_scaled, y_test = prep_data(cancer_df)


    # svm stuff
    svm_t0 = build_simple_SVM(X_train_scaled, y_train, X_test_scaled, y_test)
    grid, svm_tgrid = grid_svm(PARAM_GRID, X_train_scaled, y_train)
    svm_tpredict = predict_optimized_SVM(X_test_scaled, y_test, grid)

    # rvm stuff
    rvm_ttrain, rvm_tpredict = build_and_run_rvc(X_train_scaled, y_train, X_test_scaled, y_test)

    times = [svm_t0, svm_tgrid, svm_tpredict, rvm_ttrain, rvm_tpredict]

    return summarize_runtime(info, times)


if __name__ == "__main__":
    
    cancer = load_breast_cancer()
    cancer_df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], 
                            columns = np.append(cancer['feature_names'], ['target']))

    # first just use on the unmodified dataset
    t = str(round(time.time(), 0))
    results_df = simple_run(cancer_df, t + ": No bootstrap, no noise")

    # explore what happens if we introduce random noise
    for i in [1, 2]:
        
        print("\n")
        print("BEGIN RUN WITH BOOTSTRAP MULTIPLE ", i)
        print("\n")
        
        noisy_cancer_df = helpers.bootstrap_with_noise(cancer_df, i)
        print(noisy_cancer_df.shape)
        noisy_cancer_df.head()

        t = str(round(time.time(), 0))
        out = simple_run(noisy_cancer_df, t + " Bootstrap x " + str(i) + " with Gaussian noise")
        results_df = results_df.append(out)
        
        print("\n")
        print("#======================================#")
    
    results_df.to_csv("results_svm-vs-rvm.csv")



