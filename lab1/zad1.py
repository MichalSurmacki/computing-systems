import threading
import glob
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold

datasets = []
for file in glob.glob("./datasets/*.csv"):
    dataset = np.genfromtxt(file, delimiter=",")
    datasets.append(dataset)

adaClf = AdaBoostClassifier()
gaussClf = GaussianNB()

kf_2x5 = RepeatedKFold(n_splits=2, n_repeats=5, random_state=12345)

results = []

for dataset in datasets:

    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)    

    gaussianScores = []
    adaBoostScores = []

    for train_index, test_index in kf_2x5.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        adaClf.fit(X_train, y_train)
        predictAda = adaClf.predict(X_test)
        adaBoostScores.append(accuracy_score(y_test, predictAda))

        gaussClf.fit(X_train, y_train)
        predictGauss = gaussClf.predict(X_test)
        gaussianScores.append(accuracy_score(y_test, predictGauss))

    results.append([np.mean(gaussianScores), np.mean(adaBoostScores)])
    print([np.mean(gaussianScores), np.mean(adaBoostScores)])

print(results)