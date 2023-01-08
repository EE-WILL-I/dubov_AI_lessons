
from lab1.ExcelClass import ExcelClass
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from sklearn.cluster import estimate_bandwidth, MeanShift
from itertools import cycle


class TrainClass(ExcelClass):
    # 80% train data, 20% test data
    train_test_ratio = 4/5
    results = []
    names = []
    models = []
    X = []
    data = []
    radius = 0
    bandwidth = 0

    def __init__(self, path, sep, names):
        super().__init__(path, sep, names)
        self.train_data = self.dataset.head(int(self.len() * self.train_test_ratio))
        self.test_data = self.dataset.tail(int(self.len() * 1-self.train_test_ratio))
        self.X = self.dataset.values[:, 0:4]
        y = self.dataset.values[:, 4]
        self.centroids = []
        self.X_train, self.X_validation, self.Y_train, self.Y_validation \
            = train_test_split(self.X, y, test_size=1-self.train_test_ratio, random_state=1)
        print("\ntrain:\n", self.train_data, "\ntest:\n", self.test_data)

    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

    def evaluate_models(self):
        for name, model in self.models:
            kfold = StratifiedKFold(n_splits=3, random_state=1, shuffle=True)
            cv_results = cross_val_score(model, self.X_train, self.Y_train, cv=kfold, scoring='accuracy')
            self.results.append(cv_results)
            self.names.append(name)
            print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    def shuffle(self):
        np.random.shuffle(self.data)

    def get_bandwidth(self):
        self.bandwidth = estimate_bandwidth(self.X, quantile=0.2, n_samples=500)
        print("bandwidth is : ", self.bandwidth)

    def fit(self):
        ms = MeanShift(bandwidth=self.bandwidth, bin_seeding=True)
        ms.fit(self.X)
        labels = ms.labels_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        print("number of estimated clusters : %d" % n_clusters_)

        plt.figure(1)
        plt.clf()

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = ms.cluster_centers_[k]
            plt.plot(self.X[my_members, 0], self.X[my_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
