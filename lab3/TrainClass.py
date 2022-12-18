
import pandas as pd
from matplotlib import pyplot
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


class TrainClass(ExcelClass):
    # 80% train data, 20% test data
    train_test_ratio = 4/5
    results = []
    names = []
    models = []

    def __init__(self, path, sep, names):
        super().__init__(path, sep, names)
        self.train_data = self.dataset.head(int(self.len() * self.train_test_ratio))
        self.test_data = self.dataset.tail(int(self.len() * 1-self.train_test_ratio))
        X = self.dataset.values[:, 0:4]
        y = self.dataset.values[:, 4]
        self.X_train, self.X_validation, self.Y_train, self.Y_validation \
            = train_test_split(X, y, test_size=1-self.train_test_ratio, random_state=1)
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
