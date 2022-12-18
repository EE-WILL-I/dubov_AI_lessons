
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor


class DataClass:
    data = []
    label_encoder = []
    X = []
    y = []
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    y_pred = []
    regressor = ExtraTreesRegressor
    test_datapoint_encoded = []

    def load(self, path):
        input_file = 'traffic_data.txt'
        data = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                items = line[:-1].split(',')
                data.append(items)
        self.data = np.array(data)
        self.string_to_numeric()

    def string_to_numeric(self):
        X_encoded = np.empty(self.data.shape)
        for i, item in enumerate(self.data[0]):
            if item.isdigit():
                X_encoded[:, i] = self.data[:, i]
            else:
                self.label_encoder.append(preprocessing.LabelEncoder())
                X_encoded[:, i] = self.label_encoder[-1].fit_transform(self.data[:, i])
        self.X = X_encoded[:, :-1].astype(int)
        self.y = X_encoded[:, -1].astype(int)

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_test_split(self.X, self.y, test_size=0.15, random_state=5)

    def train_regressor(self):
        params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
        self.regressor = ExtraTreesRegressor(**params)
        self.regressor.fit(self.X_train, self.y_train)

    def evaluate_regressor(self):
        self.y_pred = self.regressor.predict(self.X_test)
        print("Average abs error:", round(mean_absolute_error(self.y_test, self.y_pred), 2))

    def test(self):
        test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
        test_datapoint_encoded = [-1] * len(test_datapoint)
        count = 0
        for i, item in enumerate(test_datapoint):
            if item.isdigit():
                test_datapoint_encoded[i] = int(test_datapoint[i])
            else:
                test_datapoint_encoded[i] = int(self.label_encoder[count].transform([test_datapoint[i]]))
                count = count + 1
        self.test_datapoint_encoded = np.array(test_datapoint_encoded)

    def predict(self):
        print("Predicted traffic:", int(self.regressor.predict([self.test_datapoint_encoded])[0]))
