
from DataClass import DataClass

d = DataClass()
d.load('traffic_data.txt')
d.split_data()
d.train_regressor()
d.evaluate_regressor()
d.test()
d.predict()
