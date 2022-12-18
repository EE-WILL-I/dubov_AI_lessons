
from TrainClass import TrainClass


t = TrainClass('iris.csv', ',', names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])
t.evaluate_models()
