
from lab3.TrainClass import TrainClass

t = TrainClass('sales.csv', ',', [])
t.shuffle()
t.get_bandwidth()
t.fit()
