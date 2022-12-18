
from lab1.ExcelClass import ExcelClass

e = ExcelClass('iris.csv', ',', names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])
e.get_dim()
print(e.dataset.head(20))
e.describe()
e.get_instances('class')
e.draw_property(2, 2)
e.draw_scatter_matrix()
