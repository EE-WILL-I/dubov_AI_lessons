
import pandas as pd
from matplotlib import pyplot


class ExcelClass:
    def __init__(self, path, sep, names):
        if len(names) > 0:
            self.dataset = pd.read_csv(path, names=names, sep=sep)
        else:
            self.dataset = pd.read_csv(path, sep=sep)
        print('data:\n', self.dataset)

    def sort(self, col, asc):
        self.dataset.sort_values(by=[col], ascending=asc)
        print('sorted', col, 'by', 'asc\n' if asc else 'desc\n', self.dataset)

    def get_row(self, ind):
        return self.dataset.iloc[ind]

    def get_sub_set(self, col, cond):
        return self.dataset.loc[self.dataset[col] > cond]

    def get_dim(self):
        print(self.dataset.shape)

    def len(self):
        return len(self.dataset)

    def describe(self):
        print(self.dataset.describe())

    def get_instances(self, classifier):
        print(self.dataset.groupby(by=classifier, axis=0, level=None, sort=True, group_keys=True, dropna=True).size())

    def draw_property(self, layout_x, layout_y):
        # kde requires scipy lib
        self.dataset.plot(kind='kde', subplots=True, layout=(layout_x, layout_y), sharex=False, sharey=False)
        pyplot.show()

    def draw_scatter_matrix(self):
        # kde requires scipy lib
        pd.plotting.scatter_matrix(self.dataset, alpha=0.5, figsize=None, grid=False, diagonal='kde', marker='.')
        pyplot.show()

    def save(self, name):
        self.dataset.to_csv(name, index=False)
