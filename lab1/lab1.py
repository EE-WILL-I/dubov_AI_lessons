
from ExcelClass import ExcelClass
# lab 1 tasks
e = ExcelClass('excel1.csv', ';', [])
e.sort('res', True)
e.sort('val1', False)
print('row 2 is: ', e.get_row(2))
print('sub-set:\n', e.get_sub_set('res', 6))
print('row count: ', e.len())
e.save('excel2.csv')
