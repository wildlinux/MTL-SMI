import numpy as np

# data = np.random.random(size=10)
data1 = np.asarray(a=[[i] * 5 for i in range(20)])
data2 = np.asarray(a=[[i] * 5 for i in range(40, 50)])
data3 = np.random.choice(a=data2.shape[0], size=data1.shape[0])
# replace:True表示可以取相同数字，False表示不可以取相同数字
data4 = np.random.choice(a=data2.shape[0], size=data1.shape[0], replace=True)

print(data1, '\n')
print(data2, '\n')
print(data3, '\n')
print(data4, '\n')
