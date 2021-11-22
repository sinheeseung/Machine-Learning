from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

df = pd.read_csv('iris_mod.csv', header=None)
data = np.array(df, dtype='float64')
target = np.array(data[:,4], dtype='int64')

(h,w) = np.shape(data)
data_nor = np.ones((h,w-1)) # 정규화 값을 저장 할 배열 생성

for i in range(3):
    #정규화 과정
    data_mean = np.mean(data[:,i]) # data set의 평균을 구함
    data_var = np.var(data[:,i]) # data set의 분산을 구함
    data_std = np.sqrt(data_var) # data set의 표준편차를 구함
    data_nor[:,i] = (data[:,i] - data_mean) / data_std
    #정규화 작업을 수행함

knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1_nor = KNeighborsClassifier(n_neighbors=1)

x_train,x_test,y_train,y_test = train_test_split(data[:,0:3],target, random_state=0)
x_train_nor,x_test_nor,y_train_nor,y_test_nor = train_test_split(data_nor,target, random_state=0)

knn_1.fit(x_train, y_train)
knn_1_nor.fit(x_train_nor, y_train_nor)

print('K = {:d}일 때 normaliation 수행 전 테스트 셑의 정확도 : {:.2f}'.format(1,knn_1.score(x_test,y_test)))
print('K = {:d}일 때 normaliation 수행 후 테스트 셑의 정확도 : {:.2f}'.format(1,knn_1_nor.score(x_test_nor,y_test_nor)))

