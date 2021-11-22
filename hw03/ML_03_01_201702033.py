from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

df = pd.read_csv('iris.csv', header=None)
data = np.array(df, dtype='float32')
target = np.array(data[:,4], dtype='int64')

#클래스 객체 생성 , 이웃의 개수 지정
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_10 = KNeighborsClassifier(n_neighbors=10)

x_train,x_test,y_train,y_test = train_test_split(data[:,0:3],target, random_state=0)

knn_1.fit(x_train, y_train)
knn_5.fit(x_train, y_train)
knn_10.fit(x_train, y_train)

print('K = {:d}일 때 테스트 셑의 정확도 : {:.2f}'.format(1,knn_1.score(x_test,y_test)))
print('K = {:d}일 때 테스트 셑의 정확도 : {:.2f}'.format(5,knn_5.score(x_test,y_test)))
print('K = {:d}일 때 테스트 셑의 정확도 : {:.2f}'.format(10,knn_10.score(x_test,y_test)))

