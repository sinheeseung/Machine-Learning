import pandas as pd
import numpy as np

df = pd.read_csv('small_iris.csv', header=None)
data = np.array(df, dtype='float32')
vector = np.zeros((3,5))
for i in range(len(data)):
    if data[i][4] == 1:
        vector[0][0] = vector[0][0] + data[i][0]
        vector[0][1] = vector[0][1] + data[i][1]
        vector[0][2] = vector[0][2] + data[i][2]
        vector[0][3] = vector[0][3] + data[i][3]
        vector[0][4] += 1
    if data[i][4] == 2:
        vector[1][0] = vector[1][0] + data[i][0]
        vector[1][1] = vector[1][1] + data[i][1]
        vector[1][2] = vector[1][2] + data[i][2]
        vector[1][3] = vector[1][3] + data[i][3]
        vector[1][4] += 1
    if data[i][4] == 3:
        vector[2][0] = vector[2][0] + data[i][0]
        vector[2][1] = vector[2][1] + data[i][1]
        vector[2][2] = vector[2][2] + data[i][2]
        vector[2][3] = vector[2][3] + data[i][3]
        vector[2][4] += 1

for i in range(3):
    print('클래스 {}의 평균 벡터는 [{:.2f},{:.2f},{:.2f},{:.2f}]'.format(i+1, vector[i][0]/vector[i][4], vector[i][1]/vector[i][4], vector[i][2]/vector[i][4], vector[i][3]/vector[i][4]))