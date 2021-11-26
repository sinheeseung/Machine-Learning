from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
x, y = mnist['data'], mnist['target']
x_train, x_test, y_train, y_test = train_test_split(x, y)

model_1 = LinearSVC(C=0.1)
model_1 = model_1.fit(x_train, y_train)
model1_pred = model_1.predict(x_train)
print("kernel = linear, C = 0.1 의 모델 성능 평가 :{:.4f}".format(np.mean(model1_pred == y_train)))

model_2 = LinearSVC(C=5)
model_2 = model_2.fit(x_train, y_train)
model2_pred = model_2.predict(x_train)
print("kernel = linear, C = 5 의 모델 성능 평가 :{:.4f}\n".format(np.mean(model2_pred == y_train)))

model_3 = SVC(kernel='rbf',C=0.1)
model_3 = model_3.fit(x_train, y_train)
model3_pred = model_3.predict(x_train)
print("kernel = rbf, C = 0.1 의 모델 성능 평가: {:.4f}".format(np.mean(model3_pred == y_train)))

model_4 = SVC(kernel='rbf', C=5)
model_4 = model_4.fit(x_train, y_train)
model4_pred = model_4.predict(x_train)
print("kernel = rbf, C = 5 의 모델 성능 평가 :{:.4f}\n".format(np.mean(model4_pred == y_train)))

model_5 = SVC(kernel='sigmoid',C=0.1)
model_5 = model_5.fit(x_train, y_train)
model5_pred = model_5.predict(x_train)
print("kernel = sigmoid, C = 0.1 의 모델 성능 평가 :{:.4f}".format(np.mean(model5_pred == y_train)))

model_6 = SVC(kernel='sigmoid', C=5)
model_6 = model_6.fit(x_train, y_train)
model6_pred = model_6.predict(x_train)
print("kernel = sigmoid, C = 5 의 모델 성능 평가 :{:.4f}\n".format(np.mean(model6_pred == y_train)))

result = model_4.predict(x_test)
print("kernel = rbf, C = 5 의 test set 정화도 :{:.2f}".format(np.mean(result == y_test)))