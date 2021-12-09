import chardet
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

files = load_files("./bbcsport/")
X,y = files.data, files.target

for i in range(len(X)):
    if chardet.detect(X[i]) != "utf-8":
        X[i] = X[i].decode(chardet.detect(X[i])['encoding']).encode('utf8')

X = [doc.replace(b"",b"") for doc in X]
X = [doc.replace(b"\n",b" ") for doc in X]

vectorizer = TfidfVectorizer(norm='l2', stop_words='english')
vectorizer = vectorizer.fit_transform(X).toarray()

print("데이터의 크기: ",vectorizer.shape)

knn_Tfid = KNeighborsClassifier(n_neighbors=1).fit(vectorizer, y)
score_Tfid = cross_val_score(knn_Tfid, vectorizer, y, cv=5)

print("Tfid의 교차 검증 점수:",score_Tfid)

pca_2 = PCA(n_components=2)
pca_10 = PCA(n_components=10)

pca_2 = pca_2.fit(vectorizer)
pca_10 = pca_10.fit(vectorizer)

X_pca2 = pca_2.transform(vectorizer)
X_pca10 = pca_10.transform(vectorizer)

knn_pca2 = KNeighborsClassifier(n_neighbors=1).fit(X_pca2, y)
knn_pca10 = KNeighborsClassifier(n_neighbors=1).fit(X_pca10, y)

score_pca2 = cross_val_score(knn_pca2, X_pca2, y, cv=5)
score_pca10 = cross_val_score(knn_pca10, X_pca10, y, cv=5)

print("PCA2의 교차 검증 점수:",score_pca2)
print("PCA10의 교차 검증 점수:",score_pca10)

