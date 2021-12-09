from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

reviews_train = load_files("./aclImdb/train/")
reviews_test = load_files("./aclImdb/test/")

x_train, y_train = reviews_train.data, reviews_train.target
x_test, y_test = reviews_test.data, reviews_test.target

x_train = [doc.replace(b"<br />", b" ") for doc in x_train]
x_test = [doc.replace(b"<br />", b" ") for doc in x_test]


vect1 = CountVectorizer().fit(x_train + x_test)
vect2 = CountVectorizer(min_df=5).fit(x_train + x_test)
vect3 = CountVectorizer(stop_words="english").fit(x_train + x_test)
vect4 = CountVectorizer(stop_words="english", min_df=5).fit(x_train + x_test)

bag_of_word_train_1 = vect1.transform(x_train)
bag_of_word_test_1 = vect1.transform(x_test)

bag_of_word_train_2 = vect2.transform(x_train)
bag_of_word_test_2 = vect2.transform(x_test)

bag_of_word_train_3 = vect3.transform(x_train)
bag_of_word_test_3 = vect3.transform(x_test)

bag_of_word_train_4 = vect4.transform(x_train)
bag_of_word_test_4 = vect4.transform(x_test)

logreg1 = LogisticRegression(max_iter=1000).fit(bag_of_word_train_1, y_train)
logreg2 = LogisticRegression(max_iter=1000).fit(bag_of_word_train_2, y_train)
logreg3 = LogisticRegression(max_iter=1000).fit(bag_of_word_train_3, y_train)
logreg4 = LogisticRegression(max_iter=1000).fit(bag_of_word_train_4, y_train)

print("Bag of words : {:.3f}".format(logreg1.score(bag_of_word_test_1, y_test)))
print("Bag of words(mid_df=5) : {:.3f}".format(logreg2.score(bag_of_word_test_2, y_test)))
print("Bag of words(stop words removal) : {:.3f}".format(logreg3.score(bag_of_word_test_3, y_test)))
print("Bag of words(stop words removal, mid_df=5) : {:.3f}".format(logreg4.score(bag_of_word_test_4, y_test)))