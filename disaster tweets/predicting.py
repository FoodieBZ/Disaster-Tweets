import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection
from nltk.corpus import stopwords
from sklearn import linear_model
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import string
import nltk
from nltk import WordNetLemmatizer 
import re
from nltk.stem import PorterStemmer 
import time
from scipy import spatial


pd.set_option('display.width', 10000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_colwidth', 10000)

def gatherData(file1, file2):
    train_df = remove(pd.read_csv(file1, sep = ","))
    pred_df = remove(pd.read_csv(file2, sep = ","))
    vectorizer = TfidfVectorizer(stop_words = stopwords.words("english"))
    #vectorizer = TfidfVectorizer()
    #vectorizer = CountVectorizer(stop_words = stopwords.words("english"))
    x_train = vectorizer.fit_transform(train_df["text"])
    x_test = vectorizer.transform(pred_df["text"])
    y_train = train_df["target"]
    return y_train, x_train, x_test, train_df

def remove(df):
    array = []
    i = 0
    lem = WordNetLemmatizer()
    p = PorterStemmer() 
    for x in df["text"].values:
        x = x.lower()
        x = x.replace("#", "")
        x = x.translate(str.maketrans("", "", string.punctuation))
        x = re.sub(r"http\S+", "", x)  #word start with http remove until whitespace
        x = nltk.word_tokenize(x)
        #x = " ".join([lem.lemmatize(word) for word in x])
        x = " ".join([p.stem(word) for word in x])
        array.append(x)
    df["text"] = array
    return df

def NearestNeighbor(train, test, k, labels, df):
    dist = []
    kitems = []
    i = 0
    predict = []
    prediction = open("predict.dat","w")
    for x in range(len(test)):
        for z in range(len(train)):
            cos_sym = 1 - spatial.distance.cosine((train[z]), test[x])
            data = [df["id"].values[z], cos_sym]
            dist.append(data)
        kitems = sorted(dist, key = lambda x : x[1], reverse = True)
        pos = 0
        neg = 0
        klimit = 0
        for j in kitems:
            if klimit == k:
                break
            d = df.loc[df["id"] == j[0]] 
            if d["target"].values[0] == 1:
                pos += 1
            else:
                neg += 1
            klimit += 1
        if pos > neg:
            predict.append(1)
            prediction.write("1\n")
        elif pos < neg:
            predict.append(0)
            prediction.write("0\n")
        else:
            predict.append(labels[kitems[i][0]] +"\n") 
            prediction.write(labels[kitems[i][0]] +"\n") 
        docNumber = 0
        dist = []
        print(x)
    prediction.close()
    return predict

def predict_file(file1, y_test):
    predict = pd.read_csv(file1, sep = ",")
    predict["target"] = y_test
    predict.to_csv("predict.csv", index=False)
    
if __name__ == "__main__":
    y_train, x_train, x_test, train_df = gatherData("train.csv", "test.csv")
    #extremely slow still. I have commmented knn out.
    '''
    t = time.time()
    scores = NearestNeighbor(x_train.todense(), x_test.todense(), 1, y_train, train_df)
    print("KNN:\n")
    print(time.time() - t)
    predict_file("sample_submission.csv", scores)
    '''

    print("Starting DecisionTree F1 score:\n")
    t = time.time()
    c = tree.DecisionTreeClassifier(random_state = 0)
    scores = model_selection.cross_val_score(c, x_train, y_train, scoring="f1")
    df_scores = pd.DataFrame(data = sorted(scores, reverse = True), columns = ["DecisionTree"])
    print(time.time() - t)
    print("Finished MultinomialNB\n")

    print("Starting MultinomialNB F1 score\n")
    t = time.time()
    c = MultinomialNB()
    scores = model_selection.cross_val_score(c, x_train, y_train, scoring="f1")
    df_scores["MultinomialNB"] = sorted(scores, reverse = True)
    print(time.time() - t)
    print("Finished MultinomialNB\n")

    print("Starting BernoulliNB F1 score\n")
    t = time.time()
    c = BernoulliNB()
    scores = model_selection.cross_val_score(c, x_train, y_train, scoring="f1")
    df_scores["BernoulliNB"] = sorted(scores, reverse = True)
    print(time.time() - t)
    print("Finished BernoulliNB\n")
    
    print("Starting Perceptron F1 score\n")
    t = time.time()
    c = Perceptron(random_state = 0)
    scores = model_selection.cross_val_score(c, x_train, y_train, scoring="f1")
    df_scores["Perceptron"] = sorted(scores, reverse = True)
    print(time.time() - t)
    print("Finished Perceptron\n")

    print("Starting LogisticRegression F1 score\n")
    t = time.time()
    c = LogisticRegression(random_state = 0)
    scores = model_selection.cross_val_score(c, x_train, y_train, scoring="f1")
    df_scores["LogisticRegression"] = sorted(scores, reverse = True)
    print(time.time() - t)
    print("Finished LRegression\n")

    print("Printing F1 Score table")
    print(df_scores)

    c = BernoulliNB()
    y_test = c.fit(x_train, y_train).predict(x_test)
    predict_file("sample_submission.csv", y_test)






