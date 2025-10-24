import streamlit as st
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

st.title("Simple Classifier App")

st.write("""
# Example app
## This is a simple Streamlit application.
""")

dataset_name = st.sidebar.selectbox("select dataset", ("Iris", "Breast cancer", "wine Dataset"))

classifier_name = st.sidebar.selectbox("select classifier",("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if(dataset_name == "Iris"):
        data = datasets.load_iris()
    elif(dataset_name == "Breast cancer"):
        data = datasets.load_breast_cancer()
    else: 
        data = datasets.load_wine()
    X = data.data
    Y = data.target
    return X, Y
X, Y = get_dataset(dataset_name)
st.write("shape of dataset", X.shape)
st.write("number of calsses", len(np.unique(Y)))

def add_paramenter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 2, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params
params = add_paramenter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params["K"])
    elif clf_name == "SVM":
        clf = SVC(C = params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth = params["max_depth"], random_state = 1234)
    return clf
clf = get_classifier(classifier_name, params)

#Classification
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")

# plot
pca = PCA(2)
x_projected = pca.fit_transform(X)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=Y, alpha=0.8, cmap = "viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

# plt.show() 
st.pyplot(fig)