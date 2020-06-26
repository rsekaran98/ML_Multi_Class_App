# Multiple classficiationML app
# Author : Sekaran Ramalingam
# Company: SeaportAi
# Cleint: for General Client Demo purpose on streamlit
# Date: 22-06-2020

# Importing all the libraries
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Seaport image
img = Image.open("SeaportAI.jpg")
st.sidebar.image(img,width=70,caption  = ' ')

# Title on HTML
title = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Multiple Classfication ML App</h2>
    </div>
    """
st.markdown(title,unsafe_allow_html=True)

# Notes
st.write("""
## :joy: -------------------------------------------------------:heart: 
#### Author: Sekaran Ramalingam - SeaportAI
#### I have used Iris, Breast Cancer, Wine datasets on KNN, SVM, Random Forest classifiers... !!!

#### "Have used Streamlit caching, so datasets will not reload on every page refresh...Hurray"

""")

# Dataset & Classfier selection
dataset_name  = st.sidebar.selectbox("Select the Dataset...",("Iris","Breast Cancer","Wine"))

class_name    = st.sidebar.selectbox("Select the Classifier...",("KNN","SVM","Random Forest"))

@st.cache
def filter_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X  = data.data
    y  = data.target

    return X,y

X,y = filter_dataset(dataset_name)

if dataset_name == "Iris":
    data = datasets.load_iris()
elif dataset_name == "Breast Cancer":
    data = datasets.load_breast_cancer()
else:
    data = datasets.load_wine()

df1 = pd.DataFrame(data.data, columns=data.feature_names)
df1['target'] = pd.Series(data.target)
 

st.write("\n")
st.write("Shape of the dataset is :",X.shape)
st.write("Number of targets :",len(np.unique(y)))
st.write("Targets names are :", data.target_names)

# Display the Selected dataset
show_data = st.checkbox("Display the dataset")
if show_data:
    st.write(df1)

# Parameter setting for the selected clasiifier
def set_params(class_name):
    d_params  = dict()
    if class_name == "KNN":
        n_neighbors = st.sidebar.slider("Select the 'n_neighbors' value ",1,15)
        d_params["n_neighbors"] = n_neighbors
    elif class_name == "SVM":
        C = st.sidebar.slider("Select the 'C - Regularization' value ",0.01,10.0)
        d_params["C"] = C
    elif class_name == "Random Forest":
        n_estimators = st.sidebar.slider("Select the 'n_estimators' value ",1,100) # no of trees
        max_depth    = st.sidebar.slider("Select the 'max_depth' value ",2,15) # no of depth for each tree
        d_params["n_estimators"] = n_estimators
        d_params["max_depth"]    = max_depth
    return d_params


params = set_params(class_name)

# Setting up the selected classifier
def get_classifier(class_name,params):
    if class_name == "KNN":
        v_clf  =  KNeighborsClassifier(n_neighbors = params["n_neighbors"])
    elif class_name == "SVM":
        v_clf  = SVC(C = params["C"])
    elif class_name == "Random Forest":
        v_clf  = RandomForestClassifier(n_estimators = params["n_estimators"],max_depth = params["max_depth"], random_state  = 111)
    return v_clf


clf = get_classifier(class_name,params)

# Dataset split
X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size  = 0.2, random_state = 111)

# Model fit
clf.fit(X_train, y_train)

# prediction
y_pred  = clf.predict(X_test)

# Accuracy
accu  = accuracy_score(y_test, y_pred)

st.write(f"Classfier Name : {class_name}")

st.write(f"Accuracy Level : {accu}")

# Party balloons
ball  = st.checkbox("Check for  party time...!!!")
if ball:
    st.balloons()


plt_text = """
    <body style="background-color:powderblue;">
        <p style="color:blue";"font-size:20">Dataset Plotting</p>
    </body>

    """
st.markdown(plt_text,unsafe_allow_html=True)

# 2D plotting
pca  = PCA(2)
X_projected = pca.fit_transform(X)
x1  = X_projected[:,0]
x2  = X_projected[:,1]

fig  = plt.figure()
plt.scatter(x1,x2,c=y,alpha = 0.8,cmap = "viridis")
plt.xlabel('Priniciple Component 1')
plt.ylabel('Priniciple Component 2')
plt.colorbar()
st.pyplot()


