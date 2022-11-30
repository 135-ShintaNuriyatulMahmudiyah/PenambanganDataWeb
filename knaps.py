import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib

st.title("Fruit With Color")

st.write("=========================================================================")

st.write("Name :Shinta Nuriyatul Mahmudiyah")
st.write("Nim  :200411100135")
st.write("Grade: Penambangan Data A")
tab1, tab2, tab3, tab4 = st.tabs(["Import Data", "Preprocessing", "Modelling", "Implementation"])
with tab1:
    st.write("Import Data")
    st.write("Disini dataset yang saya gunakan adalah Fruit With Color Dataset dengan jumlah data 59. berikut link datasetnya : https://raw.githubusercontent.com/135-ShintaNuriyatulMahmudiyah/Data/main/fruit_data_with_colors.txt ")
    data = pd.read_table("https://raw.githubusercontent.com/135-ShintaNuriyatulMahmudiyah/Data/main/fruit_data_with_colors.txt")
    st.dataframe(data)
with tab2:
    st.write("""# Preprocessing""")
    data['fruit_name'].value_counts()
    X=data.drop(columns=['fruit_name','fruit_subtype'],axis=1)

    X
    
    x = data[["mass","width","height","color_score"]]
    y = data["fruit_label"].values
    
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    x_scaled= scaler.fit_transform(x)
    x_scaled
    

   x.shape, y.shape

   

   
