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
    "### Membuang fitur yang tidak diperlukan"
    data['fruit_name'].value_counts()
    X=data.drop(columns=['fruit_name','fruit_subtype'],axis=1)

    X
    
    x = data[["mass","width","height","color_score"]]
    y = data["fruit_label"].values
    
    st.write("""# Normalisasi MinMaxScaler""")
    "###Mengubah skala nilai terkecil dan terbesar dari dataset ke skala tertentu.pada dataset ini skala terkecil =0, skala terbesar=1"
    
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    x_scaled= scaler.fit_transform(x)
    x_scaled
with tab3:
    x_train, x_test,y_train,y_test= train_test_split(x,y,random_state=0)    
    x_train_scaled, x_test_scaled,y_train_scaled,y_test_scaled= train_test_split(x_scaled,y,random_state=0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    # from sklearn.feature_extraction.text import CountVectorizer
    # cv = CountVectorizer()
    # x_train = cv.fit_transform(x_train)
    # x_test = cv.fit_transform(x_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")
    

   

   

   
