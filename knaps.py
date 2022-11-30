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

st.title("Penambangan Data Web")

st.write("=========================================================================")

st.write("Name :Shinta Nuriyatul Mahmudiyah")
st.write("Nim  :200411100135")
st.write("Grade: Penambangan Data A")
tab1,tab2,tab3,tab4 = st.tabs(["Upload Data", "Prepocessing", "Modeling", "Implementation"])
with tab1:
    st.write("""# Upload File""")
    st.write("Dataset yang digunakan adalah fruit with color dataset yang diambil dari https://www.kaggle.com/datasets/mjamilmoughal/fruits-with-colors-dataset")
    st.write("Total datanya adalah ... dengan data training ..... dan data testing ...... ")
    uploaded_files = st.file_uploader("Upload file TXT", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_table(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)
with tab2:
    st.write("""# Preprocessing""")
    #membuang data yang tidak digunakan
    df=df.drop(['fruit_name','fruit_subtype'],axis=1)
    df.head()
    
    #pisahkan dengan dataset asli
    #Modal x menyimpan fitur kumpulan data kami tanpa label.
    #Huruf kecil y memegang label yang sesuai untuk instance di x.
    x = df[["mass","width","height","color_score"]]
    y = df["fruit_label"].values
    
    #fungsi MinMaxScaler digunakan untuk mengubah skala nilai terkecil dan terbesar dari dataset ke skala tertentu.pada dataset ini skala terkecil =0, skala terbesar=1
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    x_scaled= scaler.fit_transform(x)
    x_scaled
    
    x_train, x_test,y_train,y_test= train_test_split(x,y,random_state=0)
    x_train_scaled, x_test_scaled,y_train_scaled,y_test_scaled= train_test_split(x_scaled,y,random_state=0)
  
   

   

   
