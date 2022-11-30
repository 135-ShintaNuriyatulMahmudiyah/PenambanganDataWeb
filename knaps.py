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
    "### Mengubah skala nilai terkecil dan terbesar dari dataset ke skala tertentu.pada dataset ini skala terkecil = 0, skala terbesar= 1"
    
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
    
    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(x_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(x_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    # prediction
    dt.score(x_test, y_test)
    y_pred = dt.predict(x_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)
with tab4:
    st.write("# Implementation")
    mass = st.number_input('Masukkan berat buah')

    # width
    width = st.number_input('Masukkan lebar buah')
    

    # height
    width = st.number_input('Masukkan tinggi buah')

    #color_score
    color_score = st.number_input('Masukkan nilai warna buah')
def submit():
        # input
        inputs = np.array([[mass,width,height,color_score]])
        le = joblib.load("le.save")
        model1 = joblib.load("knn.joblib")
        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang Anda masukkan, maka anda dinyatakan : {le.inverse_transform(y_pred3)[0]}")

all = st.button("Submit")
    if all :
        st.balloons()
        submit()

    

   

   

   
