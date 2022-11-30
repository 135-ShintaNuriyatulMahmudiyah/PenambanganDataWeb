import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC

st.write(""" 
# Penambangan Data Web
""")

st.write("=========================================================================")

st.write("Name :Shinta Nuriyatul Mahmudiyah")
st.write("Nim  :200411100135")
st.write("Grade: Penambangan Data A")
tab1,tab2,tab3,tab4 = st.tabs(["Upload Data", "Prepocessing", "Modeling", "Implementation"])
with tab1:
    st.write("""# Upload File""")
    st.write("Dataset yang digunakan adalah healthcare-dataset-stroke-data dataset yang diambil dari https://www.kaggle.com/datasets/mjamilmoughal/fruits-with-colors-dataset")
    st.write("Total datanya adalah ... dengan data training ..... dan data testing ...... ")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_table(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)
with tab2:
  
   

   

   
