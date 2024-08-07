import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

##-------------------------------------------------------------Pickle----------------------------------------------------------------------------------------##
with open("D:/project/.venv/INSURANCE_FINAL_PROJECT/prediction.pkl",'rb') as file:
    insurance_model = pickle.load(file)
with open("D:/project/.venv/INSURANCE_FINAL_PROJECT/Cat_Columns_Encoded_value.json",'rb') as file:
    encode_file = json.load(file)

st.set_page_config(page_title='Insurance Customer conversion', layout="wide")
st.title(':green[*Insurance Customer Conversion prediction Model By Abdul Haleem*]')

def home():
    col1,col2=st.columns(2)
    with col1:
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("## :orange[*Overview*] : Build Classification model to predict the Customer conversion. The dataset provided contains information about a seriesof marketing calls made to potential customers by an insurance company.")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("## :blue[*Technologies used*] : Python, Pandas, Numpy, Pickel, Matplotlib, Seaborn, Scikit-learn, Streamlit.")
    with col2:
        col2.markdown("# ")
        col2.markdown("# ")
        st.image(Image.open(r'D:/project/.venv/INSURANCE_FINAL_PROJECT/insurance_1.jpg'),width=400)
        col2.markdown("# ")
        col2.markdown("# ")
        col2.markdown("# ")
        col2.markdown("# ")
        col2.markdown("# ")
        col2.markdown("# ")
        col2.markdown("# ")
        col2.markdown("# ")
        st.image(Image.open(r'D:/project/.venv/INSURANCE_FINAL_PROJECT/insurance_2.jpg'),width=400)

def model_prediction():
    job_dict={'management':4,'technician':9,'entrepreneur':2,'retired':5,'admin.':0,'services':7,'blue-collar':1,'self-employed':6,'unemployed':10,'housemaid':3,'student':8}
    marital_dict={'married':1, 'single':2, 'divorced':0}
    Educatio_dict={'tertiary':2, 'secondary':1, 'primary':0}
    call_type_dict={'unknown':2, 'cellular':0, 'telephone':1}
    Mon_dict={'may':8,'jun':6,'jul':5,'aug':1,'oct':10,'nov':9,'dec':2,'jan':4,'feb':3,'mar':7,'apr':0,'sep':11}
    prev_outcome_dict={'unknown':3, 'failure':0, 'other':1, 'success':2}
    
    
    with st.form("Regression"):
        col1,col2,col3=st.columns([0.5,0.2,0.5])
        with col1:
            age=st.number_input("Select the**Cutomer Age**",min_value=18,max_value=95,step=1)
            job=st.selectbox("Select the **Job type**",encode_file['job_initial'])
            marital=st.selectbox("Select tha **marital status**",encode_file['marital_initial'])
            education=st.selectbox("Select the **Education**",encode_file['education_qual_initial'])
            call_type=st.selectbox("Select the **Call Type**",encode_file['call_type_initial'])
        
        with col3:
            day=st.number_input("Select the **Day**",min_value=1,max_value=31,step=1)
            month=st.selectbox("Select the **Month**",encode_file['mon_initial'])
            duration=st.number_input("Select the **Duration**",min_value=0,max_value=4918,step=1)
            Number_call=st.number_input("Select the **Number of call**",min_value=1,max_value=63,step=1)
            Prev_outcome=st.selectbox("Select the **Last Outcome**",encode_file['prev_outcome_initial'])
        
        with col2:
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            st.markdown('Click below button to predict')
            button=st.form_submit_button(label='Predict')
    if button:
        job_encode=job_dict.get(job)
        marital_encode= marital_dict.get(marital)
        education_encode=Educatio_dict.get(education)
        call_type_encode=call_type_dict.get(call_type)
        month_encode=Mon_dict.get(month)
        Prev_outcome_encode=prev_outcome_dict.get(Prev_outcome)

        input_arr=np.array([[age,job_encode,marital_encode,education_encode,call_type_encode,day, month_encode,duration,Number_call,Prev_outcome_encode]])
        Y_pred=insurance_model.predict(input_arr)

        if Y_pred[0]==1:
            st.header( "Customer will take insurance")
        else:
            st.header("Customer will not take insurance")




with st.sidebar:
    option = option_menu("Main menu",['Home','Model Prediction'],
                       icons=["house","cloud-upload","list-task","pencil-square"],
                       menu_icon="cast",
                       styles={"nav-link": {"font-size": "20px", "text-align": "left", "margin": "-2px", "--hover-color": "green"},
                                   "nav-link-selected": {"background-color": "green"}},
                       default_index=0)
if option=='Home':
    home()
if option=='Model Prediction':
    model_prediction()