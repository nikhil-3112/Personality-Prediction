#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:15:54 2021

@author: parag
"""

import streamlit as st
#from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
## Importing cox model 

model=keras.models.load_model('model2')

mapping_dict={0:'Dependable',1:'Extraverted',2:'Lively',3:'Responsible',4:'Serious'}
dict1 = {'Strongly Disagree':1,'Disagree' : 3 ,'Neither agree nor disagree' : 5,'Strongly agree' : 7,'Agree' : 9}
#('Strongly Disagree','Disagree','Neither agree nor disagree','Strongly agree','Agree')

def main():
     



    st.header("Personality Prediction")
    col1,col2,col3,col4=st.columns((1,1,2,2))
    with col1:
       # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        rd = st.radio('Gender',options = ['Female','Male'],index = 1)
        if rd=='Female':
            Gender=1
        if rd=='Male':
            Gender=0        
    
    with col2 :
         age=st.text_input("Age",value = 20)
                 
    with col3:
        openness=st.selectbox("Openness",list(dict1.keys()))  
        openness = dict1[openness]
        #st.write(openess)
    with col4:
        neuroticism=st.selectbox("Neuroticism",list(dict1.keys()))
        neuroticism = dict1[neuroticism]
        
    col1,col2,col3=st.columns((2,2,2))
    with col1:
        conscientiousness=st.selectbox("Conscientiousness",list(dict1.keys())) 
        conscientiousness = dict1[conscientiousness]
            
            
    with col2 :
        agreeableness=st.selectbox("Agreeableness",list(dict1.keys())) 
        agreeableness = dict1[agreeableness]

    with col3:
        extraversion=st.selectbox("Extraversion",list(dict1.keys()))    
        extraversion = dict1[extraversion]
            
    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: green;
    
     }
     </style>""", unsafe_allow_html=True)
    
    b = st.button("Predict Personality")
    ### model predicton
    if b:
        
        values=[Gender,int(age),int(openness),int(neuroticism),int(conscientiousness),int(agreeableness),int(extraversion)]
        print("values:",values)
        prediction=model.predict([values])
        prediction=prediction.argmax(axis=1)
        prediction=list(prediction)[0]
        
        prediction=mapping_dict[prediction]
        st.write("Personality of Person is:",prediction)
                    
if __name__ == '__main__':
    main()



    