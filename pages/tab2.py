import hydralit as st
import pickle
from matplotlib import pyplot as plt

def run():
    html_temp = """ 
    <div style ="background-color:black;padding:1px"> 
    <h3 style ="color:white;text-align:center;">Overall feature importance, confusion matrix</h3> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.title("Model Visualization")

    pickle_in = open('classifier.pkl', 'rb') 
    classifier = pickle.load(pickle_in)
    print(type(classifier))
    feature_importance = classifier.feature_importance_
    print(feature_importance)
    feature_names = ["Gender", "Married", "ApplicantIncome", "LoanAmount", "Credit_History"]
    st.bar_chart(feature_importance, columns = feature_names)

