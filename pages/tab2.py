import hydralit as st
import pickle
import altair as alt
import pandas as pd

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

    st.subheader("Feature Importance") 
    feature_importance = classifier.feature_importances_
    sorted_idx = feature_importance._argsort()
    feature_names = ["Gender", "Married", "ApplicantIncome", "LoanAmount", "Credit_History"]
    bar_data = pd.DataFrame(feature_importance[sorted_idx], feature_names[sorted_idx])
    st.bar_chart(bar_data) 

    st.subheader("Confusion Matrix") 
    st.write("Confusion Matrix Visual will be placed here.")

