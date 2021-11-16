import hydralit as st
import pickle
import altair as alt
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
    feature_names = ["Gender", "Married", "ApplicantIncome", "LoanAmount", "Credit_History"]
    # graph = plt.barh(feature_names, feature_importance)
    bar_data = pd.DataFrame(feature_importance, feature_names)
    st.title(f"bar data: {bar_data}")
    st.bar_chart(bar_data) 
