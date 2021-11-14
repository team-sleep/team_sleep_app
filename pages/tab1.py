import hydralit as st

def run():
    html_temp = """ 
    <div style ="background-color:black;padding:1px"> 
    <h3 style ="color:white;text-align:center;">Questions and viz of feature importance</h3> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html = True)
