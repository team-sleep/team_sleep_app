import hydralit as st

def run():
    html_temp = """ 
    <div style ="background-color:black;padding:1px"></div> 
    """
    st.markdown(html_temp, unsafe_allow_html = True)
