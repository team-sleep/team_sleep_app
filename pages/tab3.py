import hydralit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz
import altair as alt
import plotly.express as px 
import math

def run():
    html_temp = """ 
    <div style ="background-color:black;padding:1px"></div> 
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.title("Random Forest Visualization")
    
    if 'prediction' not in st.session_state:
        st.header('To view this page, first go to the home tab and complete the questionnaire.')
        return

    st.markdown('The random forest algorithm determines an outcome based on the predictions from **150** decision trees. ' +
            'Use the options below to view the decision trees supporting or not supporting to your overall prediction of **' + st.session_state.prediction + '**.')

    scaled_data = st.session_state.scaled_data

    if st.session_state.prediction == 'Sleep Apnea':
        apnea_prediction=True
    else:
        apnea_prediction=False

    pickle_in = open('classifier_rf.pkl', 'rb') 
    classifier = pickle.load(pickle_in)

    # used to summarize the prediction from each of the 150 trees
    ap_count = 0
    no_ap_count = 0

    # list of the trees predicting sleep apnea or not
    apnea_trees =[]
    no_apnea_trees =[]

    for i in range(len(classifier.estimators_)):
        
        estimator = classifier.estimators_[i]
        #get the prediction from the i-th tree
        result = estimator.predict(scaled_data)

        if(result == 1.0):
            #apnea
            ap_count += 1
            apnea_trees.append(i)
        else:
            #not apnea
            no_ap_count += 1
            no_apnea_trees.append(i)

    # build pie chart

    categ = np.array(["Sleep Apnea", "No Sleep Apnea"])
    counts = np.array([ap_count, no_ap_count])

    rt_preduction_summary = pd.DataFrame({"Prediction": categ, 
                            "Counts": counts})

    fig = px.pie(rt_preduction_summary, values=rt_preduction_summary.Counts, 
                    names=rt_preduction_summary.Prediction, color=rt_preduction_summary.Prediction,
                    color_discrete_map={'Sleep Apnea':'#003057', 'No Sleep Apnea':'#A28D5B'}) #gatech colors :-)
    fig.update_layout(
    title="<b>Breakdown of how each of the 150 decision trees predicted your outcome</b>")
    st.plotly_chart(fig)
    
    if(apnea_prediction):
        supporting_num_trees = len(apnea_trees)
        not_supporting_num_trees = 150 - len(apnea_trees)
    else:
        supporting_num_trees = 150 - len(apnea_trees)
        not_supporting_num_trees = len(apnea_trees)

    opt_col1, opt_col2 = st.columns(2)

    opposite_text='Not supporting your prediction'
    display_opt = opt_col1.radio(
        "Show decision trees in the random forest",
        ('Supporting your overall prediction of ' + st.session_state.prediction, opposite_text), index=0)

    if display_opt == opposite_text:
        display_supporting=False
        if(apnea_prediction):
            display_trees = no_apnea_trees
        else:
            display_trees = apnea_trees
    else:
        display_supporting=True
        if(apnea_prediction):
            display_trees = apnea_trees
        else:
            display_trees = no_apnea_trees

    
    display_num_opt = opt_col2.select_slider("Number of Trees to display", options=["one", "few", "half", "all"], on_change=None, args=None, kwargs=None)
    
    if(display_num_opt == 'one'):
        num_cols=1
        display_num_trees=1
    elif(display_num_opt == 'few'):
        num_cols=2
        display_num_trees=4
    elif(display_num_opt == 'half'):
        num_cols=5
        if(display_supporting):
            display_num_trees=supporting_num_trees / 2
        else:
            display_num_trees=not_supporting_num_trees / 2
    else:
        num_cols=5
        if(display_supporting):
            display_num_trees=supporting_num_trees
        else:
            display_num_trees=not_supporting_num_trees

    num_rows = int(math.ceil(display_num_trees / (num_cols)))
    display_num_trees = int(display_num_trees)

    st.subheader(str(display_num_trees) + " tree(s) displayed out of a total of " + str(len(display_trees)) + " " + display_opt.lower())

    idx=-1
    for r in range(0,num_rows): # number of rows in table
        cols = st.columns(num_cols)
        for c in range(0, num_cols):
            idx = idx + 1
            if(idx < display_num_trees): 
                tree_id = display_trees[idx]
                img = 'images/rtree_image_%i.png' %tree_id
                cols[c].image(img, use_column_width=True)
    
    # print("display_num_trees", display_num_trees, "num_rows", num_rows, "num images displayed", idx + 1)
    # print(display_trees)


    # display_num_trees = st.select_slider("Number of Trees to display", options=[1,4,50,150], on_change=None, args=None, kwargs=None)
    
    # if(display_num_trees == 1):
    #     num_cols=1
    # elif(display_num_trees == 4):
    #     num_cols=2
    # else:
    #     num_cols=5

    # num_rows = int(display_num_trees / num_cols)

    # for r in range(0,num_rows): # number of rows in table
    #     cols = st.columns(num_cols)
    #     for c in range(0, num_cols):
    #         count = r + c + 1
    #         cols[c].image('images/rtree_image_%i.png' %count, use_column_width=True)

    