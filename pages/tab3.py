import hydralit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz
import plotly.express as px 
import math

MAX_TREES=150

def run():
    html_temp = """ 
    <div style ="background-color:black;padding:1px"></div> 
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    st.title("Random Forest Visualization")
    
    if 'prediction' not in st.session_state:
        st.header('To view this page, first go to the home tab and complete the questionnaire.')
        return

    st.subheader('The random forest algorithm determines an outcome based on the predictions from ' + str(MAX_TREES) + ' decision trees.')

    html_predicton = """ 
    <p style="font-size:24px">Your overall prediction = <bold>""" + st.session_state.prediction + "</bold></p>" 
    st.markdown(html_predicton, unsafe_allow_html = True)

    # parameters used to generate the prediction on the home tab
    # we need this to exec predict() on each decision tree
    scaled_data = st.session_state.scaled_data

    # the home tab also stashes the prediction in the session.
    if st.session_state.prediction == 'Sleep Apnea':
        apnea_prediction=True
    else:
        apnea_prediction=False

    pickle_in = open('classifier_rf.pkl', 'rb') 
    classifier = pickle.load(pickle_in)

    # used to summarize the prediction from each of the 150 trees for pie chart
    ap_count = 0
    no_ap_count = 0

    # list of the trees predicting sleep apnea or not
    # we use this to help filter the display of the 150 trees
    apnea_trees =[]
    no_apnea_trees =[]

    # loop through each tree
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


    ######### build pie chart

    categ = np.array(["Sleep Apnea", "No Sleep Apnea"])
    counts = np.array([ap_count, no_ap_count])

    rt_preduction_summary = pd.DataFrame({"Prediction": categ, 
                            "Counts": counts})

    fig = px.pie(rt_preduction_summary, values=rt_preduction_summary.Counts, 
                    names=rt_preduction_summary.Prediction, color=rt_preduction_summary.Prediction,
                    color_discrete_map={'Sleep Apnea':'#003057', 'No Sleep Apnea':'#A28D5B'}) #gatech colors :-)
    fig.update_layout(title="<b>Breakdown of how each of the " + str(MAX_TREES) + " decision trees predicted your outcome</b>",
                      margin=dict(l=30, r=30, t=50, b=0))

    st.plotly_chart(fig, use_container_width=True)


    ######### filter and display the tree images
    
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
    
    # set a few things to unify determining half vs all counts in the block of 
    # code further below regardless of prediction.
    if(apnea_prediction):
        supporting_num_trees = len(apnea_trees)
        not_supporting_num_trees = MAX_TREES - len(apnea_trees)
    else:
        supporting_num_trees = MAX_TREES - len(apnea_trees)
        not_supporting_num_trees = len(apnea_trees)

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

    # round up so there are enough rows for the number of images to display
    num_rows = int(math.ceil(display_num_trees / num_cols))
    display_num_trees = int(display_num_trees)

    html_tree_count = """ 
    <p style="font-size:24px">""" + str(display_num_trees) + " tree(s) displayed out of a total of " + str(len(display_trees)) + " " + display_opt.lower() + ".</p>"
    st.markdown(html_tree_count, unsafe_allow_html = True)

    idx=-1
    for r in range(0,num_rows): # for each row in table
        cols = st.columns(num_cols)
        for c in range(0, num_cols): # for each column in table
            idx = idx + 1
            # guard to not overshoot when (rows times columns) < number of trees to display
            if(idx < display_num_trees): 
                # grab the image number from the array.  
                # idx is incremented each loop to walk this array sequentially.
                # display trees is set above depending on the value of the radio's filter.
                tree_id = display_trees[idx]
                img = 'images/rtree_image_%i.png' %tree_id
                cols[c].image(img, use_column_width=True)
    
    # deadt code
    
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

    