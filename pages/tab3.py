import hydralit as st
import pickle
from sklearn.tree import export_graphviz

def run():
    # html_temp = """ 
    # <div style ="background-color:black;padding:1px"> 
    # <h3 style ="color:white;text-align:center;">Add to tab 2 in a subsequent iteration...random forest visualization</h3> 
    # </div> 
    # """
    # st.markdown(html_temp, unsafe_allow_html = True)

    # load the trained model
    # pickle_in = open('classifier.pkl', 'rb') 
    # classifier = pickle.load(pickle_in)

    # estimator = classifier.estimators_[1]

    # tree = export_graphviz(estimator, 
    #             feature_names = ['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History'],
    #             class_names = ['Not_Approved', "Approved"],
    #             rounded = True, proportion = False, 
    #             precision = 2, filled = True)

    # st.graphviz_chart(tree)

    # export_graphviz(estimator, out_file='tree.dot', 
    #                 feature_names = ['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History'],
    #                 class_names = ['Not_Approved', "Approved"],
    #                 rounded = True, proportion = False, 
    #                 precision = 2, filled = True)

    # # Convert to png using system command (requires Graphviz)
    # from subprocess import call
    # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    #st.sidebar.title("Preferences")
    #display_num_trees = st.sidebar.select_slider("Trees to display", options=[1,4,50,100], on_change=None, args=None, kwargs=None)
    st.markdown('The random forest algorithm determines an outcome based on the predictions from **100** decision trees. ' +
            'Use the slider to visulize a subset or all of the trees in the forest.')

    display_num_trees = st.select_slider("Number of Trees to display", options=[1,4,50,100], on_change=None, args=None, kwargs=None)
    
    if(display_num_trees == 1):
        num_cols=1
    elif(display_num_trees == 4):
        num_cols=2
    else:
        num_cols=5

    num_rows = int(display_num_trees / num_cols)

    for r in range(0,num_rows): # number of rows in table
        cols = st.columns(num_cols)
        for c in range(0, num_cols):
            count = r + c + 1
            cols[c].image('images/rtree_image_%i.png' %count, use_column_width=True)

    