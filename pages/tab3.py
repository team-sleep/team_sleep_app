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

    for i in range(0,5): # number of rows in  table
        cols = st.columns(5) # number of columns in each row

        cols[0].image('images/rtree_row%i_col0.png' %i, use_column_width=True)
        cols[1].image('images/rtree_row%i_col1.png' %i, use_column_width=True)
        cols[2].image('images/rtree_row%i_col2.png' %i, use_column_width=True)
        cols[3].image('images/rtree_row%i_col3.png' %i, use_column_width=True)
        cols[4].image('images/rtree_row%i_col4.png' %i, use_column_width=True)

    
    