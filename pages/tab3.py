import hydralit as st
import pickle
from sklearn.tree import export_graphviz

def run():
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

    