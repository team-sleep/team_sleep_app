import hydralit as st
import pickle
import altair as alt
import pandas as pd
import plotly.figure_factory as ff


def run():
    html_temp = """ 
    <div style ="background-color:black;padding:1px"></div> 
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.title("Model Visualization")

    pickle_in = open('classifier_rf.pkl', 'rb') 
    classifier = pickle.load(pickle_in)

    st.subheader("Feature Importance") 
    feature_importance = classifier.feature_importances_
    feature_names = ["What Treatment Since Enrollment Study Cpap", "MAP Score","rMEQ Total Score","What Treatment Since Enrollment Study Med","Body Mass Index (BMI)"]
    # bar_data = pd.DataFrame(feature_importance, feature_names)
    # st.bar_chart(bar_data) 

    bar_data = pd.DataFrame({"Feature Importance": feature_importance, 
                         "Feature": feature_names})
    c = alt.Chart(bar_data).mark_bar().encode(
    x = "Feature Importance",
    y = alt.Y('Feature', sort='-x'))

    st.altair_chart(c, use_container_width=True)
    
    st.subheader("Confusion Matrix") 
    
    z = [[73, 44],
     [20, 116]]

    x = ['Non-Apnea', 'Sleep Related Breating Disorder Apnea']
    y = ['Non-Apnea', 'Sleep Related Breating Disorder Apnea']

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')


    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",      
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    # fig.show()
    st.plotly_chart(fig, use_container_width=True)

