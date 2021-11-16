# import hydralit automatically imports all of Streamlit
# https://github.com/TangleSpace/hydralit
import hydralit as st
#import psycopg2  #for postgres
import pandas as pd
import pickle
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)

def prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   

# Pre-processing user input    
        if Gender == "Male":
            Gender = 0
        else:
            Gender = 1

        if Married == "Unmarried":
            Married = 0
        else:
            Married = 1

        if Credit_History == "Unclear Debts":
            Credit_History = 0
        else:
            Credit_History = 1  

        LoanAmount = LoanAmount / 1000

        # Making predictions 
        prediction = classifier.predict( 
            [[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])

        if prediction == 0:
            pred = 'Rejected'
        else:
            pred = 'Approved'
        return pred

def find_importances (Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   
	if Gender == "Male":
		Gender = 0
	else:
        	Gender = 1
		
	if Married == "Unmarried":
       		Married = 0
	else:
        	Married = 1
	
	if Credit_History == "Unclear Debts":
		Credit_History = 0
	else:
		Credit_History = 1  

	LoanAmount = LoanAmount / 1000

	d = {'Gender': [Gender], 'Married': [Married], "ApplicantIncome":[ApplicantIncome], "LoanAmount":[LoanAmount], "Credit_History":[Credit_History] }
	df = pd.DataFrame(data=d)


	from treeinterpreter import treeinterpreter as ti
	prediction, bias, contributions = ti.predict(classifier, df)
	N = 6 # no of entries in plot , 4 ---> features & 1 ---- class label


	import matplotlib.pyplot as plt
	import numpy as np

	col = ['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History',"result"]

	one = []
	zero = []

	for j in range(2):
		list_ =  [one ,zero]
		for i in range(5):
			val = contributions[0,i,j]
			list_[j].append(val)

	zero.append(prediction[0,0]/6)
	one.append(prediction[0,1]/6)
	fig, ax = plt.subplots()
	ind = np.arange(N)   
	width = 0.15        
	p1 = ax.bar(ind, one, width, color='red', bottom=0)
	p2 = ax.bar(ind+width, zero, width, color='green', bottom=0)
	ax.set_title('Contribution of all feature for a particular sample')
	ax.set_xticks(ind + width / 2)
	ax.set_xticklabels(col, rotation = 90)
	ax.legend((p1[0], p2[0]), ('one', 'zero') , bbox_to_anchor=(1.04,1), loc="upper left")
	ax.autoscale_view()
	st.pyplot(fig)
   
def pca_maker(data_import):
	numerical_columns_list = []
	categorical_columns_list = []

	for i in data_import.columns:
		if data_import[i].dtype == np.dtype("float64") or data_import[i].dtype == np.dtype("int64"):
			numerical_columns_list.append(data_import[i])
		else:
			categorical_columns_list.append(data_import[i])

	numerical_data = pd.concat(numerical_columns_list, axis=1)
	categorical_data = pd.concat(categorical_columns_list, axis=1)

	numerical_data = numerical_data.apply(lambda x: x.fillna(np.mean(x)))

	scaler = StandardScaler()

	scaled_values = scaler.fit_transform(numerical_data)

	pca1 = PCA()

	pca_data = pca1.fit_transform(scaled_values)

	pca_data = pd.DataFrame(pca_data)

	new_column_names = ["PCA_" + str(i) for i in range(1, len(pca_data.columns) + 1)]

	column_mapper = dict(zip(list(pca_data.columns), new_column_names))

	pca_data = pca_data.rename(columns=column_mapper)

	output = pd.concat([data_import, pca_data], axis=1)

	return pca1 , scaled_values , output, list(categorical_data.columns), new_column_names , list(numerical_data.columns)

    
def perform_pca(data_import):
	pca , scaled_values , pca_data, cat_cols, pca_cols , num_data = pca_maker(data_import)
	categorical_variable = "Loan_Status"
	pca_1 = "Credit_History"
	pca_2 = "ApplicantIncome"

	st.plotly_chart(px.scatter(data_frame=pca_data, x=pca_1, y=pca_2, color=categorical_variable, template="simple_white", height=800), use_container_width=True)
 
        


app = st.HydraApp(title='Team Sleep App', nav_horizontal=True, 
      use_banner_images=[None,None,{'header':"<h1 style='text-align:center;padding: 0px 0px;color:black;font-size:200%;'>App title here</h1><br>"},None,None], 
      banner_spacing=[5,30,60,30,5],)

@app.addapp(title="User", is_home=True) #set to false to not use "home icon" for this page
def tab1():
	from pages import tab1
	st.sidebar.title("Sleep Questionnaire")
	Gender = st.sidebar.selectbox('Gender',("Male","Female"))
	Married = st.sidebar.selectbox('Marital Status',("Unmarried","Married")) 
	ApplicantIncome = st.sidebar.number_input("Applicants monthly income") 
	LoanAmount = st.sidebar.number_input("Total loan amount")
	Credit_History = st.sidebar.selectbox('Credit_History',("Unclear Debts","No Unclear Debts"))
	result =""
    
	if st.sidebar.button("Predict"): 
		result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) 
		st.sidebar.success('Your loan is {}'.format(result))
		print(LoanAmount)
        
        
# feat_importances = pd.Series(importances.feature_importances_, index=["Gender", "Married", "ApplicantIncome", "LoanAmount", "Credit_History"]).sort_values(ascending=False)
# impPlot(feat_importances, 'Random Forest Classifier')
		find_importances(Gender, Married, ApplicantIncome, LoanAmount, Credit_History)  
		st.write('\n')
		
		#PCA
		data_import = pd.read_csv("train_ctrUa4k.csv")
		perform_pca(data_import)

   

 	 #  @st.cache()
	tab1.run()

@app.addapp(title="Model")
def tab2():
	from pages import tab2
	tab2.run()

@app.addapp(title="Temporary")
def tab_temporary_put_on_tab2_later():
	from pages import tab3
	tab3.run()

#Hydralit: navbar, state management and app isolation, all with this tiny amount of work.
app.run()
