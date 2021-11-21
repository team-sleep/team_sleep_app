# import hydralit automatically imports all of Streamlit
# https://github.com/TangleSpace/hydralit
import hydralit as st
#import psycopg2  #for postgres
import pandas as pd
import pickle
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


pickle_in = open('classifier_5.pkl', 'rb') 
classifier = pickle.load(pickle_in)

def prediction(age, gender, bmi, papdev, lsnore, sngasp, breathst, sq1, sq2, sq3, sq4, sq5, sq6, slmed):
	# Pre-processing user input    
	#Age, gender, BMI
	if gender == "Male":
		gender = 0
	else:
		gender = 1


# Feature 0
	if papdev == "CPAP":
		papdev = 0
	elif papdev == "BiPAP/BiLevel":
		papdev = 1
	else:
		papdev = -55

# Feature 1
	#Q1
	if lsnore == "Never":
		lsnore = 0
	elif lsnore == "Rarely":
		lsnore = 1
	elif lsnore == "Sometimes":
		lsnore = 2
	elif lsnore == "Frequently":
		lsnore = 3
	elif lsnore == "Always":
		lsnore = 4
	else:
		lsnore = 5
                
#Q2    
	if sngasp == "Never":
		sngasp = 0
	elif sngasp == "Rarely":
		sngasp = 1
	elif sngasp == "Sometimes":
		sngasp = 2
	elif sngasp == "Frequently":
		sngasp = 3
	elif sngasp == "Always":
		sngasp = 4
	else:
		sngasp = 5
            
        #Q3   
	if breathst == "Never":
		breathst = 0
	elif breathst == "Rarely":
		breathst = 1
	elif breathst == "Sometimes":
		breathst = 2
	elif breathst == "Frequently":
		breathst = 3
	elif breathst == "Always":
		breathst = 4
	else:
		breathst = 5
            
	# Calculate index_1    
	total = lsnore + sngasp + breathst
	index_1 = total/3
        
	#Calculate map_score
	#first x is calculated where [x=-8.16+(1.299*index_1)+ (0.163*BMI)-(0.028*index_1*BMI)+(0.032*Age)+(1.278*Gender)] 
	#then map_score is calculated by [=EXP(x)/(1+EXP(x)) 
        
	xcalc = -8.16+(1.299*index_1)+(0,163*bmi)-(0.028*index_1*bmi)+(0.032*age)+(1.278*gender)
	map_score = exp(x)/(1+exp(x))

	
#Feature 2

	if sq1 == "11am-12pm":
		sq1 = 1
	elif sq1 == "9:45-11am":
		sq1 = 2
	elif sq1 == "7:45-9:45am":
		sq1 = 3
	elif sq1 == "6:30-7:45am":
		sq1 = 4
	else:
		sq1 = 5
            
	if sq2 == "Very tired":
		sq2 = 1
	elif sq2 == "Fairly tired":
		sq2 = 2
	elif sq2 == "Fairly refreshed":
		sq2 = 3
	else:
		sq2 = 4
            
            
	if sq3 == "1:45-3am":
		sq3 = 1
	elif sq3 == "12:30-1:45am":
		sq3 = 2
	elif sq3 == "10:15pm-12:30am":
		sq3 = 3
	elif sq3 == "9-10:15pm":
		sq3 = 4
	else:
		sq3 = 5
            
	if sq4 == "10pm-12am":
		sq4 = 1
	elif sq4 == "5-10pm":
		sq4 = 2
	elif sq4 == "10am-5pm":
		sq4 = 3
	elif sq4 == "8-10am":
		sq4 = 4
	elif sq4 == "5-8am":
		sq4 = 5
	else:
		sq4 = 6
            
	if sq5 == "Definitely evening":
		sq5 = 1
	elif sq5 == "More evening than morning":
		sq5 = 2
	elif sq5 == "More morning than evening":
		sq5 = 3
	else:
		sq5 = 4
            
	if sq6 == "No":
		sq6 = 0
	else:
		sq6 = 1
            	
	rmeqscore = sq1+sq2+sq3+sq4+sq5+sq6

# Feature 3
	if slmed == "No":
		slmed = 0
	else:
		slmed = 1



# Making predictions 
#scaled_data = scaler.transform( 
#     [[papdev, map_score, rmeqscore, slmed, bmi]])
	prediction = classifier.predict([[papdev, map_score, rmeqscore, slmed, bmi]])

	if prediction == "SRBD Apnea":
		pred = 'Sleep Apnea'
	else:
		pred = 'No Sleep Apnea'
	return pred
    	
	
	
def find_importances (Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   
	# Pre-processing user input    
	#Age, gender, BMI
	if gender == "Male":
		gender = 0
	else:
		gender = 1


# Feature 0
	if papdev == "CPAP":
		papdev = 0
	elif papdev == "BiPAP/BiLevel":
		papdev = 1
	else:
		papdev = -55

# Feature 1
	#Q1
	if lsnore == "Never":
		lsnore = 0
	elif lsnore == "Rarely":
		lsnore = 1
	elif lsnore == "Sometimes":
		lsnore = 2
	elif lsnore == "Frequently":
		lsnore = 3
	elif lsnore == "Always":
		lsnore = 4
	else:
		lsnore = 5
                
#Q2    
	if sngasp == "Never":
		sngasp = 0
	elif sngasp == "Rarely":
		sngasp = 1
	elif sngasp == "Sometimes":
		sngasp = 2
	elif sngasp == "Frequently":
		sngasp = 3
	elif sngasp == "Always":
		sngasp = 4
	else:
		sngasp = 5
            
        #Q3   
	if breathst == "Never":
		breathst = 0
	elif breathst == "Rarely":
		breathst = 1
	elif breathst == "Sometimes":
		breathst = 2
	elif breathst == "Frequently":
		breathst = 3
	elif breathst == "Always":
		breathst = 4
	else:
		breathst = 5
            
	# Calculate index_1   
	total = lsnore + sngasp + breathst
	index_1 = = total/3
        
	#Calculate map_score
	#first x is calculated where [x=-8.16+(1.299*index_1)+ (0.163*BMI)-(0.028*index_1*BMI)+(0.032*Age)+(1.278*Gender)] 
	#then map_score is calculated by [=EXP(x)/(1+EXP(x)) 
        
	xcalc = -8.16+(1.299*index_1)+(0,163*bmi)-(0.028*index_1*bmi)+(0.032*age)+(1.278*gender)
	map_score = exp(x)/(1+exp(x))

	
#Feature 2

	if sq1 == "11am-12pm":
		sq1 = 1
	elif sq1 == "9:45-11am":
		sq1 = 2
	elif sq1 == "7:45-9:45am":
		sq1 = 3
	elif sq1 == "6:30-7:45am":
		sq1 = 4
	else:
		sq1 = 5
            
	if sq2 == "Very tired":
		sq2 = 1
	elif sq2 == "Fairly tired":
		sq2 = 2
	elif sq2 == "Fairly refreshed":
		sq2 = 3
	else:
		sq2 = 4
            
            
	if sq3 == "1:45-3am":
		sq3 = 1
	elif sq3 == "12:30-1:45am":
		sq3 = 2
	elif sq3 == "10:15pm-12:30am":
		sq3 = 3
	elif sq3 == "9-10:15pm":
		sq3 = 4
	else:
		sq3 = 5
            
	if sq4 == "10pm-12am":
		sq4 = 1
	elif sq4 == "5-10pm":
		sq4 = 2
	elif sq4 == "10am-5pm":
		sq4 = 3
	elif sq4 == "8-10am":
		sq4 = 4
	elif sq4 == "5-8am":
		sq4 = 5
	else:
		sq4 = 6
            
	if sq5 == "Definitely evening":
		sq5 = 1
	elif sq5 == "More evening than morning":
		sq5 = 2
	elif sq5 == "More morning than evening":
		sq5 = 3
	else:
		sq5 = 4
            
	if sq6 == "No":
		sq6 = 0
	else:
		sq6 = 1
            	
	rmeqscore = sq1+sq2+sq3+sq4+sq5+sq6

# Feature 3
	if slmed == "No":
		slmed = 0
	else:
		slmed = 1

	d = {'PAP Device': [papdev], 'Multivariable Apnea Prediction Score': [map_score], "reduced Morningness Eveningness Questionnaire":[rmeqscore], "Sleep Aid Medication":[slmed], "BMI":[bmi] }
	df = pd.DataFrame(data=d)


	from treeinterpreter import treeinterpreter as ti
	prediction, bias, contributions = ti.predict(classifier, df)
	N = 6 # no of entries in plot , 4 ---> features & 1 ---- class label


	import matplotlib.pyplot as plt
	import numpy as np

	col = ['PAP Device', 'Multivariable Apnea Prediction Score', 'reduced Morningness Eveningness Questionnaire', 'Sleep Aid Medication', 'BMI',"result"]

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
	st.pyplot(fig)
    



app = st.HydraApp(title='Team Sleep App', nav_horizontal=True, 
      use_banner_images=[None,None,{'header':"<h1 style='text-align:center;padding: 0px 0px;color:black;font-size:200%;'>Team 001: Sleep Diagnosis</h1><br>"},None,None], 
      banner_spacing=[5,30,60,30,5],)

@app.addapp(title="User", is_home=True) #set to false to not use "home icon" for this page
def tab1():
	from pages import tab1
	#age, gender, bmi, papdev, lsnore, sngasp, breathst, sq1, sq2, sq3, sq4, sq5, sq6, slmed
	st.sidebar.title("Sleep Questionnaire")
	age = st.sidebar.number_input('Age')
	bmi = st.sidebar.number_input('BMI')
	gender = st.sidebar.selectbox('Gender',("Male", "Female"))
    
    
	papdev = st.sidebar.selectbox('Type of PAP Device',("CPAP","BiPAP/BiLevel","Do not know")) 
	lsnore = st.sidebar.selectbox('How often have you had or been told you had loud snoring?',("Never","Sometimes","Frequently","Always","Don't know"))
	sngasp = st.sidebar.selectbox('How often have you had or been told you had snorting/gasping?',("Never","Sometimes","Frequently","Always","Don't know"))
	breathst = st.sidebar.selectbox('How often have you had or been told you stopped breathing?',("Never","Sometimes","Frequently","Always","Don't know"))

	sq1 = st.sidebar.selectbox('What time of day would you get up if you were entirely free to plan your day?',("11am-12pm","9:45-11am","7:45-9:45am","6:30-7:45am","5-6:30am")) 
	sq2 = st.sidebar.selectbox('During the first half hour after having awakened in the morning, how tired do you feel? ',("Very tired","Fairly tired","Fairly refreshed","Very refreshed")) 
	sq3 = st.sidebar.selectbox('At what time in the evening do you feel tired?',("1:45-3am","12:30-1:45am","10:15pm-12:30am","9-10:15pm","8-9pm")) 
	sq4 = st.sidebar.selectbox('At what time of the day do you think that you feel your best?',("10pm-12am","5-10pm","10am-5pm","8-10am","5-8am","12-5am")) 
	sq5 = st.sidebar.selectbox('At what time in the evening do you feel tired?',("Definitely evening","More evening than morning","More morning than evening","Definitely morning")) 
	sq6 = st.sidebar.selectbox('Do you routinely travel to other time zones?',("Yes","No")) 

	slmed = st.sidebar.selectbox('Do you take a sleeping aid once a week or more?',("Yes","No")) 


	result = ""
	
	if st.sidebar.button("Predict"): 
		result = prediction(age, gender, bmi, papdev, lsnore, sngasp, breathst, sq1, sq2, sq3, sq4, sq5, sq6, slmed) 
		st.sidebar.success('Your result is {}'.format(result))
		#print(LoanAmount)


		# feat_importances = pd.Series(importances.feature_importances_, index=["Gender", "Married", "ApplicantIncome", "LoanAmount", "Credit_History"]).sort_values(ascending=False)
		# impPlot(feat_importances, 'Random Forest Classifier')
		find_importances(age, gender, bmi, papdev, lsnore, sngasp, breathst, sq1, sq2, sq3, sq4, sq5, sq6, slmed)  
		st.write('\n')
        

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
