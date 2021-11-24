import pickle
from sklearn.tree import export_graphviz

def gen_with_seq_number():
	# loading the trained model
	pickle_in = open('classifier_rf.pkl', 'rb') 
	classifier = pickle.load(pickle_in)

	print("total random trees =",len(classifier.estimators_))
	print("appending number on each rt image from 0 to", str(len(classifier.estimators_) - 1))

	scaler_in = open('scaler_5.pkl','rb')
	scaler5 = pickle.load(scaler_in)
	scaled_data = scaler5.transform([[1.0, 1.0, 1.0, 1.0, 1.0]])

	for i in range(len(classifier.estimators_)):
		outfile = "images/rtree_image_" + str(i)
		estimator = classifier.estimators_[i]
		
		print("writing image", outfile)

        
		result = estimator.predict(scaled_data)
		print("est=",i, result)
		
		export_graphviz(estimator, out_file=outfile + ".dot", 
						feature_names = ["What Treatment Since Enrollment Study Cpap", "MAP Score","rMEQ Total Score","What Treatment Since Enrollment Study Med","Body Mass Index (BMI)"],
						class_names = ['Sleep_Apnea', "No_Sleep_Apnea"],
						rounded = True, proportion = False, 
						precision = 2, filled = True)

		# Convert to png using system command (requires Graphviz)
		from subprocess import call
		call(['dot', '-Tpng', outfile + '.dot', '-o', outfile + '.png', '-Gdpi=100']) #200 is a good number
	
	print("writing images is done.")
	pass

if __name__=='__main__':
	gen_with_seq_number()
	#gen_cols(max_columns_on_page=5)