import pickle
from sklearn.tree import export_graphviz

def gen_with_seq_number():
	# loading the trained model
	pickle_in = open('classifier_rf.pkl', 'rb') 
	classifier = pickle.load(pickle_in)

	print("total random trees =",len(classifier.estimators_))
	print("appending number on each rt image from 0 to", str(len(classifier.estimators_) - 1))

	for i in range(len(classifier.estimators_)):
		outfile = "images/rtree_image_" + str(i)
		estimator = classifier.estimators_[i]
		
		print("writing image", outfile)
		
		export_graphviz(estimator, out_file=outfile + ".dot", 
						feature_names = ["What Treatment Since Enrollment Study Cpap", "MAP Score","rMEQ Total Score","What Treatment Since Enrollment Study Med","Body Mass Index (BMI)"],
						class_names = ['Sleep_Apnea', "No_Sleep_Apnea"],
						rounded = True, proportion = False, 
						precision = 2, filled = True)

		# Convert to png using system command (requires Graphviz)
		from subprocess import call
		call(['dot', '-Tpng', outfile + '.dot', '-o', outfile + '.png', '-Gdpi=200'])
	
	print("writing images is done.")
	pass

if __name__=='__main__':
	gen_with_seq_number()
	#gen_cols(max_columns_on_page=5)