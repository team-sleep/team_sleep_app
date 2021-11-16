import pickle
from sklearn.tree import export_graphviz


def gen_cols(max_columns_on_page=5):
# loading the trained model
	pickle_in = open('loan_classifier.pkl', 'rb') 
	classifier = pickle.load(pickle_in)

	print("total random trees =",len(classifier.estimators_))
	print("writing images supporting a layout with", max_columns_on_page, "columns")

	row=-1

	for i in range(len(classifier.estimators_)):
		col = i % max_columns_on_page
		
		if(col==0):
			row=row + 1

		outfile="images/rtree_row" + str(row) + "_col" + str(col)
		estimator = classifier.estimators_[i]

		export_graphviz(estimator, out_file=outfile + ".dot", 
						feature_names = ['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History'],
						class_names = ['Not_Approved', "Approved"],
						rounded = True, proportion = False, 
						precision = 2, filled = True)

		# Convert to png using system command (requires Graphviz)
		from subprocess import call
		call(['dot', '-Tpng', outfile + '.dot', '-o', outfile + '.png', '-Gdpi=200'])
	
	print("generated images for a grid with ", row+1, "rows and ", col + 1, "columns")
	pass

def gen_with_seq_number():
# loading the trained model
	pickle_in = open('loan_classifier.pkl', 'rb') 
	classifier = pickle.load(pickle_in)

	print("total random trees =",len(classifier.estimators_))
	print("appending number on each rt image from 0 to", str(len(classifier.estimators_) - 1))

	for i in range(len(classifier.estimators_)):
		outfile="images/rtree_image_" + str(i)
		estimator = classifier.estimators_[i]
		print("writing image", outfile)

		export_graphviz(estimator, out_file=outfile + ".dot", 
						feature_names = ['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History'],
						class_names = ['Not_Approved', "Approved"],
						rounded = True, proportion = False, 
						precision = 2, filled = True)

		# Convert to png using system command (requires Graphviz)
		from subprocess import call
		call(['dot', '-Tpng', outfile + '.dot', '-o', outfile + '.png', '-Gdpi=200'])
	
	print("writing images is done.")
	pass

if __name__=='__main__':
	gen_with_seq_number()
	gen_cols(max_columns_on_page=5)