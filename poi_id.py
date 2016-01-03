#!/usr/bin/python

import sys, os
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from numpy import nan
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import SnowballStemmer
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import RandomizedPCA, TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from itertools import product


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
 'salary',
 'to_messages',
 # 'deferral_payments', # Most NaN's of all features w/ variances
 'total_payments',
 'exercised_stock_options',
 'bonus',
 'restricted_stock',
 'shared_receipt_with_poi',
 # 'restricted_stock_deferred', # No Variance for POI
 'total_stock_value',
 'expenses',
 # 'loan_advances', # Only 3 data points
 'from_messages',
 'other',
 'from_this_person_to_poi',
 # 'director_fees', # No Variance for POI
 'deferred_income',
 'long_term_incentive',
 'from_poi_to_this_person']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers

#s The TOTAL entry is essentially an erroneous entry
data_dict.pop('TOTAL')

# EUGENE has no features at all, other than that he is not a POI
## NOTE: Looking at feature_format, it will also do this by default
data_dict.pop('LOCKHART EUGENE E')

# TRAVEL AGENCY IN THE PARK is indicative of shady dealing but not a person
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

def omit_NaN(d_dict):
	"""Utility function for bringing into a Pandas DataFrame. Removing value
	instead of having "NaN" allows pandas to assign the np.nan missing value"""
	for name in d_dict.keys():
		d = dict([(field, d_dict[name][field]) for field in d_dict[name].keys() 
			if d_dict[name][field] != "NaN"])
		d['name'] = name
		yield(d)

# Bring into Pandas DataFrame for easier exploration
df = pd.DataFrame(omit_NaN(data_dict))
numeric_cols = [col for col in df.columns if col not in ['name','email_address', 'poi']]
for col in numeric_cols:
	df[col] = df[col].astype('float32')

# # Boxplots vs POI plot of input vars
# n_cols = 4
# n_rows = len(numeric_cols) // n_cols
# if len(numeric_cols) % n_cols > 0:
# 	n_rows += 1
# for i, col in enumerate(numeric_cols):	
# 	sb.plt.subplot(n_rows, n_cols, i + 1)
# 	sb.boxplot(df[col].dropna(), df.poi)
# sb.plt.show()

### Task 3: Create new feature(s)


### Extract features and labels from dataset for local testing

class OrderedEmailTextExtractor(BaseEstimator, TransformerMixin):
	""" The reason for the mixin and everything is that I wanted this to be a
	part of the sklearn pipeline. I wasn't really able to get it to work, 
	though, so it's just a part of preparing the features now.

	This relies on the sort_keys = True of featureFormat to work correctly. It
	looks for a file named to_<email>.txt in the emails_by_address directory.
	If it finds that it goes through each of the file paths (emails) listed.
	It does stemming and also removes any word with a digit in it. """

	names = sorted(data_dict.keys())
	
	def fit(self, x, y=None):
		return self

	def transform(self, x):
		"""Return a list of full stemmed email texts in same order as 
		data"""
		texts = []
		email_list_dir = "emails_by_address"
		email_files = os.listdir(email_list_dir)
		# This is the same order feature_format uses
		# names = sorted(data_dict.keys())
		for name in self.names:
			fname = "to_%s.txt" % data_dict[name]['email_address']
			#If there is no email address, a empty string
			if data_dict[name]['email_address'] == "NaN":
				texts.append("")
			# Same if can't find a from_<email> file
			elif fname not in email_files:
				texts.append("")
			else:
				name_email_file = os.path.join(email_list_dir,fname)
				texts.append(self.parse_emails(name_email_file))
		return texts


	def parse_emails(self, fname):
		"""Called by transform, takes a list of files and grabs the stemmed
		email text from each of them."""
		text = []
		stemmer = SnowballStemmer("english")
		trans_table_digits = string.maketrans(string.digits, "0"*10)
		with open(fname, 'r') as f:
			email_file_list = [item.strip() for item in f.readlines()][::10]
		for email_file in email_file_list:
			efname = os.path.join("..", email_file[20:].replace("/", os.path.sep))
			with open(efname, 'r') as f:
				f.seek(0)
				all_text = f.read()
			content = all_text.split("X-FileName:")
			if len(content) < 2: continue
			content = content[1].split("-----Original Message-----")[0]
			content = content.translate(trans_table_digits, string.punctuation)
			# Ignore if ANY number is a digit
			words = [stemmer.stem(word) for word in content.split() if "0" not in word]
			text.extend(words)
		return " ".join(text)

# Integrate some email data
use_text_data = False
if use_text_data:
	read_from_cache = True
	if not read_from_cache:
		o = OrderedEmailTextExtractor()
		t = transform(labels) # The input argument is not used
		pickle.dump(t, open('email_text.pkl', 'w'))
	else:
		t = pickle.load(open('email_text.pkl', 'r'))

	# Do some dimensionality reduction due ot size of dataset
	vect = TfidfVectorizer(sublinear_tf=True, max_df = 0.5, stop_words="english")
	freqs = vect.fit_transform(t)

	# Run a little parameter study on number of components vs explained var ratio
	# comps = [2, 5, 10, 25, 50, 75, 100, 150]
	# evar_rat =[]
	# for num in comps:
	# 	pca = TruncatedSVD(n_components=num)
	# 	pca.fit(freqs)
	# 	evar_rat.append(sum(pca.explained_variance_ratio_))

	# for n, rat in zip(comps, evar_rat):
	# 	print "# Components:%s, Explained Variance Ratio: %s" % (n, rat)

	# Going with 50, as that gets us 80% there
	final_pca_comps = 50 
	pca = TruncatedSVD(n_components=final_pca_comps)
	text_features = pca.fit_transform(freqs)


	# Merge email text features in with rest
	for name, row in zip(sorted(data_dict.keys()), text_features):
		for i, val in enumerate(row):
			data_dict[name]["pca_%i" % i] = val

	features_list += ["pca_%i" % num for num in range(final_pca_comps)]

### Store to my_dataset for easy export below.
my_dataset = data_dict

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Do a manual grid search to tune the algroithm params for GaussianNB
# for k in [5, 10, 15]: #[5, 10, 20, 30, 40]:
# 	clf = make_pipeline(MinMaxScaler(), SelectKBest(k = k), GaussianNB())
# 	test_classifier(clf, my_dataset, features_list)
#
clf = make_pipeline(StandardScaler(), SelectKBest(k=5), GaussianNB())


# Did a manual grid search for C=[0.1, 1, 10, 100], k = [5, 10, 20, 30, 40]
# for C, k in product([0.1, 1, 10, 100], [5, 10, 15]):#,20, 30, 40]):
# 	clf = make_pipeline(MinMaxScaler(), SelectKBest(k = k), LogisticRegression(C=C))
# 	test_classifier(clf, my_dataset, features_list)

# Best LR params
# clf = make_pipeline(MinMaxScaler(), SelectKBest(k = 10), LogisticRegression(C=100))

# estim = AdaBoostClassifier()
# for lr, ne in product([0.1, 1, 5, 10], [10, 25, 50, 100]):
# 	clf = AdaBoostClassifier(learning_rate=lr, n_estimators=ne)
# 	test_classifier(clf, my_dataset, features_list)

# param_grid = dict(learning_rate= [0.1, 1, 5, 10],
# 				n_estimators= [10, 25, 50, 100])
# cv = GridSearchCV(estim, param_grid)
# cv.fit(features, labels)
# cv.best_estimator_
# # Rebuild Estimator with optimal params for tester
# clf = AdaBoostClassifier(learning_rate=5, n_estimators=10)

# Aborted attempt at a FeatureUnion with text features
# clf = Pipeline([
# 	('union', FeatureUnion([
# 		('financial_features', Pipeline([
# 			("std_scaler", StandardScaler()),
# 			])),
# 		('text_features', Pipeline([
# 			("email_extract", OrderedEmailTextExtractor()),
# 			("tfid", TfidfVectorizer(sublinear_tf=True, stop_words="english", max_df=0.5)),
# 			("text_pca", TruncatedSVD())#RandomizedPCA())
# 			])
# 		)])
# 	),
# 	('select_k_best', SelectKBest()),
# 	('classifier', GaussianNB())
# 	])



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)