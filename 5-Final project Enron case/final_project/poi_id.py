#!/usr/bin/python

import sys
import pickle
import pprint #for visualizing dictionnary
import pandas as pd
import numpy as np
sys.path.append("../tools/")
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
                 'poi',
                 'salary',
                 'salary_bonus_ratio',#--> engineered feature
                 'deferral_payments',
                 'total_payments',
                 'loan_advances',
                 'bonus',
                 #'restricted_stock_deferred',
                 'deferred_income',
                 #'total_stock_value',
                 #'expenses',
                 #'director_fees',
                 'exercised_stock_options',
                 #'other',
                 #'long_term_incentive',
                 'restricted_stock',
                 #'to_messages',
                 #'from_poi_to_this_person',
                 #'from_messages',
                 #'from_this_person_to_poi',
                 ]
# Explore dataset

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

#Let's explore the dataset
total_count = len(data_dict)
print "Number of record of the dataset:", total_count
## 146 records counted
print data_dict.keys()[0]

df = pd.DataFrame.from_dict(data_dict,orient='index')
#print df
#pprint.pprint(data_dict[data_dict.keys()[0]])
## The dataset has the following structure:
##{'METTS MARK',
##{'bonus': 600000,
## 'deferral_payments': 'NaN',
## 'deferred_income': 'NaN',
## 'director_fees': 'NaN',
## 'email_address': 'mark.metts@enron.com',
## 'exercised_stock_options': 'NaN',
## 'expenses': 94299,
## 'from_messages': 29,
## 'from_poi_to_this_person': 38,
## 'from_this_person_to_poi': 1,
## 'loan_advances': 'NaN',
## 'long_term_incentive': 'NaN',
## 'other': 1740,
## 'poi': False,
## 'restricted_stock': 585062,
## 'restricted_stock_deferred': 'NaN',
## 'salary': 365788,
## 'shared_receipt_with_poi': 702,
## 'to_messages': 807,
## 'total_payments': 1061827,
## 'total_stock_value': 585062}}


## a lot of values are "NaN", lets assess the quality of the data we have:
poi_count = 0
no_poi_count = 0
missing_value_map = {
    'bonus': {'count':0, 'percentage_total':0, 'poi':0},
    'deferral_payments': {'count':0, 'percentage_total':0, 'poi':0},
    'deferred_income': {'count':0, 'percentage_total':0, 'poi':0},
    'director_fees': {'count':0, 'percentage_total':0, 'poi':0},
    'email_address': {'count':0, 'percentage_total':0, 'poi':0},
    'exercised_stock_options': {'count':0, 'percentage_total':0, 'poi':0},
    'expenses': {'count':0, 'percentage_total':0, 'poi':0},
    'from_messages': {'count':0, 'percentage_total':0, 'poi':0},
    'from_poi_to_this_person': {'count':0, 'percentage_total':0,'poi':0},
    'from_this_person_to_poi': {'count':0, 'percentage_total':0, 'poi':0},
    'loan_advances': {'count':0, 'percentage_total':0, 'poi':0},
    'long_term_incentive': {'count':0, 'percentage_total':0, 'poi':0},
    'other': {'count':0, 'percentage_total':0, 'poi':0},
    'poi': {'count':0, 'percentage_total':0, 'poi':0},
    'restricted_stock': {'count':0, 'percentage_total':0, 'poi':0},
    'restricted_stock_deferred': {'count':0, 'percentage_total':0, 'poi':0},
    'salary': {'count':0, 'percentage_total':0, 'poi':0},
    'shared_receipt_with_poi': {'count':0, 'percentage_total':0, 'poi':0},
    'to_messages': {'count':0, 'percentage_total':0, 'poi':0},
    'total_payments': {'count':0, 'percentage_total':0, 'poi':0},
    'total_stock_value': {'count':0, 'percentage_total':0, 'poi':0}}

for person, features in data_dict.iteritems():
    isPoi = False
    if features['poi'] == True:
        poi_count += 1
        isPoi = True
    else:
        no_poi_count += 1
    for name, value in features.iteritems():
        if value == 'NaN':
            missing_value_map[name]['count'] += 1
            missing_value_map[name]['percentage_total'] = float(missing_value_map[name]['count'])/float(total_count)
            if isPoi:
                missing_value_map[name]['poi'] += 1
print "Number of POI:", poi_count
print "Number of non POI:", no_poi_count
print "Missing values for each feature"
df = pd.DataFrame.from_dict({(i): missing_value_map[i]
                           for i in missing_value_map.keys()
                           },
                       orient='index')
print df
##Number of POI: 18
##Number of non POI: 128
##Missing values for each feature
##                           count  poi  percentage_total
##bonus                         64    2          0.438356
##deferral_payments            107   13          0.732877
##deferred_income               97    7          0.664384
##director_fees                129   18          0.883562
##email_address                 35    0          0.239726
##exercised_stock_options       44    6          0.301370
##expenses                      51    0          0.349315
##from_messages                 60    4          0.410959
##from_poi_to_this_person       60    4          0.410959
##from_this_person_to_poi       60    4          0.410959
##loan_advances                142   17          0.972603
##long_term_incentive           80    6          0.547945
##other                         53    0          0.363014
##poi                            0    0          0.000000
##restricted_stock              36    1          0.246575
##restricted_stock_deferred    128   18          0.876712
##salary                        51    1          0.349315
##shared_receipt_with_poi       60    4          0.410959
##to_messages                   60    4          0.410959
##total_payments                21    0          0.143836
##total_stock_value             20    0          0.136986

# Looking at the dataset, 2 records need to be removed:
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
data_dict.pop( "TOTAL", 0 )

#uncomment to show graph
# features = ["salary", "bonus"]
# data = featureFormat(data_dict, features)
# for point in data:
#     salary = point[0]
#     bonus = point[1]
#     matplotlib.pyplot.scatter( salary, bonus )
#
# matplotlib.pyplot.xlabel("salary")
# matplotlib.pyplot.ylabel("bonus")
# matplotlib.pyplot.show()


### Task 3: Create new feature(s)
## engineer one new feature: only using financial data for the project
# ratio salary/bonus
for employee, features in data_dict.iteritems():
	if features['salary'] == "NaN" or features['bonus'] == "NaN":
		features['salary_bonus_ratio'] = "NaN"
	else:
		features['salary_bonus_ratio'] = float(features['salary']) / float(features['bonus'])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# list of classifier to be tried
scaler = MinMaxScaler()
select = SelectKBest()
dtc = DecisionTreeClassifier()
svc = SVC()
knc = KNeighborsClassifier()
rfc = RandomForestClassifier()

#Loading the pipeline with the different steps
steps = [
		 # Preprocessing only for
         #('min_max_scaler', scaler), # only with k-nearest neighbours algorithm
         # Feature selection
         ('feature_selection', select), #feature selection together with dtc to select only the best features
         # Classifier
         #('dtc', dtc),
         #('rfc', rfc),
         ('knc', knc),
         ]

# Creating the pipeline
pipeline = Pipeline(steps)

# list of Parameters to test in GridsearchCV
parameters = dict(
                  feature_selection__k=[5, 6, 7, 8, 9],
                  # dtc__criterion=['entropy'], #['gini', 'entropy'],
                  # #dtc__splitter=['random'], #['best', 'random'],
                  # dtc__max_depth=[None, 1, 2, 3, 4],
                  # dtc__min_samples_split=[1, 5, 10, 20, 25, 30, 35, 40,45],
                  # dtc__class_weight=['balanced'], #[None, 'balanced'],
                  # dtc__random_state=[35, 42, 45, 50, 60],
                  knc__n_neighbors=[1, 2, 3, 4, 5,6,7],
                  knc__leaf_size=[1, 10, 30, 60],
                  knc__algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'],
                  #rfc__n_estimators=[10,20,30,40,50],
                  #rfc__max_features=['auto', 'sqrt', 'log2']
                  )


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

# Create training sets and test sets
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Cross-validation for parameter tuning in grid search
sss = StratifiedShuffleSplit(
    labels_train,
    n_iter = 20,
    test_size = 0.5,
    random_state = 0
    )

# Create, fit, and make predictions with grid search
gs = GridSearchCV(pipeline,
	              param_grid=parameters,
	              scoring="f1",
	              cv=sss,
	              error_score=0
                  )
gs.fit(features_train, labels_train)
labels_predictions = gs.predict(features_test)

# Pick the classifier with the best tuned parameters
clf = gs.best_estimator_
print "\n", "Best parameters are: ", gs.best_params_, "\n"

# Print features selected and their importances (rfc,dtc) / scores (select)
# comment if no selection process
features_selected=[features_list[i+1] for i in clf.named_steps['feature_selection'].get_support(indices=True)]
print clf.named_steps['feature_selection'].get_support(indices=True)
scores = clf.named_steps['feature_selection'].scores_
#change here the name of the classifier: dtc, rfc. but not for knc
#importances = clf.named_steps['rfc'].feature_importances_

indices = np.argsort(scores)[::-1]
print scores, features_selected
#print 'The ', len(features_selected), " features selected and their importances:"
#for i in range(len(features_selected)):
    #for dtc and rfc
    #print "|{}|{}|{}|{}|".format(i+1,features_selected[indices[i]],  round(importances[indices[i]],3), round(scores[indices[i]],3))
    #for knc only
   #print "|{}|{}|{}|".format(i+1,features_selected[indices[i]], round(scores[indices[i]],3))
   #Print classification report (focus on precision and recall)
report = classification_report( labels_test, labels_predictions )
print(report)

# ### Task 6: Dump your classifier, dataset, and features_list so anyone can
# ### check your results. You do not need to change anything below, but make sure
# ### that the version of poi_id.py that you submit can be run on its own and
# ### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
