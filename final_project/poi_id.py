#!/usr/bin/python
from __future__ import division
import sys
import pickle
import numpy as np
import matplotlib.pyplot
sys.path.append("../tools/")

from tester import dump_classifier_and_data, test_classifier
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import pandas as pd

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'exercised_stock_options','restricted_stock', 'restricted_stock_deferred','total_stock_value',
                 'from_messages','to_messages','from_poi_to_this_person', 'from_this_person_to_poi',
                 'shared_receipt_with_poi','from_poi_to_this_person_ratio','from_this_person_to_poi_ratio',
                 'shared_receipt_with_poi_ratio',
                 'loan_advances','expenses', 'long_term_incentive','salary', 'total_payments',
                 'bonus', 'deferral_payments', 'deferred_income','director_fees','other'] # You will need to use more features

### Load the dictionary containing the dataset
#features_list = ['poi',"long_term_incentive", "expenses"]
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#print data_dict


### Task 2: Remove outliers

## find outliers
# data = featureFormat(my_dataset, features_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)
# features_list = ['poi','salary','bonus']
# for point in data:
#     point_1 = point[1]
#     point_2 = point[2]
#     matplotlib.pyplot.scatter( point_1, point_2 )
#
# matplotlib.pyplot.xlabel(features[1])
# matplotlib.pyplot.ylabel(features[2])
# matplotlib.pyplot.show()
#
# for k,v in data_dict.items():
#     for f,s in v.items():
#         if f=="bonus" and s!= 'NaN' and s>=5000000:
#             print k

## remove outliers
remove_list = ['TOTAL','THE TRAVEL AGENCY IN THE PARK']
for name in remove_list:
    data_dict.pop(name,0)

### Task 3: Create new feature(s)

## add from_poi_to_this_person_ratio,from_this_person_to_poi_ratio,shared_receipt_with_poi_ratio
from_poi_to_this_person_ratio=[]
from_this_person_to_poi_ratio=[]
shared_receipt_with_poi_ratio=[]
for name,item in data_dict.items():
    if item['to_messages'] == 0 or item['to_messages'] == 'NaN':
        data_dict[name]['from_poi_to_this_person_ratio'] = 0
    elif item['from_poi_to_this_person'] == 0 or item['from_poi_to_this_person'] == 'NaN':
        data_dict[name]['from_poi_to_this_person_ratio'] = 0
    else:
        data_dict[name]['from_poi_to_this_person_ratio'] = data_dict[name]['from_poi_to_this_person']/data_dict[name]['to_messages']

    if item['from_messages'] == 0 or item['from_messages'] == 'NaN':
        data_dict[name]['from_this_person_to_poi_ratio'] = 0
    elif item['from_this_person_to_poi'] == 0 or item['from_this_person_to_poi'] == 'NaN':
        data_dict[name]['from_this_person_to_poi_ratio'] = 0
    else:
        data_dict[name]['from_this_person_to_poi_ratio'] = data_dict[name]['from_this_person_to_poi']/data_dict[name]['from_messages']

    if item['to_messages'] == 0 or item['to_messages'] == 'NaN':
        data_dict[name]['shared_receipt_with_poi_ratio'] = 0
    elif item['shared_receipt_with_poi'] == 0 or item['shared_receipt_with_poi'] == 'NaN':
        data_dict[name]['shared_receipt_with_poi_ratio'] = 0
    else:
        data_dict[name]['shared_receipt_with_poi_ratio'] = data_dict[name]['shared_receipt_with_poi']/data_dict[name]['to_messages']

    from_poi_to_this_person_ratio.append(data_dict[name]['from_poi_to_this_person_ratio'])
    from_this_person_to_poi_ratio.append(data_dict[name]['from_this_person_to_poi_ratio'])
    shared_receipt_with_poi_ratio.append(data_dict[name]['shared_receipt_with_poi_ratio'])

## handle Missing values
for feature in features_list[1:]:
    for key in data_dict.keys():
        if data_dict[key][feature] == 'NaN':
            data_dict[key][feature] = 0.

## clean unimportant feature
# remove_features_list = ['restricted_stock_deferred', 'from_messages', 'to_messages',
#                         'from_this_person_to_poi', 'from_poi_to_this_person_ratio',
#                         'loan_advances', 'deferral_payments', 'director_fees',
#                         'exercised_stock_options', 'restricted_stock', 'total_stock_value',
#                         'from_poi_to_this_person', 'shared_receipt_with_poi',
#                         'long_term_incentive', 'total_payments', 'deferred_income']
# features_list = [x for x in features_list if x not in remove_features_list]
# print features_list


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
scaler = MinMaxScaler()
skb = SelectKBest(f_classif)
pca = PCA()
svc = SVC()
nb = GaussianNB()
dt = DecisionTreeClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


## use train_test_split to split train and test data
# from sklearn.model_selection import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.1, random_state=42)


## k-flod cross validation
# from sklearn.cross_validation import KFold
# from sklearn.metrics import f1_score
# n_folds = 5
# kf = KFold(len(labels),n_folds)
# score=[]
#
# for train_index, test_index in kf:
#
#     features_train, features_test = np.array(features)[train_index], np.array(features)[test_index]
#     labels_train, labels_test = np.array(labels)[train_index],np.array(labels)[test_index]
#     clf.fit(features_train,labels_train)
#     print zip(features_list[1:],clf.feature_importances_ )
#     score.append(f1_score(labels_test,clf.predict(features_test)))
# print score
# print "%d-Fold-cross-validation's averge score is: %f" % (n_folds,np.average(score))



## use StratifiedShuffleSplit to split train and test data

n_folds = 1000
from sklearn.cross_validation import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(labels, n_folds, random_state = 42)
for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

### use GridSearchCV to find the best parameters

## use DecisionTreeClassifier model

# pca_params = {'PCA__n_components':(3,4,5)}
# kbest_params = {'SKB__k':(5,6,7)}
# dt_params = { 'dt__min_samples_split' : [2, 4, 6, 8, 10, 15, 20, 25, 30],
#                'dt__criterion' : ['gini', 'entropy']
# }
# pca_params.update(kbest_params)
# pca_params.update(dt_params)
# pipe= Pipeline([('scaler',scaler),("SKB", skb),("PCA", pca),("dt", dt)])
#
# clf = GridSearchCV(pipe,pca_params, scoring='f1')
# clf.fit(features_train, labels_train)
# new_pred = clf.predict(features_test)
# clf_top = clf.best_estimator_.get_params()
# print clf.best_score_
# for x in sorted(pca_params.keys()):
#     print "\t%s:%r" % (x,clf_top[x])
#
# features_selected_bool = clf.best_estimator_.named_steps['SKB'].get_support()
# features_selected_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]
# print features_selected_list

## use the best parameters

# features_list = ['poi','exercised_stock_options', 'total_stock_value', 'expenses','restricted_stock', 'from_this_person_to_poi_ratio','salary', 'bonus']
#
# pca = PCA(n_components=3)
# dt = DecisionTreeClassifier(criterion='gini',min_samples_split=2)
#
# clf = Pipeline([("minmax", scaler), ("PCA", pca), ("dt", dt)])
# clf.fit(features_train, labels_train)

## get the score
# test_classifier(clf, my_dataset, features_list)



## use svc model

# pca_params = {'PCA__n_components':(3,4,5)}
# kbest_params = {'SKB__k':(5,6,7)}
# svc_params = { 'svc__kernel':('linear', 'rbf','poly'),
#               'svc__C':(1,5,10),
#               'svc__decision_function_shape':('ovo','over','None'),
#               'svc__tol':(1e-3,1e-4,1e-5)
# }
# pca_params.update(kbest_params)
# pca_params.update(svc_params)
# pipe= Pipeline([('scaler',scaler),("SKB", skb),("PCA", pca),("svc", svc)])
#
# clf = GridSearchCV(pipe,pca_params, scoring='f1')
# clf.fit(features_train, labels_train)
# new_pred = clf.predict(features_test)
# clf_top = clf.best_estimator_.get_params()
# print clf.best_score_
# for x in sorted(pca_params.keys()):
#     print "\t%s:%r" % (x,clf_top[x])
#
# features_selected_bool = clf.best_estimator_.named_steps['SKB'].get_support()
# features_selected_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]
# print features_selected_list

# use the best parameters and get the score

features_list = ['poi','exercised_stock_options', 'restricted_stock', 'total_stock_value', 'from_this_person_to_poi_ratio', 'salary', 'bonus']

pca = PCA(n_components=5)
svc = SVC(C=10,decision_function_shape='ovo',tol=0.001,kernel='linear')

clf = Pipeline([("minmax", scaler), ("PCA", pca), ("svc", svc)])
clf.fit(features_train, labels_train)

test_classifier(clf, my_dataset, features_list)




# from sklearn.metrics import f1_score
# from sklearn.feature_selection import SelectKBest,f_classif
# selector = SelectKBest(f_classif,6)
# features_train_select = selector.fit_transform(features_train,labels_train)
# clf.fit(features_train_select,labels_train)
# features_test_select = selector.transform(features_test)
# print f1_score(labels_test,clf.predict(features_test_select))

# import matplotlib.pyplot as plt
# from sklearn.cross_validation import cross_val_score
# from sklearn.feature_selection import SelectKBest,f_classif
# k = range(1,20)
# result =[]
# for i in k:
#     selector = SelectKBest(f_classif,i)
#     features_select = selector.fit_transform(features,labels)
#     scores = cross_val_score(clf,features_select,labels,cv=5)
#     result.append(scores.mean())
# print result
# opt = np.where(result==max(result))[0]
# print "best is %d" % k[opt]
#
# plt.plot(k,result)
# plt.show()



## find unimportant features for RandomForestClassifier
# clf.fit(features_train,labels_train)
# print zip(features_list[1:],clf.feature_importances_ )
# unimportant_features = []
# for x,y in zip(features_list[1:],clf.feature_importances_ ):
#     if y<0.09:
#         unimportant_features.append(x)
# print unimportant_features



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)