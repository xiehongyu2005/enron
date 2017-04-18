#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
from __future__ import division
import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


count=0
for k,v in enron_data.items():
    count = count+1
    if count==66:
        print k
#        for f,s in v.items():
#           if f == 'total_payments' and s == 'NaN':
#               count=count+1
# print count/len(enron_data)
# print len(enron_data)
#
# print enron_data[str.upper("FASTOW ANDREW S")]

