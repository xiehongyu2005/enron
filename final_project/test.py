import sys
import pickle
sys.path.append("../tools/")
import pandas as pd

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','shared_receipt_with_poi', 'to_messages', 'total_payments',
                 'total_stock_value'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


print list(pd.DataFrame(data_dict).index)
