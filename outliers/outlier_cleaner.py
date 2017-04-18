#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    import numpy as np
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    error = abs(predictions.transpose()-net_worths.transpose())[0]
    err=np.sort(error)
    print err
    age = ages.transpose()[0]
    net_worth=net_worths.transpose()[0]
    cleaned_data = np.array([tuple(i) for i in zip(age,net_worth,error)])
    cleaned_data= cleaned_data[error<=err[81]]
    print len(ages)
    print len(cleaned_data)
    print cleaned_data
    return cleaned_data

