import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from FeatureFilter import FeatureFilter
from Utils import Utils
from Visualizer import Visualizer
from Network import Network

def testTrainSplit(n, df, p):
    unique_flights = n.f.getUniqueFlights(df)
    X_train, X_test = None, None
    y_train, y_test = None, None
    identifiers_train, identifiers_test = None, None

    for flt, flt_df in unique_flights:
        X, y, identifiers = encodeFlight(flt, flt_df)

        if np.random.uniform(0, 1) <= p:
            X_train = vStackMatrices(X_train, X)
            y_train = hStackMatrices(y_train, y)
            identifiers_train = vStackMatrices(identifiers_train, identifiers)
        else:
            X_test = vStackMatrices(X_test, X)
            y_test = hStackMatrices(y_test, y)
            identifiers_test = vStackMatrices(identifiers_test, identifiers)

    return ((X_train, y_train, identifiers_train), (X_test, y_test, identifiers_test))

def encodeFlight(flt, df):
    X = None
    y = None
    identifiers = None

    for bc, bc_df in df.groupby('BC'):
        enc_bc = encodeBookingClass(bc)
        keyday = -1 * bc_df['KEYDAY']
        bkd = bc_df['BKD']
        auth = bc_df['AUTH']
        avail = bc_df['AVAIL']

        keyday, bkd, auth, avail = Utils.sortByIndex(keyday, bkd, auth, avail)

        delta_bkd = np.diff(bkd)
        delta_t = np.diff(keyday)

        keyday, bkd, avail, auth, delta_t, delta_bkd = filterDataForKeyDay(-90, keyday[1:], bkd[1:], avail[1:], auth[1:], delta_t, delta_bkd)

        nums = (bkd, avail, auth, keyday, delta_t)
        nums = np.column_stack(nums)
        cats = encodeCategoricalData(flt, bc)
        cats = np.tile(cats, (len(nums), 1)) 

        features = hStackMatrices(cats, nums)

        X = vStackMatrices(X, features)

        y = hStackMatrices(y, delta_bkd)

        identifiers = vStackMatrices(identifiers, np.column_stack(flt+(bc,)))

    return X, y, identifiers 

def filterDataForKeyDay(time, keydays, *args):
    index = [i for i, k in enumerate(keydays) if k > time][0]
    return [keydays[index:]] + [arg[index:] for arg in args]

def encodeCategoricalData(flt, bc):
    date, flt_num, org, des = flt
    enc_date = encodeDate(date)

    enc_bc = encodeBookingClass(bc)
    features = (enc_date, enc_bc)
    features = np.hstack(features)

    return features

def vStackMatrices(x, new_x):
    return stackMatrices(x, new_x, np.vstack)

def hStackMatrices(x, new_x):
    return stackMatrices(x, new_x, np.hstack)

def colStackMatrices(x, new_x):
    return stackMatrices(x, new_x, np.column_stack)

def stackMatrices(x, new_x, fun):
    if x is None:
        x = new_x
    else: 
        x = fun((x, new_x))

    return x

def encodeDate(date):
    """
    Returns a 1-to-K encoding of DATE. 
    
    example:
    date = "4/8/2014"
    day = "Tuesday"
    returns [0, 0, 1, 0, 0, 0, 0]
    """
    day = Utils.date2DayOfWeek(date)
    index = Utils.days_of_week.index(day)
    vector = [0] * len(Utils.days_of_week)
    vector[index] = 1
    return vector

def encodeBookingClass(bc):
    """ Returns a 1-to-K encoding of BC."""
    cabin, rank = Utils.mapBookingClassToCabinHierarchy(bc)
    encoded_vector = [0] * len(Utils.bc_economy_hierarchy)
    encoded_vector[rank] = 1
    return encoded_vector

def meanAbsoluteError(ground_truth, predictions):
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    diff = np.abs(ground_truth - predictions)
    return np.sum(diff)/np.size(diff)

# WHAT ARE WE DOING HERE AHHHHHHHHHHHHHHHHH
def meanPercentError(actual, predicted):
    actual = 1.0 * np.array(actual)
    predicted = 1.0 * np.array(predicted)
    percent = (actual - predicted) / actual
    return np.sum(percent) / np.size(diff)

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def cmp_deltaBKD_curve(y_test, y_predict, X_test, identifiers_test, result_dir):
    ensure_dir(result_dir) # ensure directory for figures to be saved in

    KEYDAY_INDEX = -2 # keyday is located on the 2nd the last column of X
    index = 0
    current_snapshot = 0
    while(True):
        # Initialize variables
        current_keyday = X_test[current_snapshot, KEYDAY_INDEX]
        keyday_vector = []
        y_test_vector = []
        y_predict_vector = []
        
        try:
            while X_test[current_snapshot + 1, KEYDAY_INDEX] > current_keyday:
                keyday_vector.append(current_keyday)
                y_test_vector.append(y_test[current_snapshot])
                y_predict_vector.append(y_predict[current_snapshot])
                current_snapshot += 1
                current_keyday = X_test[current_snapshot, KEYDAY_INDEX]
        except IndexError: # Reached ending row of X_test
            print "Plotting Complete"
            break
        # Create Figure and Save
        plt.clf()
        plt.hold(True)
        plt.plot(keyday_vector, y_test_vector)
        plt.plot(keyday_vector, y_predict_vector, '.-')
        plt.title(identifiers_test[index,:]) # Identifier could be (date, flt, org, des, bc) for one row
        plt.legend(['test','predict'],loc=3)
        plt.xlabel('-KEYDAY from Departure')
        plt.ylabel('delta BKD')
        plt.savefig(result_dir + str(index))
        index += 1
        current_snapshot += 1

# print "\nMeanAbsoluteError: " + str(meanAbsoluteError(y_test,y_predict))

# # <codecell>

# model.feature_importances_

# # <rawcell>

# # Visualize deltaBKD (y) for prediction on the test set vs the actual

# # <codecell>

# result_dir = os.path.join(os.path.abspath("."),"Results/deltaBKD-over-deltaT/RandomForestRegressor/DMMDXB/")




# KEYDAY_INDEX = -2
# BKD_INDEX = -5
# totalbkd_test_vector = []
# totalbkd_predict_vector = []

# index = 0
# current_snapshot = 0
# while(True):
#     initial_bkd = X_test[current_snapshot, BKD_INDEX]
#     current_keyday = X_test[current_snapshot, KEYDAY_INDEX]
#     totalbkd_test = initial_bkd
#     totalbkd_predict = initial_bkd
#     try:
#         while X_test[current_snapshot + 1, KEYDAY_INDEX] > current_keyday:
#             totalbkd_test += y_test[current_snapshot]
#             totalbkd_predict += y_predict[current_snapshot]
#             current_snapshot += 1
#             current_keyday = X_test[current_snapshot, KEYDAY_INDEX]
#     except IndexError:
#         print ("TotalBKD from deltaBKD summation complete")
#         break
#     totalbkd_test_vector.append(totalbkd_test)
#     totalbkd_predict_vector.append(totalbkd_predict)
#     index += 1
#     current_snapshot += 1

# # <codecell>

# print "\nMean Absolute Error of Calculating TotalBKD for a particular Booking Class: " + str(meanAbsoluteError(totalbkd_test_vector, totalbkd_predict_vector))

# # <codecell>


# total = float(len(cum_deltabkd))
# mid = len([x for x in cum_deltabkd if x >= 0 and x < 3])
# neg = len([x for x in cum_deltabkd if x < 0])
# pos = len([x for x in cum_deltabkd if x > 3])

# print neg / total
# print mid / total
# print pos / total

# # <codecell>

# import thinkstats2
# import thinkplot

# cdf = thinkstats2.MakeCdfFromList(cum_deltabkd_obsrvd)

# # cdf = thinkstats2.MakeCdfFromList([x for x in cum_deltabkd if x >= 0 and x < 2],'Stevie Cum')
# thinkplot.Cdf(cdf)
# thinkplot.show()

# # <codecell>

# plt.plot(keyday_interp, bkd_interp, '.')
# plt.show()


def main():
    wd = os.path.abspath(".")
    data_dir = os.path.join(wd, "Data/")

    num_records = 50000
    data_dir = os.path.join(wd, "Data/")
    normalized = "Normalized_BKGDAT_Filtered_ZeroTOTALBKD.txt"
    unnormalized = "BKGDAT_ZeroTOTALBKD.txt"
    filename = data_dir + unnormalized
    n = Network(num_records, filename)
    v = Visualizer()

    firstflight = n.f.getDrillDown(orgs=['DMM'],dests=['DXB'],cabins=["Y"])
    (X_train, y_train, identifiers_train), (X_test, y_test, identifiers_test) = testTrainSplit(n, firstflight, 0.66)

    # print X_train.shape, y_train.shape, identifiers_train.shape, X_test.shape, y_test.shape, identifiers_test.shape
    # print identifiers_test
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result_dir = os.path.join(wd, "Results/Market/DXBDMM/")
    cmp_deltaBKD_curve(y_test, y_pred, X_test, identifiers_test, result_dir)

if __name__ == '__main__':
    main()
