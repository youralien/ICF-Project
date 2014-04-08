import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

from FeatureFilter import FeatureFilter
from Utils import Utils
from Visualizer import Visualizer
from Network import Network

def testTrainSplit(n, df, p):
    unique_flights = n.f.getUniqueFlights(df)
    X_train, X_test = None, None
    y_train, y_test = None, None

    for flt, flt_df in unique_flights:
        X_cat = encodeCategoricalData(flt, flt_df)
        X_flt, y = encodeFlight(flt, flt_df)
        X = np.hstack(X_cat, X_flt)

        if np.random.uniform(0, 1) <= p:
            stackMatrices(X_train, X)
            stackMatrices(y_train, y)
        else:
            stackMatrices(X_test, X)
            stackMatrices(y_test, y)

    return ((X_test, y_test), (X_train, y_train))

def encodeFlight(flt, df):
    X = None
    y = None

    for bc, bc_df in df.groupby('BC'):
        enc_bc = encodeBookingClass(bc)
        keyday = -1 * bc_df['KEYDAY']
        bkd = bc_df['BKD']
        auth = bc_df['AUTH']
        avail = bc_df['AVAIL']

        keyday, bkd, auth, avail = Utils.sortByIndex(keyday, bkd, auth, avail)

        delta_bkd = np.diff(bkd)
        delta_t = np.diff(keyday)

        features = (bkd[1:], avail[1:], auth[1:], keyday[1:], delta_t)
        features = np.column_stack(features)
        
        X = stackMatrices(X, features)
        y = stackMatrices(y, delta_bkd)

    return X, y    

def encodeCategoricalData(flt, df):
    date, flt_num, org, des = flt
    enc_date = encodeDate(date)
    X = None

    for bc, bc_df in df.groupby('BC'):
        m, n = bc_df.shape
        enc_bc = encodeBookingClass(bc)
        features = (enc_date, enc_bc)
        features = np.hstack(features)
        features = np.tile(features, (n, 1))
        stackMatrices(X, features)

    return X

def stackMatrices(x, new_x):
    if x is None:
        x = new_x
    else:
        x = np.vstack((x, new_x))

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

# for flt, flt_df in n.f.getUniqueFlights(firstflight):
    
#     adding_to_training = np.random.uniform(0, 1)  # Will this unique flight be added to Training or Test Set?
#     date, flt_num, org, des = flt
#     day_of_week = Utils.date2DayOfWeek(date)
#     enc_day_of_week = encodeDayOfWeek(day_of_week)
    
#     for bc, bc_df in flt_df.groupby('BC'):
        
#         enc_bc = encodeBookingClass(uniquebc)
        
#         keyday = np.array( -bc_df['KEYDAY']  )

#         bkd = np.array( bc_df['BKD'] )
#         auth = np.array( bc_df['AUTH'] )
#         avail = np.array( bc_df['AVAIL'] )
        
#         keyday, bkd, auth, avail = Utils.sortByIndex(keyday, bkd, auth, avail)
        
#         deltaBKD = np.diff(bkd)
#         deltaT = np.diff(keyday)
        
#         X_continuous = np.column_stack((bkd[1:],avail[1:],auth[1:],keyday[1:],deltaT))
#         m_rows, n_cols = X_continuous.shape
        
#         # Create a encoded categorical description for all the interpolated keydays in this fltbc
#         flattened_categorical_features = []
#         for i in range(m_rows):
#             categorical_features_for_row = np.array(enc_day_of_week + enc_bc) 
#             flattened_categorical_features.append(categorical_features_for_row)
#         categorical_features_matrix = np.vstack(flattened_categorical_features)
        
#         X = np.hstack((categorical_features_matrix, X_continuous))
#         y = deltaBKD
        
#         if adding_to_training:
#             X_train = np.vstack([X_train, X]) if X_train is not None else X
#             y_train = np.concatenate([y_train, y]) if y_train is not None else y
#         else:
#             X_test = np.vstack([X_test, X]) if X_test is not None else X
#             y_test = np.concatenate([y_test, y]) if y_test is not None else y
#         os.sys.stdout.write('.')
#     os.sys.stdout.write('|')

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


# model = RandomForestRegressor()
# model.fit(X_train, y_train)
# y_predict = model.predict(X_test)

# print "\nMeanAbsoluteError: " + str(meanAbsoluteError(y_test,y_predict))

# # <codecell>

# model.feature_importances_

# # <rawcell>

# # Visualize deltaBKD (y) for prediction on the test set vs the actual

# # <codecell>

# result_dir = os.path.join(os.path.abspath("."),"Results/deltaBKD-over-deltaT/RandomForestRegressor/DMMDXB/")

# KEYDAY_INDEX = -2

# index = 0
# current_snapshot = 0
# while(True):
#     current_keyday = X_test[current_snapshot, KEYDAY_INDEX]
#     keyday_vector = []
#     y_test_vector = []
#     y_predict_vector = []
#     try:
#         while X_test[current_snapshot + 1, KEYDAY_INDEX] > current_keyday:
#             keyday_vector.append(current_keyday) # Build up KEYDAY_VECTOR
#             y_test_vector.append(y_test[current_snapshot])
#             y_predict_vector.append(y_predict[current_snapshot])
#             current_snapshot += 1
#             current_keyday = X_test[current_snapshot, KEYDAY_INDEX]
#     except IndexError:
#         print "Plotting Complete"
#         break
#     plt.clf()
#     plt.hold(True)
#     plt.plot(keyday_vector, y_test_vector)
#     plt.plot(keyday_vector, y_predict_vector)
#     plt.legend(['test','predict'],loc=3)
#     plt.xlabel('-KEYDAY from Departure')
#     plt.ylabel('delta BKD')
#     plt.savefig(result_dir + str(index))
#     index += 1
#     current_snapshot += 1

# # <codecell>

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
    data_dir = os.path.join(os.path.abspath("."), "Data/")

    num_records = 1000
    data_dir = os.path.join(os.path.abspath("."), "Data/")
    normalized = "Normalized_BKGDAT_Filtered_ZeroTOTALBKD.txt"
    unnormalized = "BKGDAT_ZeroTOTALBKD.txt"
    filename = data_dir + unnormalized
    n = Network(num_records, filename)
    v = Visualizer()

    firstflight = n.f.getDrillDown(orgs=['DMM'],dests=['DXB'],cabins=["Y"])
    testTrainSplit(n, firstflight, 0.7)


if __name__ == '__main__':
    main()