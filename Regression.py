import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

from FeatureFilter import FeatureFilter
from Utils import Utils
from Visualizer import Visualizer
from Network import Network

def KFoldCV(X, y, identifiers, n_folds):
    kf = KFold(len(X), n_folds, indices=True)

    for train, test in kf:
        X_train, y_train, X_test, y_test = None, None, None, None
        for each_x, each_y in zip(X[train], y[train]):
            X_train = vStackMatrices(X_train, each_x)
            y_train = hStackMatrices(y_train, each_y)

        for each_x, each_y in zip(X[test], y[test]):
            X_test = vStackMatrices(X_test, each_x)
            y_test = hStackMatrices(y_test, each_y)

def flightSplit(unique_flights):
    flights = [encodeFlight(flt, flt_df) for flt, flt_df in unique_flights]
    X, y, identifiers = zip(*flights)

    return np.array(X), np.array(y), np.array(identifiers)

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
    """ Returns a 1-to-K or one-hot encoding of BC."""
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
# We are caluating percent error element wise (which corresponds to percent error for each snapshot in the dataset)
# Then we return the mean Percent error
# SO MANY FUCKING NAN'S AND INF'S (DIVISION BY ZERO)
def MAPE(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    percent = np.zeros(len(actual))
    for i in xrange(len(actual)):
        if float(actual[i]) == 0.0:
            percent[i] = np.abs((actual[i] - predicted[i]) / 1) # Funky Fix by dividing by 1?
        else:
            percent[i] = np.abs((actual[i] - predicted[i]) / actual[i])
    return np.around(100.0 * np.mean(percent), decimals=1)

# MAPE_alt (as found from Wikipedia) still throws NaN's
def MAPE_alt(actual, predicted):
    """ The difference with the original formula is that 
    each Actual Value (At) of the series is replaced by 
    the average Actual Value (At_bar) of that series. Hence, 
    the distortions are smoothed out."""
    actual, predicted = np.array(actual), np.array(predicted)
    average_actual = np.array([np.mean(actual) for i in xrange(len(actual))])
    percent = np.abs((average_actual - predicted) / average_actual)
    return np.around(100.0 * np.mean(percent), decimals=1)

def ensure_dir(f):
    """ Ensures the directory exists within the filesystem.
    If it does not currently exists, the directory is made """
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def cmp_deltaBKD_curve(y_test, y_predict, X_test, identifiers_test, result_dir):
    """ Compares y_test and y_predict by visualizing how close the regression was
    to predicting the deltaBKD curve for a particular identification. 
    This identification could be date, flt, org, des, bc.  Saves the plot to a 
    specified result directory. Note that the result directory is created 
    automatically if it does not already exist in the file system. """

    ensure_dir(result_dir) # ensure directory for figures to be saved in

    KEYDAY_INDEX = -2 # keyday is located on the 2nd the last column of X
    if not X_test[0, KEYDAY_INDEX] < 0:
        print "Keyday feature is not properly setup. Check if Keydays start negative and approach 0 near departure"
        return

    index = 0
    current_snapshot = 0
    while(True):
        # Initialize variables
        current_keyday = X_test[current_snapshot, KEYDAY_INDEX]
        keyday_vector = []
        y_test_vector = []
        y_predict_vector = []
        
        # Build up keyday, y_test, y_predict vectors
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

        mean_percent_error = MAPE_alt(y_test_vector, y_predict_vector)
        print mean_percent_error
        # Create Figure and Save
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.hold(True)
        ax.plot(keyday_vector, y_test_vector,'b')
        ax.plot(keyday_vector, y_predict_vector, 'r')
        ax.set_title(identifiers_test[index,:]) # Identifier could be (date, flt, org, des, bc) for one row
        ax.legend(['test','predict'],loc=3)
        ax.set_xlabel('-KEYDAY from Departure')
        ax.set_ylabel('delta BKD')
        ax.text(0.95, 0.01, "Mean Percent Error: {}%".format(mean_percent_error), 
            verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=13)
        plt.savefig(result_dir + str(index))
        plt.close(fig)

        index += 1
        current_snapshot += 1

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
    unique_flights = n.f.getUniqueFlights(firstflight)
    X, y, identifiers = flightSplit(unique_flights)
    KFoldCV(X, y, identifiers, 3)
        
    # (X_train, y_train, identifiers_train), (X_test, y_test, identifiers_test) = flightSplit(n, firstflight, 0.66)

    # model = RandomForestRegressor()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    # result_dir = os.path.join(wd, "Results/Market/DXBDMM/")
    # cmp_deltaBKD_curve(y_test, y_pred, X_test, identifiers_test, result_dir)

if __name__ == '__main__':
    main()

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
