import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split

from FeatureFilter import FeatureFilter
from Utils import Utils
from Visualizer import Visualizer
from Network import Network
from AirportCodes import AirportCodes

def aggregateTrainTestSplit(X, y, ids, p):
    train_X, test_X, train_y, test_y, train_ids, test_ids = train_test_split(X, y, ids, train_size=p)
    X_train, X_test, y_train, y_test, ids_train, ids_test = (None,) * 6

    for each_x, each_y, each_id in zip(train_X, train_y, train_ids):
        X_train = vStackMatrices(X_train, each_x)
        y_train = hStackMatrices(y_train, each_y)
        ids_train = vStackMatrices(ids_train, each_id)

    for each_x, each_y, each_id in zip(test_X, test_y, test_ids):
        X_test = vStackMatrices(X_test, each_x)
        y_test = hStackMatrices(y_test, each_y)
        ids_test = vStackMatrices(ids_test, each_id)

    return X_train, y_train, X_test, y_test, ids_train, ids_test


def flightSplit(unique_flights, encoder):
    flights = [encoder(flt, flt_df) for flt, flt_df in unique_flights]
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

        keyday, bkd, auth, avail, delta_t, delta_bkd = filterDataForKeyDay(
            -90, keyday[:-1], bkd[:-1], auth[:-1], avail[:-1], delta_t, delta_bkd)

        nums = (auth, avail, delta_t, bkd, keyday)
        nums = np.column_stack(nums)
        cats = encodeCategoricalData(flt, bc)
        cats = np.tile(cats, (len(nums), 1)) 

        features = hStackMatrices(cats, nums)

        X = vStackMatrices(X, features)

        y = hStackMatrices(y, delta_bkd)

        identifiers = vStackMatrices(identifiers, np.column_stack(flt+(bc,)))

    return X, y, identifiers 

def encodeInterpolatedFlight(flt, df):
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
        keyday, bkd, auth, avail = filterDataForKeyDay(-90, keyday, bkd, auth, avail)

        keyday_interp = np.arange(-90, 1, 3)
        bkd_interp, auth_interp, avail_interp = interpolate(keyday_interp, keyday, bkd, auth, avail)

        delta_bkd = np.diff(bkd_interp)

        nums = (auth_interp[:-1], avail_interp[:-1], bkd_interp[:-1], keyday_interp[:-1])
        nums = np.column_stack(nums)
        cats = encodeCategoricalData(flt, bc)
        cats = np.tile(cats, (len(nums), 1)) 

        features = hStackMatrices(cats, nums)

        X = vStackMatrices(X, features)

        y = hStackMatrices(y, delta_bkd)

        identifiers = vStackMatrices(identifiers, np.column_stack(flt+(bc,)))

    return X, y, identifiers 

def interpolate(keyday_vals, keyday, *args):
    interps = []
    for arg in args:
        interp = np.interp(keyday_vals, keyday, arg, left=0)
        interps.append(interp)
    return interps

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
    encoded_vector = [0] * len(Utils.bc_hierarchy)
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
# SO MANY NAN'S AND INF'S (DIVISION BY ZERO)
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

def cmp_deltaBKD_curve(y_test, y_pred, X_test, identifiers_test, result_dir):
    """ Compares y_test and y_pred by visualizing how close the regression was
    to predicting the deltaBKD curve for a particular identification. 
    This identification could be date, flt, org, des, bc.  Saves the plot to a 
    specified result directory. Note that the result directory is created 
    automatically if it does not already exist in the file system. """

    # For column indicies, See encodeFlight and encodeInterpolatedFlight 
    # in a line that says nums = (..., bkd, keyday)
    KEYDAY_INDEX = -1 
    BKD_INDEX = -2
    
    if not X_test[0, KEYDAY_INDEX] < 0:
        print "Feature accessed is not negative. Assert that keydays \
            start at a negative value or assert that KEYDAY_INDEX is correct"
        return

    index = 0
    current_snapshot = 0
    while(True):
        # Initialize variables
        current_keyday = X_test[current_snapshot, KEYDAY_INDEX]
        initial_bkd = X_test[current_snapshot, BKD_INDEX]    
        keyday_vector = []
        y_test_vector = []
        y_pred_vector = []
        totalbkd_test = initial_bkd
        totalbkd_pred = initial_bkd
        # Build up keyday, y_test, y_pred vectors, and totalbkd sums
        try:
            # While Keydays are ascending from -90 -> 0 (i.e. Same Flight)
            while X_test[current_snapshot + 1, KEYDAY_INDEX] > current_keyday: 
                keyday_vector.append(current_keyday)
                y_test_vector.append(y_test[current_snapshot])
                y_pred_vector.append(y_pred[current_snapshot])
                totalbkd_test += y_test[current_snapshot]
                totalbkd_pred += y_pred[current_snapshot]
                
                current_snapshot += 1
                current_keyday = X_test[current_snapshot, KEYDAY_INDEX]
        except IndexError: # Reached ending row of X_test
            print "Plotting Complete"
            break

        print identifiers_test[index,:]

        mean_percent_error = MAPE(y_test_vector, y_pred_vector)
        totalbkd_percent_error = 100*np.abs(totalbkd_pred-totalbkd_test)/float(totalbkd_test)
        # Create Figure and Save
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.hold(True)
        ax.plot(keyday_vector, y_test_vector,'b')
        ax.plot(keyday_vector, y_pred_vector, 'r')
        ax.set_title(str(identifiers_test[index,:])+ 
            "\nTOTALBKD: actual, predicted, error | {0}, {1}, {2}%".format(
                round(totalbkd_test,1),round(totalbkd_pred,1),round(totalbkd_percent_error,1)) )
        ax.legend(['test','predict'],loc=3)
        ax.set_xlabel('-KEYDAY from Departure')
        ax.set_ylabel('delta BKD')
        ax.text(0.95, 0.01, "Mean Percent Error: {}%".format(mean_percent_error), 
            verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=13)
        plt.savefig(result_dir + str(index))
        plt.close(fig)

        index += 1
        current_snapshot += 1
        

def defineWorkingDirectory():
    return os.path.abspath(".")

def defineDataDirectory(working_dir):
    return os.path.join(working_dir, "Data/")

def defineResultDirectory(working_dir, market, interpolate, regressor):
    interpolated = "Interpolated" if interpolate else ''
    return os.path.join(working_dir, "Results/Market/{0}{1}{2}/".format(market, interpolated, regressor.__name__))

def RegressOnMarket(market, encoder, regressor):
    """ market is in the form of an airport code i.e. "LHR" 
    See AirportCodes.py for encapsulation of the strings """

    working_dir = defineWorkingDirectory()
    data_dir = defineDataDirectory(working_dir)
    result_dir = defineResultDirectory(working_dir, market, False, regressor)
    ensure_dir(result_dir) # ensure directory for figures to be saved in

    print "Loading from CSV"
    num_records = 'all'
    normalized = "Normalized_BKGDAT_Filtered_ZeroTOTALBKD.txt"
    unnormalized = "BKGDAT_ZeroTOTALBKD.txt"
    filename = data_dir + unnormalized
    n = Network(num_records, filename)
    v = Visualizer()

    print "Filtering for specified Market"
    firstflight = n.f.getDrillDown(orgs=['DXB', market], dests=['DXB', market], cabins=["Y"])
    unique_flights = n.f.getUniqueFlights(firstflight)

    print "Formatting Features and Targets into train and test sets"
    X, y, ids = flightSplit(unique_flights, encoder)
    X_train, y_train, X_test, y_test, ids_train, ids_test = aggregateTrainTestSplit(X, y, ids, 0.75)
    
    print "Training the Model"
    model = regressor()
    model.fit(X_train, y_train)
    print "\nFeature Importances: [ ..., auth, avail, (deltat), bkd, keyday]", model.feature_importances_
    y_pred = model.predict(X_test)

    print "Saving figures"
    cmp_deltaBKD_curve(y_test, y_pred, X_test, ids_test, result_dir)

def main():
    RegressOnMarket(AirportCodes.London, encodeFlight, RandomForestRegressor)
    # RegressOnMarket(AirportCodes.Bangkok, encodeInterpolatedFlight, RandomForestRegressor)
    # RegressOnMarket(AirportCodes.Delhi, encodeInterpolatedFlight, RandomForestRegressor)
    # RegressOnMarket(AirportCodes.Bahrain, encodeInterpolatedFlight, RandomForestRegressor)
    # RegressOnMarket(AirportCodes.Frankfurt, encodeInterpolatedFlight, RandomForestRegressor`)

if __name__ == '__main__':
    main()

# print "\nMeanAbsoluteError: " + str(meanAbsoluteError(y_test,y_pred))

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
#             totalbkd_predict += y_pred[current_snapshot]
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
