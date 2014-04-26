import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold, train_test_split

import thinkstats2
import thinkplot

from FeatureFilter import FeatureFilter
from Utils import Utils
from Visualizer import Visualizer
from Network import Network
from AirportCodes import AirportCodes

def kFoldSplit(X, y, ids, n_folds):
    """
    args:
        X: np.array of flight feature matricies
        y: np.array of flight target vectors
        identifiers: np.array of flt identifiers
        n_folds: number of folds to split to KFold
    """
    n = len(X)
    kf = KFold(n, n_folds=n_folds, shuffle=True)
    X_train, y_train, X_test, y_test, ids_train, ids_test = (None,) * 6

    for train_index, test_index in kf:
        for each_x, each_y, each_id in zip(X[train_index], y[train_index], ids[train_index]):
            X_train = vStackMatrices(X_train, X[train_index])
            y_train = hStackMatrices(y_train, y[train_index])
            ids_train = vStackMatrices(ids_train, ids[train_index])

        for each_x, each_y, each_id in zip(X[test_index], y[test_index], ids[test_index]):
            X_test = vStackMatrices(X_test, each_x)
            y_test = hStackMatrices(y_test, each_y)
            ids_test = vStackMatrices(ids_test, each_id)
        
        yield X_train, y_train, X_test, y_test, ids_train, ids_test

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

def encodeFlights(flights, interp_params, cat_encoding):
    data = [encodeFlight(flt, flt_df, interp_params, cat_encoding) for flt, flt_df in flights]
    X, y, identifiers = zip(*data)
    return np.array(X), np.array(y), np.array(identifiers)

def encodeFlight(flt, df, interp_params, cat_encoding):
    """
    args:
        interp_params: tuple of (start, stop, number_of_points) to use in
                       interpolate
        cat_encoding: tuple of (bin_size, date_reduction) specifying how 
                      compressed the BC and day of week categorical features
                      should be in the final feature matrix
    returns:
        tuple of (features, targets, flight IDs) suitable for use in training
        and graph generation
    """
    X = None
    y = None
    identifiers = None
    bc_groupby = df.groupby('BC')
    bc_groupby = sortBCGroupby(bc_groupby)

    for bc, bc_df in bc_groupby:
        # Unpack relevant columns of the dataframe
        keyday = -1.0 * bc_df['KEYDAY']
        bkd = bc_df['BKD']
        auth = bc_df['AUTH']
        avail = bc_df['AVAIL']
        cap = bc_df['CAP']

        # Stack the numerical and categorical data into a feature matrix
        delta_bkd, nums = encodeNumericalData(
            interp_params, keyday, bkd, auth, avail, cap)
        cats = encodeCategoricalData(flt, bc, len(nums), cat_encoding)
        features = hStackMatrices(cats, nums)

        # Save the new features in the X and y sets
        X = vStackMatrices(X, features)
        y = hStackMatrices(y, delta_bkd)
        identifiers = vStackMatrices(identifiers, np.array(flt+(bc,)))

    skip = interp_params[2] - 1
    bkd_lower = extractBKDLower(X, skip, -5)
    X = colStackMatrices(X, bkd_lower)

    # Return BC Y only
    X = X[:skip,:]
    y = y[:skip]
    identifiers = identifiers[:skip]

    return X, y, identifiers

def encodeNumericalData(interp_params, keyday, bkd, auth, avail, cap):
    start, stop, num_points = interp_params
    keyday, bkd, auth, avail = Utils.sortByIndex(keyday, bkd, auth, avail)
    keyday, bkd, auth, avail, cap = filterDataForKeyDay(
        start, keyday, bkd, auth, avail, cap)
    keyday, bkd, auth, avail, cap = interpolateFlight(
        interp_params, keyday, bkd, auth, avail, cap)

    # Create any other features
    delta_bkd = np.diff(bkd)
    clf = bkd / cap

    # Stack the numerical data into a feature matrix
    nums = [each[:-1] for each in [keyday, bkd, auth, avail, cap, clf]]
    nums = np.column_stack(nums)

    return delta_bkd, nums

def extractBKDLower(X, skip, bkd_idx):
    """ calculates BKD for BC lower in the hiearchy for all interpolated 
    keydays and for all BC

    X: Feature Set of a Flight Entity, presorted by BC hiearchy and KEYDAY
    skip: number of rows between the first entries of adjacent
        BCs
    bkd_idx: column index of BKD for X feature set
    """
    m, n = X.shape
    num_BC = m / skip
    BC_by_rank = range(num_BC)
    bkd_lower = np.zeros((m,1))

    for bc_i in xrange(0,m,skip):
        for i in xrange(0,skip):
            bkd_lower[bc_i+i] = sum([X[r,bkd_idx] for r in xrange(bc_i+i+skip,m,skip)])

    return bkd_lower

def interpolateFlight(interp_params, keyday, bkd, auth, avail, cap):
    start, stop, num_points = interp_params
    keyday_vals = np.linspace(start, stop, num_points)
    bkd, auth, avail = interpolate(
        keyday_vals, keyday, bkd, auth, avail)

    cap = float(cap.iget(0))
    cap = np.array([cap] * len(keyday_vals))

    return keyday_vals, bkd, auth, avail, cap

def encodeCategoricalData(flt, bc, num_length, cat_encoding):
    """
    args:
        num_length: int, representing length of the numerical features
        cat_encoding: tuple, (bin_size, date_reduction)

    returns: 
        features, the categorical features, as a matrix appropriate
        to stack with thenumerical features 
    """

    date, flt_num, org, des = flt
    bin_size, date_reduction = cat_encoding
    enc_date = encodeDate(date, date_reduction)

    enc_bc = encodeBookingClass(bc, bin_size)
    features = (enc_date, enc_bc)
    features = np.hstack(features)
    features = np.tile(features, (num_length, 1))

    return features

def encodeDate(date, date_reduction):
    """
    args:
        date: In the form month/day/year i.e. "4/8/2014"
        date_reduction: -1 for one_hot and is_weekend
                        0 for is_weekend
                        1 for one_hot only

    returns:
        various encodings of date
    """
    day = Utils.date2DayOfWeek(date)
    one_hot_day = oneHotDay(day)
    is_weekend = [Utils.isWeekend(day)]
    
    if date_reduction == -1:
        return np.hstack((one_hot_day, is_weekend))
    
    elif date_reduction == 0:
        return is_weekend
    
    elif date_reduction == 1:
        return one_hot_day
        
def oneHotDay(day):
    """
    Returns a One Hot Encoding of specific day

    day: i.e. "Tuesday"

    return: a one hot or 1-K encoding of day of week 

    example:

    >>> oneHoteDay("Tuesday")
    [0, 0, 1, 0, 0, 0, 0]
    """
    vector = np.zeros(len(Utils.days_of_week))
    index = Utils.days_of_week.index(day)
    vector[index] = 1
    return vector

def encodeBookingClass(bc, bin_size):
    """ Returns a various encodings of BC.
    including OneHot or 1-K Methods and binning the rank
    """
    return oneHotBookingClass(bc, bin_size)

def oneHotBookingClass(bc, bin_size=1):
    """ Returns a binned 1-to-K or one-hot encoding of BC.
    
    bc: a booking class letter
    bin_size: number of bc that fit into one bin
    
    if bin_size=1 (default), we have a true 1-K encoding
    """
    assert len(Utils.bc_hierarchy) % bin_size == 0, 
        "Error: Bin Size must evenly divide into the number of booking classes"

    cabin, rank = Utils.mapBookingClassToCabinHierarchy(bc)
    
    enc_vector = np.zeros(len(Utils.bc_hierarchy)/bin_size)
    enc_vector[rank/bin_size] = 1

    return enc_vector

def sortBCGroupby(groupby):
    tups = [(bc, bc_df) for bc, bc_df in groupby]
    return sorted(tups, key=lambda tup: Utils.compareBCs(tup[0]))

def interpolate(keyday_vals, keydays, *args):
    interps = [np.interp(keyday_vals, keydays, arg, left=0) for arg in args]

    return interps

def filterDataForKeyDay(time, keydays, *args):
    index = next((i for i, k in enumerate(keydays) if k > time))
    filtered_keydays = keydays[index:]
    filtered_args = [arg[index:] for arg in args]
    return [filtered_keydays] + filtered_args

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
    automatically if it does not already exist in the file system. 


    return: pred_minus_actual TOTALBKD, to be turned into a distribution
            index, the total number of flight/bc pairs 
    """

    # Feature Indicies Determined in encodeFlight
    KEYDAY_INDEX = 32
    BKD_INDEX = 33
    pred_minus_actual = [] # TOTALBKD

    if not X_test[0, KEYDAY_INDEX] < 0:
        print "Feature accessed is not negative. Assert that keydays \
            start at a negative value or assert that KEYDAY_INDEX is correct"
        return

    index = 0
    current_snapshot = 0
    
    while(True):
        current_keyday = X_test[current_snapshot, KEYDAY_INDEX]
        initial_bkd = X_test[current_snapshot, BKD_INDEX]    
        keyday_vector = []
        y_test_vector = []
        y_pred_vector = []
        bkd_test = [initial_bkd]
        bkd_pred = [initial_bkd]
        
        # Build up keyday, y_test, y_pred vectors, and totalbkd sums
        try:
            # While Keydays are ascending from -90 -> 0 (i.e. Same Flight)
            while X_test[current_snapshot + 1, KEYDAY_INDEX] > current_keyday: 
                keyday_vector.append(current_keyday)
                y_test_vector.append(y_test[current_snapshot])
                y_pred_vector.append(y_pred[current_snapshot])
                bkd_test.append(bkd_test[-1]+y_test[current_snapshot])
                bkd_pred.append(bkd_pred[-1]+y_pred[current_snapshot])
                
                current_snapshot += 1
                current_keyday = X_test[current_snapshot, KEYDAY_INDEX]
        except IndexError: # Reached ending row of X_test
            print "Plotting Complete"
            break

        totalbkd_pred = bkd_pred.pop()
        totalbkd_test = bkd_test.pop()
        pred_minus_actual.append(totalbkd_pred - totalbkd_test)

        if index < 100:
            mean_percent_error = MAPE(y_test_vector, y_pred_vector)
            totalbkd_percent_error = 100*np.abs(totalbkd_pred-totalbkd_test)/float(totalbkd_test)
            # Create Figure and Save
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax1 = fig.add_subplot(2,1,1)
            ax2 = fig.add_subplot(2,1,2)

            # Turn off axis lines and ticks of the big subplot
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

            plt.hold(True)

            ax1.plot(keyday_vector, y_test_vector,'b')
            ax1.plot(keyday_vector, y_pred_vector, 'r')
            ax2.plot(keyday_vector, bkd_test,'b')
            ax2.plot(keyday_vector, bkd_pred, 'r')


            fig.suptitle(str(identifiers_test[index,:])+ 
                "\nTOTALBKD: actual, predicted, error | {0}, {1}, {2}%".format(
                    round(totalbkd_test,1),round(totalbkd_pred,1),round(totalbkd_percent_error,1)) )
            
            ax1.legend(['test','predict'],loc=3)
            ax1.set_xlabel('-KEYDAY from Departure')
            ax1.set_ylabel('delta BKD')
            ax1.text(0.95, 0.01, "Mean Percent Error: {}%".format(mean_percent_error), 
                verticalalignment='bottom', horizontalalignment='right', transform=ax1.transAxes, color='green', fontsize=13)

            ax2.legend(['test','predict'],loc=3)
            ax2.set_xlabel('-KEYDAY from Departure')
            ax2.set_ylabel('BKD')
            # ax1.text(0.95, 0.01, "Mean Percent Error: {}%".format(mean_percent_error), 
                # verticalalignment='bottom', horizontalalignment='right', transform=ax2.transAxes, color='green', fontsize=13)

            plt.savefig(result_dir + str(index), bbox_inches='tight')
            plt.close(fig)

        index += 1
        current_snapshot += 1
    
    return pred_minus_actual, index      

def defineWorkingDirectory():
    return os.path.abspath(".")

def defineDataDirectory(working_dir):
    return os.path.join(working_dir, "Data/")

def defineResultDirectory(working_dir, market, interpolate, regressor):
    interpolated = "Interpolated" if interpolate else ''
    return os.path.join(working_dir, 
        "Results/Market/{0}{1}{2}/".format(
            market, str(regressor).split("(")[0], interpolated))

def RegressOnMarket(market, encoder, model, interpolate):
    """ market is in the form of an airport code i.e. "LHR" 
    See AirportCodes.py for encapsulation of the strings """

    working_dir = defineWorkingDirectory()
    data_dir = defineDataDirectory(working_dir)
    result_dir = defineResultDirectory(working_dir, market, interpolate, model)
    ensure_dir(result_dir) # ensure directory for figures to be saved in

    print "Loading from CSV"
    num_records = 'all'
    normalized = "Normalized_BKGDAT_Filtered_ZeroTOTALBKD.txt"
    unnormalized = "BKGDAT_ZeroTOTALBKD.txt"
    filename = data_dir + unnormalized
    n = Network(num_records, filename)
    v = Visualizer()

    print "Filtering for specified Market"
    if market is not None:
        firstflight = n.f.getDrillDown(orgs=['DXB', market], dests=['DXB', market], cabins=["Y"])
    else:
        firstflight = n.f.getDrillDown(cabins=["Y"])
    unique_flights = n.f.getUniqueFlights(firstflight)

    print "Formatting Features and Targets into train and test sets"
    X, y, ids = flightSplit(unique_flights, encoder)
    X_train, y_train, X_test, y_test, ids_train, ids_test = aggregateTrainTestSplit(X, y, ids, 0.50)
    
    x = X_train
    y = y_train

    keyday, bkd, auth, avail, cap, clf = (-6, -5, -4, -3, -2, -1)

    for i, var in enumerate([keyday, bkd, auth, avail, cap, clf]):
        print x.shape
        print y.shape
    #     thinkplot.SubPlot(2,3,i+1)
    #     thinkplot.Scatter(x[:,var], y)

    assert False, "Testing Sequence Over"

    print "Training the Model"
    model.fit(X_train, y_train)

    try:
        print "\nFeature Importances: [ ..., auth, avail, (deltat), bkd, keyday]" + str(model.feature_importances_)
    except AttributeError:
        try:
            print "\nRegression Coeficients: [ ..., auth, avail, (deltat), bkd, keyday]"+str(model.coef_)
        except AttributeError:
            print "\nCannot express weights"
    y_pred = model.predict(X_test)

    print "Calculating Errors Saving figures"
    pred_minus_actual, num_uniqueids = cmp_deltaBKD_curve(y_test, y_pred, X_test, ids_test, result_dir)
    
    cdf = thinkstats2.MakeCdfFromList(pred_minus_actual)

    thinkplot.Cdf(cdf)

    thinkplot.Show(title="Number of Flt-BC: {}\n {}".format(num_uniqueids, result_dir),
        xlabel="TotalBKD Error (Predicted - Actual)",ylabel="Probability")

def main():
    # RegressOnMarket(AirportCodes.London, encodeInterpolatedFlight, KNeighborsRegressor(n_neighbors=5), True)
    RegressOnMarket(AirportCodes.London, encodeInterpolatedFlight, RandomForestRegressor(), True)
    # RegressOnMarket(AirportCodes.London, encodeInterpolatedFlight, AdaBoostRegressor(DecisionTreeRegressor(),n_estimators=300), True)
    # RegressOnMarket(AirportCodes.London, encodeInterpolatedFlight, GradientBoostingRegressor(), True)
    # RegressOnMarket(AirportCodes.Bangkok, encodeInterpolatedFlight, RandomForestRegressor(), True)
    # RegressOnMarket(AirportCodes.Delhi, encodeInterpolatedFlight, RandomForestRegressor(), True)
    # RegressOnMarket(AirportCodes.Bahrain, encodeInterpolatedFlight, RandomForestRegressor(), True)
    # RegressOnMarket(AirportCodes.Frankfurt, encodeInterpolatedFlight, RandomForestRegressor(), True)
    return

if __name__ == '__main__':
    main()
