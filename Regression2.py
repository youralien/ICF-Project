import pandas as pd
import numpy as np

from sklearn.cross_validation import KFold

from FeatureFilter import FeatureFilter
from Utils import Utils
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
        return one_hot_day + is_weekend
    
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
    assert len(Utils.bc_hierarchy) % bin_size == 0, "Invalid Bin Size"

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

def main():
    # Set parameters for loading the data
    num_records = 'all'
    csvfile = "Data/BKGDAT_ZeroTOTALBKD.txt"

    # Set parameters for filtering the data
    market = AirportCodes.London
    orgs=[AirportCodes.Dubai, market]
    dests=[AirportCodes.Dubai, market]
    cabins=["Y"]

    # Get the data, filter it, and group it by flight
    print "Loading " + csvfile
    f = FeatureFilter(num_records, csvfile)

    print "Filtering"
    data = f.getDrillDown(orgs=orgs, dests=dests, cabins=cabins)

    print "Grouping by flight"
    unique_flights = f.getUniqueFlights(data)

    # Encode the flights
    print "Encoding flight data"
    start = -90
    stop = 0
    num_points = 31

    interp_params = (start, stop, num_points)

    bin_size = 1
    date_reduction = -1
    cat_encoding = (bin_size, date_reduction)

    x, y, i = encodeFlights(unique_flights, interp_params, cat_encoding)
    foo = next(kFoldSplit(x, y, i, 3))
    print foo[0]
    print foo[4]

if __name__ == '__main__':
    main()
