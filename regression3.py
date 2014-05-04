import sklearn_pandas
import pandas as pd
import numpy as np
import thinkplot

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold, train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib

from FeatureFilter import FeatureFilter
from Utils import Utils
from AirportCodes import AirportCodes

def aggregateTrainTestSplit(X, y, ids, p):
    """
    args:
        X: np.array of flight feature matricies
        y: np.array of flight target vectors
        ids: np.array of flt identifiers
        p: a float percentage of the training set size
    """
    train_X, test_X, train_y, test_y, train_ids, test_ids = train_test_split(X, y, ids, train_size=p)
    
    X_train, y_train, ids_train = aggregate(train_X, train_y, train_ids)

    X_test, y_test, ids_test = aggregate(test_X, test_y, test_ids)

    return X_train, y_train, X_test, y_test, ids_train, ids_test

def aggregate(X, y, ids):
    """
    aggregate the flight features, targets, and ids  
    args:
        X: np.array of flight feature matricies
        y: np.array of flight target vectors
        ids: np.array of flt identifiers
    returns:
        X, y, and ids as aggregated 2 dimensional arrays
    """
    X_tmp, y_tmp, ids_tmp = (None, ) * 3

    for each_X, each_y, each_id in zip(X, y, ids):
        X_tmp = vStackMatrices(X_tmp, each_X)
        y_tmp = hStackMatrices(y_tmp, each_y)
        ids_tmp = vStackMatrices(ids_tmp, each_id)

    return X_tmp, y_tmp, ids_tmp

def encodeFlights2(flights, interp_params, df_mapper):
    data = [encodeFlight2(flt, flt_df, interp_params, df_mapper) for flt, flt_df in flights]
    X, y, identifiers = zip(*data)

    print 'encodeFlights:', len(y), len(identifiers)
    return np.array(X), np.array(y), np.array(identifiers)

def encodeFlights(flights, interp_params, cat_encoding):
    data = [encodeFlight(flt, flt_df, interp_params, cat_encoding) for flt, flt_df in flights]
    X, y, identifiers = zip(*data)

    print len(y[0]) == interp_params[2]

    return np.array(X), np.array(y), np.array(identifiers)

def encodeFlight2(flt, df, interp_params, df_mapper):
    """
    args:
        interp_params: tuple of (start, stop, number_of_points) to use in
                       interpolate
        df_mapper: a sklearn_pandas DataFrameMapper object
    returns:
        tuple of (features, targets, flight IDs) suitable for use in training
        and graph generation
    """
    X = None
    y = None
    ids = None
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
        cats = df_mapper.transform(bc_df)[0,:]
        m_rows, n_columns = nums.shape
        cats = np.tile(cats, (m_rows,1))
        features = hStackMatrices(cats, nums)
        identifiers = encodeIdentifier(flt, bc)


        # Save the new features in the X and y sets
        X = vStackMatrices(X, features)
        y = hStackMatrices(y, delta_bkd)
        ids = vStackMatrices(ids, identifiers)
    
    skip = interp_params[2] - 1
    
    bkd_idx = -4
    bkd_lower = extractBKDLower(X, skip, bkd_idx)
    X = colStackMatrices(X, bkd_lower)

    return X, y, ids


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
    ids = None
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
        cats = encodeCategoricalData(flt, bc, len(delta_bkd), cat_encoding)
        features = hStackMatrices(cats, nums)
        identifiers = encodeIdentifier(flt, bc)


        # Save the new features in the X and y sets
        X = vStackMatrices(X, features)
        y = hStackMatrices(y, delta_bkd)
        ids = vStackMatrices(ids, identifiers)
    
#     skip = interp_params[2] - 1
    
#     bkd_idx = 33
#     bkd_lower = extractBKDLower(X, skip, bkd_idx)
#     X = colStackMatrices(X, bkd_lower)

#     norm_bkd_idx = 37
#     norm_bkd_lower = extractBKDLower(X, skip, norm_bkd_idx)
#     X = colStackMatrices(X, norm_bkd_lower)
   
    # # Return BC Y only
    # X = X[:skip,:]
    # y = y[:skip]
    # identifiers = identifiers[:skip]

    return X, y, ids

def encodeIdentifier(flt, bc):
    identifier = np.array(flt + (bc,))
    return identifier

def encodeNumericalData(interp_params, keyday, bkd, auth, avail, cap):
    start, stop, num_points = interp_params
    keyday, bkd, auth, avail = Utils.sortByIndex(keyday, bkd, auth, avail)
    keyday, bkd, auth, avail, cap = filterDataForKeyDay(
        start, keyday, bkd, auth, avail, cap)
    keyday, bkd, auth, avail, cap = interpolateFlight(
        interp_params, keyday, bkd, auth, avail, cap)

    # Create any other features
    delta_bkd = np.diff(bkd)

    # Stack the numerical data into a feature matrix
    nums = [each[:-1] for each in [keyday, bkd, auth, avail, cap]]
    nums = np.column_stack(nums)

    return delta_bkd, nums

def interpolateFlight(interp_params, keyday, bkd, auth, avail, cap):
    start, stop, num_points = interp_params

    # We add one to num_points because we take one data point off all of the
    # feature vectors when we diff for deltabkd. This way num_points end up
    # actually being the number of points in the feature vector, as opposed to
    # being one greater than the length of the feature vector
    keyday_vals = np.linspace(start, stop, num_points+1)
    bkd, auth, avail = interpolate(
        keyday_vals, keyday, bkd, auth, avail)

    cap = float(cap.iget(0))
    cap = np.array([cap] * len(keyday_vals))

    return keyday_vals, bkd, auth, avail, cap

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

def scale_trans_nums(scaler, X, vsplit):
    """
    Scale transforms the numerical part of the feature set
    args:
        scaler: StandardScaler object
        X: the m by n feature matrix, where m is the number of training examples
           and n is the number of features
        vsplit: an int which is the column index of where either the categorical 
                features end or numerical features start.
        kwargs: to modify behavior of sklearn's StandardScaler object
    returns:
        scaler: StandardScaler object
        X: the transformed feature set, with numerical features scaled
    """
    
    cats, nums = cats_nums_split(X, vsplit)

    try:
        # Has the scaler been fitted yet?
        _ = scaler.mean_
        
        # Fitted: Yes
        nums = scaler.transform(nums)
    except AttributeError:
        # Fitted: No
        nums = scaler.fit_transform(nums)

    return scaler, hStackMatrices(cats, nums)

def scale_invtrans_nums(scaler, X, vsplit):
    """
    inverse Scale transforms the numerical part of the transformed feature set
    args:
        scaler: StandardScaler object
        X: the m by n feature matrix, where m is the number of training examples
           and n is the number of features
        vsplit: an int which is the column index of where either the categorical 
                features end or numerical features start.
        kwargs: to modify behavior of sklearn's StandardScaler object
    returns:
        scaler: StandardScaler object
        X: the original feature set, with numerical features inverse scaled
    """
    
    cats, nums = cats_nums_split(X, vsplit)

    nums = scaler.inverse_transform(nums)

    return scaler, hStackMatrices(cats, nums)

def cats_nums_split(X,vsplit):
    """ Splits the feature matrix into categorical and numerical feature
    matricies
    args:
        X: the m by n feature matrix, where m is the number of training examples
           and n is the number of features
        vsplit: an int which is the column index of where either the categorical 
                features end or numerical features start.
    returns:
        a tuple (cats, nums) which are the categorical and numerical feature 
        matrices
    """
    cats = X[:,:vsplit]
    nums = X[:,vsplit:]
    return cats, nums

def mainRyan():
    # Set parameters for loading the data
    num_records = 'all'
    csvfile = "Data/BKGDAT_ZeroTOTALBKD.txt"

    # Set parameters for filtering the data
    market = AirportCodes.Frankfurt
    cabins=["Y"]

    # Get the data, filter it, and group it by flight
    print "Loading " + csvfile
    f = FeatureFilter(num_records, csvfile)

    print "Filtering"
    if market is None:
        orgs=[AirportCodes.Dubai, AirportCodes.London, AirportCodes.Bahrain, AirportCodes.Frankfurt, AirportCodes.Bangkok]
        dests=[AirportCodes.Dubai, AirportCodes.London, AirportCodes.Bahrain, AirportCodes.Frankfurt, AirportCodes.Bangkok]
        data = f.getDrillDown(orgs=orgs, dests=dests, cabins=cabins)
    else:
        orgs=[AirportCodes.Dubai, market]
        dests=[AirportCodes.Dubai, market]
        data = f.getDrillDown(orgs=orgs, dests=dests, cabins=cabins)

    print "Grouping by flight"
    unique_flights = f.getUniqueFlights(data)

    # Encode the flights
    print "Encoding flight data"
    start = -15
    stop = 0
    num_points = 15
    interp_params = (start, stop, num_points)
    
    bin_size = 3
    date_reduction = 0
    cat_encoding = (bin_size, date_reduction)

    X, y, ids = encodeFlights(unique_flights, interp_params, cat_encoding)
    print 'yearrrrghhh', y[0].shape, num_points
    X_train, y_train, X_test, y_test, ids_train, ids_test = aggregateTrainTestSplit(X, y, ids, 0.75)

    return X_train, y_train, X_test, y_test, ids_train, ids_test, interp_params, cat_encoding

def regress(model, res):
    """ 
    args:
        model: Instance of an sklearn estimator
        res: the resulting tuple passed out of mainRyan
    """
    X_train, y_train, X_test, y_test, ids_train, ids_test, inter_params, cat_encoding = res
    
    print "Training the Regressor"
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return y_test, y_pred

class Error:
    
    @staticmethod
    def bkd_at_each_keyday(y_test, y_pred, ids_test, interp_params):
        """
        args:
            y_pred:
            y_test:
            interp_params:
        """
    
        start, stop, num_points = interp_params
        
        y_pred_cumsum = None
        y_real_cumsum = None
        
        for i in xrange(len(ids_test)):
            
            y_real_this_flt = y_test[(i)*num_points:(i+1)*num_points]
            y_pred_this_flt = y_pred[(i)*num_points:(i+1)*num_points]
            
            print np.array([np.cumsum(y_real_this_flt)]).shape, np.array([np.cumsum(y_pred_this_flt)]).shape
            
            y_real_cumsum = vStackMatrices(y_real_cumsum, np.array([np.cumsum(y_real_this_flt)]))
            y_pred_cumsum = vStackMatrices(y_pred_cumsum, np.array([np.cumsum(y_pred_this_flt)])) 
            
            print y_real_cumsum.shape, y_pred_cumsum.shape
            print ""
    
        return y_real_cumsum, y_pred_cumsum

if __name__ == '__main__':
    res = mainRyan()
    X_train, y_train, X_test, y_test, ids_train, ids_test, interp_params, cat_encoding = res
    y_test, y_pred = regress(KNeighborsRegressor(), res)
    X_train, y_train, X_test, y_test, ids_train, ids_test, interp_params, cat_encoding = res
