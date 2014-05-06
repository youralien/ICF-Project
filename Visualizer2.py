import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

import thinkstats2
import thinkplot

from Regression2 import encodeFlights, aggregate
from FeatureFilter import FeatureFilter
from Utils import Utils
from AirportCodes import AirportCodes
from regression3 import mainRyan, regress, Error

def bookingClassTicketFrequencies(f, data, cabin):
    print 'Grouping into unique flight/booking class combinations'
    flight_data = f.getUniqueFlightsAndBookings(data)

    bcs = Utils.mapCabinToBookingClasses(cabin)
    bcs = {bc: 0 for (bc, r) in bcs}
    
    print 'Iterating through all booking classes'
    for flight, flight_df in flight_data:
        bc = flight[-1]
        keyday = -1 * flight_df['KEYDAY']
        bkd = flight_df['BKD']

        keyday, bkd = Utils.sortByIndex(keyday, bkd)

        bcs[bc] += bkd[-1]

    total_bkd = 0.0
    for bc, num_bkd in bcs.items():
        total_bkd += num_bkd

    for bc in bcs:
        bcs[bc] /= total_bkd

    ks, vs = zip(*bcs.items())
    ks, vs = zip(*sorted(zip(ks, vs), key=lambda tup: Utils.compareBCs(tup[0])))
    indices = np.arange(len(ks))
    width = 0.75

    fig, ax = plt.subplots()
    rects = ax.bar(indices, vs, width)
    ax.set_ylabel('Percent of Total Booked')
    ax.set_title('Booking Class Ticketing Distribution - Economy Cabin')
    ax.set_xticks(indices + width/2.0)
    ax.set_xticklabels(ks)

    plt.grid()
    plt.show()


def CDFofCLF(f, data, title, xlabel=None, ylabel=None):
	"""
	Displays a Cumulative Distribution Function (CDF) of 
	Cabin Load Factor (CLF)
	args:
        f: FeatureFilter object
		data: dataframe for aggregate or separate markets. 
		title: plot title, a string
		xlabel: plot xlabel, a string
		ylabel, plot ylabel, a string

	example:
	>>> CDFofCLF(data, 'Economy CLF for all Markets from January - March 2013')
	"""
	isNormalized = True if type(data['TOTALBKD'].iget(0)) is float else False

	flights = f.getUniqueFlights(data)

	if isNormalized:
		CLFs = [d.iget(0) for g, d in flights.TOTALBKD]
	else:
		CLFs = [float(totalbkd.iget(0))/cap.iget(0) for (g, totalbkd),(g,cap) in zip(flights.TOTALBKD, flights.CAP)]

	cdf_of_clf = thinkstats2.MakeCdfFromList(CLFs)
	
	thinkplot.Cdf(cdf_of_clf)
	thinkplot.Show(	title=title,
               		xlabel='Cabin Load Factor' if xlabel is None else xlabel,
                   	ylabel='CDF' if ylabel is None else ylabel)

def inputsVsDeltaBKD(f, data):
    """ Makes scatter plots of quantitative input features versus deltaBKD 
    args:
        f: FeatureFilter object
        data: dataframe for aggregate or separate markets
    """
    isNormalized = True if type(data['TOTALBKD'].iget(0)) is float else False

    print 'Grouping by flight'
    flights = f.getUniqueFlights(data)

    print 'Encoding flight data'
    start = -90
    stop = 0
    num_points = 31
    interp_params = (start, stop, num_points)

    bin_size = 1
    date_reduction = -1
    cat_encoding = (bin_size, date_reduction)

    X, y, ids = encodeFlights(flights, interp_params, cat_encoding)
    
    print 'Aggregating Flights'
    X, y, ids = aggregate(X, y, ids)

    cats_end = 32
    nums_start = cats_end

    if isNormalized:
        features = {
            'keyday':32,
            'bkd normalized':33,
            'auth normalized':34,
            'avail normalized':35,
            'cap':36, 
            'bkd_lower normalized':37
            }
        targetlabel='delta bkd normalized'
    else:
        features = {
            'keyday':32,
            'bkd':33,
            'auth':34,
            'avail':35,
            'cap':36, 
            'bkd_lower':37
            }
        targetlabel='delta bkd'

    for i, (name, col) in enumerate(features.items()):
        thinkplot.SubPlot(2,3,i+1)
        thinkplot.Scatter(X[:,col],y)
        thinkplot.Config(xlabel='{}'.format(name), ylabel=targetlabel)

    thinkplot.Show()

def inputsVsBKD(f, data):
    """ Makes scatter plots of quantitative input features versus deltaBKD 
    args:
        f: FeatureFilter object
        data: dataframe for aggregate or separate markets
    """
    isNormalized = True if type(data['TOTALBKD'].iget(0)) is float else False

    print 'Grouping by flight'
    flights = f.getUniqueFlights(data)

    print 'Encoding flight data'
    start = -90
    stop = 0
    num_points = 31
    interp_params = (start, stop, num_points)

    bin_size = 1
    date_reduction = -1
    cat_encoding = (bin_size, date_reduction)

    X, y, ids = encodeFlights(flights, interp_params, cat_encoding)
    
    print 'Aggregating Flights'
    X, y, ids = aggregate(X, y, ids)

    cats_end = 32
    nums_start = cats_end

    if isNormalized:
        features = {
            'keyday':32,
            'bkd normalized':33,
            'auth normalized':34,
            'avail normalized':35,
            'cap':36
            }
        targetlabel='delta bkd normalized'
    else:
        features = {
            'keyday':32,
            'bkd':33,
            'auth':34,
            'avail':35,
            'cap':36
            }
        targetlabel='delta bkd'

    for i, (name, col) in enumerate(features.items()):
        thinkplot.SubPlot(2,3,i+1)
        if name == 'bkd':
            thinkplot.Scatter(X[:-1, col], X[1:, col])
            thinkplot.Config(xlabel='{}'.format(name), ylabel='bkd autocorrelation')
        else:
            thinkplot.Scatter(X[:,col],X[:, features['bkd']])
            thinkplot.Config(xlabel='{}'.format(name), ylabel=targetlabel)


def ScatterFeaturesTargets(f, market):
    """
    NOTE: Frankfurt is less Dense than say, Bahrain. Thus Scatter is going to 
    look less dense and different

    Bahrain: 1226 flights
    Frankfurt: 200 flights
    Delhi: 358 flights
    London: ???
    Bangkok: ???
    Aggregate: 10842 flights
    """ 
    print 'Filtering'
    if market is None:
        data = f.getDrillDown(cabins=cabins)
    else:
        orgs = [AirportCodes.Dubai, market]
        dests = [AirportCodes.Dubai, market]
        data = f.getDrillDown(orgs=orgs, dests=dests, cabins=cabins) 

    inputsVsDeltaBKD(f, data)

def CDFofBkdAtEachKeyDayError(bkd_error_cumsum, interp_params):
    """
    args:
        bkd_error_cumsum: a m (num ids) x n (num keydays) matrix of bkd error at
                          each keyday
        interp_params: (start, stop, num_points)
    """
    start, stop, n_keydays = interp_params

    keydays = np.linspace(start, stop, n_keydays)

    cdfs = [thinkstats2.MakeCdfFromList(bkd_error_cumsum[:,j] for j in range(n_keydays))]

    for i in range(n_keydays):
        thinkplot.Cdf(cdfs[i])
        thinkplot.Save(title='Error CDF of bkd at keyday {}'.format(keydays[i]),
                       xlabel='bkd error (bkd predicted - bkd actual)',
                       ylabel='CDF')

def CDFofErrorAtEachKeyDay(errors, interp_params):
    """
    args:
        errors: error of a particular target value at each keyday
        interp_params: (start, stop, num_points) which is fed into np linspace
    """
    start, stop, n_keydays = interp_params

    keydays = np.linspace(start, stop, n_keydays)

    cdfs = [thinkstats2.MakeCdfFromList(errors[:,j]) for j in range(n_keydays)]
    percentiles = [(cdf.Percentile(2.5), cdf.Percentile(50), cdf.Percentile(97.5)) for cdf in cdfs]
    fifth, fiftieth, ninetyfifth = zip(*percentiles)

    fig, ax = plt.subplots(1)
    ax.plot(keydays, fiftieth, label='50th percentile')
    ax.fill_between(keydays, fifth, ninetyfifth, facecolor='gray', alpha=0.5, label='Inner 90% error')

    ax.set_xlabel('keydays')
    ax.set_ylabel('bkd error (bkd predicted - bkd actual)')
    ax.legend(loc='lower left')


    fig.show()

def TotalBKDErrorComparison(errors, interp_params):
    """
    args:
        errors: error of a particular target value at each keyday
        interp_params: (start, stop, num_points) which is fed into np linspace
    """
    start, stop, n_keydays = interp_params

    keydays = np.linspace(start, stop, n_keydays)

    totalbkd_cdf = thinkstats2.MakeCdfFromList(errors[:,-1])

    return totalbkd_cdf

def bc_bars_base(y_cumsum):
    """
    args:
        y_cumsum: a sumcum deltabkd vector (either predict or actual)
    """
    totalbkd_vector = y_cumsum[:,-1]
    assert len(ids_test) == len(totalbkd_vector)
    
    bcs = {bc: 0 for (bc, r) in Utils.mapCabinToBookingClasses('Y')}
    
    for totalbkd, ids in zip(totalbkd_vector, ids_test):
        bc = ids[-1]
        bcs[bc] += totalbkd

    denom = sum(totalbkd_vector)
    
    for bc in bcs:
        bcs[bc] /= denom

    ks, vs = zip(*bcs.items())
    ks, vs = zip(*sorted(zip(ks, vs), key=lambda tup: Utils.compareBCs(tup[0])))
    indices = np.arange(len(ks))

    return ks, vs, indices

def bc_bars(y_pred_cumsum, y_real_cumsum):
    """
    args:
        y_pred_cumsum: a sumcum deltabkd vector predict 
        y_real_cumsum: a sumcum deltabkd vector actual
    """
    ks, vs_pred, indices = bc_bars_base(y_pred_cumsum)
    ks, vs_real, indices = bc_bars_base(y_real_cumsum)

    width = 0.35

    fig, ax = plt.subplots()
    rects_pred = ax.bar(indices, vs_pred, width, color='y')
    rects_real = ax.bar(indices+width, vs_real, width, color='g')
        
    ax.set_ylabel('Percent of Total Booked')
    ax.set_title('Booking Class Ticketing Distribution - Economy Cabin')
    ax.set_xticks(indices + width/2.0)
    ax.set_xticklabels(ks)
    ax.legend( (rects_pred[0], rects_real[0]), ('Predicted', 'Actual'))

    plt.grid()
    plt.show()

if __name__ == '__main__':
    normalized = False
    bkgdat = 'Data/BKGDAT_ZeroTOTALBKD.txt'
    bkgdat_norm = 'Data/NormExceptKeyday_BKGDAT_ZeroTOTALBKD.txt'
    num_records = 'all'
    csvfile =  bkgdat_norm if normalized else bkgdat
    cabins = ['Y']

    print 'Loading data from CSV'
    f = FeatureFilter(num_records, csvfile)
    res = mainRyan()
    X_train, y_train, X_test, y_test, ids_train, ids_test, interp_params, cat_encoding = res
    


    learners = [KNeighborsRegressor(n_neighbors=8), RandomForestRegressor(n_estimators=20, n_jobs=-1), Ridge(), GradientBoostingRegressor(n_estimators=100)]
    labels = ['K-nearest neighbors', 'random forest', 'ridge', 'gradient boosting']
    cdfs = []
    for learner, label in zip(learners, labels):

        # bc_bars
        y_test, y_pred = regress(learner, res)
        y_pred_cumsum, y_real_cumsum = Error.bkd_at_each_keyday(y_test, y_pred, ids_test, interp_params)

        bc_bars(y_pred_cumsum, y_real_cumsum)

        
        # Total Bkd Comparison

    #     cdf = TotalBKDErrorComparison(y_pred_cumsum - y_real_cumsum, interp_params)
    #     cdfs.append(cdf)
    #     thinkplot.Cdf(cdf, label=label)
    #     thinkplot.Config(xlabel='total bkd error', ylabel='CDF')
    # thinkplot.Show()
    # CDFofErrorAtEachKeyDay(y_pred_cumsum - y_real_cumsum, interp_params)
    # TotalBKDErrorComparison(y_pred_cumsum - y_real_cumsum)
