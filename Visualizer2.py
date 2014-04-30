import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import thinkstats2
import thinkplot

from Regression2 import encodeFlights, aggregateTrainTestSplit
from FeatureFilter import FeatureFilter
from Utils import Utils

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
    print "Grouping by flight"
    flights = f.getUniqueFlights(data)

    print "Encoding flight data"
    start = -300
    stop = 0
    num_points = 31
    interp_params = (start, stop, num_points)

    bin_size = 1
    date_reduction = -1
    cat_encoding = (bin_size, date_reduction)

    X, y, ids = encodeFlights(flights, interp_params, cat_encoding)
    
    X, _, y, _, _, _ = aggregateTrainTestSplit(X, y, ids, 1.0)
    print X.shape

    cats_end = 32
    nums_start = cats_end

    features = {
        "keyday":32,
        "bkd":33,
        "auth":34,
        "avail":35,
        "cap":36, 
        "clf":37,}
        # "bkd_lower":38}

    for i, (name, col) in enumerate(features.items()):
        thinkplot.SubPlot(2,3,i+1)
        thinkplot.Scatter(X[:,col], y)
        thinkplot.Config(xlabel='{}'.format(name), ylabel='delta bkd')

    thinkplot.Show()

def main_from_csv():
    num_records = 'all'
    csvfile = 'Data/BKGDAT_ZeroTOTALBKD.txt'
    cabin = 'Y'

    print 'Loading data from CSV'
    f = FeatureFilter(num_records, csvfile)

    print 'Filtering'
    data = f.getDrillDown(cabins=[cabin])

    print 'Pickling to file'
    with file('Serialized/feature_filter.pkl', 'w') as file_obj:
        pickle.dump(f,file_obj,pickle.HIGHEST_PROTOCOL)
    data.to_pickle('Serialized/bkgdat_economy.pkl')


def main_from_pkl():
    print 'Loading data from PKL'
    data = pd.load('Serialized/bkgdat_economy.pkl')
    

if __name__ == '__main__':
    main_from_csv()
