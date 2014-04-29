import matplotlib.pyplot as plt

import thinkstats2
import thinkplot

from FeatureFilter import FeatureFilter
from Utils import Utils

def bookingClassTicketFrequencies(data):
    print 'Grouping into unique flights'
    flight_data = f.getUniqueFlights(data)

    bcs = Utils.mapCabinToBookingClasses(cabin)
    bcs = {bc: 0 for (bc, r) in bcs}
    
    print 'Iterating through all flights'
    for flight, flight_df in flight_data:
        bc_df = f.
        keyday = -1 * flight_df['KEYDAY']
        bkd = flight_df['BKD']

        # print keyday
        print bkd
        print flight_df
        break    

def CDFofCLF(data, title, xlabel=None, ylabel=None):
	"""
	Displays a Cumulative Distribution Function (CDF) of 
	Cabin Load Factor (CLF)
	args:
		data: dataframe for aggregate or separate markets. 
		title: plot title, a string
		xlabel: plot xlabel, a string
		ylabel, plot ylabel, a string

	example:
	>>> CDFofCLF(data, 'Economy CLF for all Markets from January - March 2013')
	"""
	isNormalized = if type(data['TOTALBKD'].iget(0)) is float else False

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

def main():
    num_records = 1000
    csvfile = 'Data/BKGDAT_ZeroTOTALBKD.txt'
    cabin = 'Y'

    print 'Loading data'
    f = FeatureFilter(num_records, csvfile)

    print 'Filtering'
    data = f.getDrillDown(cabins=[cabin])

    bookingClassTicketFrequencies(data)
    


if __name__ == '__main__':
    main()
