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

def CDFofCLF(norm_data, title, xlabel, ylabel):
	"""
	Displays a Cumulative Distribution Function (CDF) of 
	Cabin Load Factor (CLF)
	args:
		norm_data: normalized dataframe for aggregate or separate markets. 
		title: plot title, a string
		xlabel: plot xlabel, a string
		ylabel, plot ylabel, a string
	"""
	unique_flights = f.getUniqueFlights(norm_data)

	CLFs = [d.iget(0) for g, d in unique_flights.TOTALBKD]

	cdf_of_clf = thinkstats2.MakeCdfFromList(CLFs)
	
	thinkplot.Cdf(cdf_of_clf)
	thinkplot.Show(	title='Economy CLF for all Markets from January - March 2013',
               		xlabel='Cabin Load Factor',
                   	ylabel='CDF')

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
