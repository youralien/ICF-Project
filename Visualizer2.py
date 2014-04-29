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