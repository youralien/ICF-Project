
# Kyle McConnaughay (2015) & Ryan Louie (2017)
# Data Science Project with ICF International
# Franklin W. Olin College of Engineering
# Spring 2014   

import csv
import datetime

class Utils():
    """
    Utility functions used by FeatureFilter, Network, and Visualizer
    """
    days_of_week = ["Sunday","Monday","Tuesday","Wednesday",
                    "Thursday","Friday","Saturday"]
                    
    bc_hierarchy = [(1,'Y','Y'), 
                    (2,'H','Y'), 
                    (3,'M','Y'),
                    (4,'L','Y'),
                    (5,'B','Y'),
                    (6,'K','Y'),
                    (7,'X','Y'),
                    (8,'Q','Y'),
                    (9,'V','Y'),
                    (10,'E','Y'),
                    (11,'S','Y'),
                    (12,'N','Y'),
                    (13,'O','Y'),
                    (14,'T','Y'),
                    (15,'U','Y'),
                    (16,'Z','Y'),
                    (17,'G','Y'),
                    (18,'W','Y'),
                    (1,'J','J'),
                    (2,'C','J'),
                    (3,'D','J'),
                    (4,'I','J'),
                    (5,'P','J'),
                    (6,'R','J')]

    @staticmethod
    def compareBCs(bc1, bc2):
        c1, r1 = Utils.mapBookingClassToCabinHierarchy(bc1)
        c2, r2 = Utils.mapBookingClassToCabinHierarchy(bc2)

        return r2 - r1


    @staticmethod
    def mapBookingClassToCabinHierarchy(booking_class):
        """
        ONLY FOR ECONOMY CLASSES
        args:
            booking_class: string of the booking class that is being looked up

        returns:
            tuple of (cabin letter, rank in the booking class hierarchy)
        """
        for r, bc, c in Utils.bc_hierarchy:
            if booking_class == bc:
                return c, r

        raise Exception('Booking Class not found')
    
    @staticmethod
    def mapCabinToBookingClasses(cabin):
        """
        args:
            cabin: string of the cabin that is being looked up. Only takes two
                   values; it can be either 'Y' or 'J' for economy or business 
                   class, respectively

        returns:
            list of tuples of (booking class, rank) associated with the given 
            cabin
        """
        with open('Data/BC_Hierarchy.csv', 'r') as bc_file:
            reader = csv.reader(bc_file)
            return [(bc, r) for r, bc, c in reader if c == cabin]

    @staticmethod
    def mapRankToBookingClass(rank, cabin):
        """
        args:
            rank: integer of rank of booking class that is being looked up

        returns:
            tuple of (cabin, booking class) associated with the given rank
        """
        for r, bc, c in Utils.bc_hierarchy:
            if rank == r and cabin == c:
                return bc

    @staticmethod
    def date2DayOfWeek(date):
        """
        args:
            date: string 'm/d/yyyy' or 'mm/dd/yyyy'

        returns: 
            string giving the day of the week that the date fell on
        """
        month, day, year = date.split('/')
        month, day, year = int(month), int(day), int(year)

        day = datetime.date(year, month, day) 
        return day.strftime("%A")

    @staticmethod
    def writeSeriesToFile(f, series, indent=''):
        for i in range(series.size):
            index = series.index[i]
            line = indent + str(index) + ": " + str(series[index]) + '\n'
            f.write(line)

    @staticmethod
    def createTitleForFeatures(orgs,dests,flights,cabins,bcs,date_ranges):
        """ Creates Plot Title that includes the user's getDrillDown args
        """
        # Handle features that were not specified for drill down 
        orgs = 'all' if orgs==None else str(orgs).strip("[]")
        dests = 'all' if dests==None else str(dests).strip("[]")
        flights = 'all' if flights==None else str(flights).strip("[]")
        cabins = 'all' if cabins==None else str(cabins).strip("[]")
        bcs = 'all' if bcs==None else str(bcs).strip("[]")
        date_ranges = 'all' if date_ranges==None else str(date_ranges).strip("[]")

        title = "Origin: {} | Destination: {}\n Flight: {} | Cabin: {}\n Booking Class: {} | Date: {}" 
        
        # Format title string with specific drillDown args
        return title.format(orgs,dests,flights,cabins,bcs,date_ranges)

    @staticmethod
    def isOverbooked(bookings):
        """ is any value for bookings overbooked? At some point is AUTH > CAP?
        
        bookings: np array
        """

        overbooked = False

        for e in bookings:
            if e > 1:
                overbooked = True

        return overbooked

    @staticmethod
    def sortByIndex(index, *args):
        if args == ():
            return index

        consolidated = (index,) + args
        zip_args = zip(*consolidated)
        sorted_args = sorted(zip_args, key=lambda tup: tup[0])
        unzipped_args = zip(*sorted_args)
        return unzipped_args


def main():
    cs = []
    classes = Utils.mapCabinToBookingClasses('Y')
    cs.extend([bc for bc, rank in classes])

    print len(cs)

    classes = Utils.mapCabinToBookingClasses('J')
    cs.extend([bc for bc, rank in classes])
    print len(cs)
    print cs
    
if __name__ == '__main__':
    main()