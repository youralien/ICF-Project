# Kyle McConnaughay (2015) & Ryan Louie (2017)
# Data Science Project with ICF International
# Franklin W. Olin College of Engineering
# Spring 2014	

from FeatureFilter import FeatureFilter
from Utils import Utils

import numpy as np
import pandas as pd

class Network():
	"""
	Network consumes data frames from FeatureFilter and calculates interesting
	statistics about the flight network
	"""

	def __init__(self, nrows, csvfile='Data/BKGDAT_Filtered.txt'):
		self.f = FeatureFilter(nrows, csvfile)
															
	def countFlightsBetweenCities(self):
		"""
		Counts the total number of flights between unique org-des pairs. 
		Similar to timeseries but it doesn't index the counts by date.

		returns:
			dictionary of {(org, des), number of flights from org to des}
		"""
		flights = self.f.getFilterUniqueFlights()
		num_flights = {}
		for flight, group in flights:
			num_flights[flight[2:]] = num_flights.get(flight[2:], 0) + 1

		return num_flights

	def countCabinCapacityPerFlight(self):
		"""
		Counts the total capcity of a flight in every cabin on the plane

		returns:
			dictionary of {flight, dictionary of {cabin, cabin capacity}}
		"""
		flights = self.f.getUniqueFlightsAndBookings()
		capacities = {}
		for booking_group, data in flights:
			flight = booking_group[0:4]
			bc = booking_group[4]
			cabin, rank = Utils.mapBookingClassToCabinHierarchy(bc)

			if flight not in capacities:
				capacities[flight] = {}

			capacities[flight][cabin] = data['CAP'].mean()

		return capacities

	def countTotalBookedPerFlight(self):
		"""
		Counts the total number of passengers on a flight in every cabin on the
		plane

		returns:
			dictionary of {flight, dictionary of {cabin, total booked}}
		"""
		flights = self.f.getUniqueFlightsAndBookings()
		total_bookings = {}
		for booking_group, data in flights:
			flight = booking_group[0:4]
			bc = booking_group[4]
			cabin, rank = Utils.mapBookingClassToCabinHierarchy(bc)

			if flight not in total_bookings:
				total_bookings[flight] = {}

			total_bookings[flight][cabin] = data['TOTALBKD'].mean()

		return total_bookings	

	def countFinalCabinLoadFactor(self):
		"""
		Computes what percentage of each flight in self.entities is filled at
		the time of departure (i.e. TOTALBKD / CAP)

		returns:
			dictionary of {flight, cabin load factor}
		"""
		capacities = self.countCabinCapacityPerFlight()
		total_bookings = self.countTotalBookedPerFlight()
		cabin_load_factors = {}

		for flight in capacities.keys():
			total_cap = sum(capacities[flight].values())
			total_booked = sum(total_bookings[flight].values())
			cabin_load_factors[flight] = total_booked / total_cap

		return cabin_load_factors
		

	def countOverbookedAndCabinLoadFactor(self):
		""" 
		Determines which flights  overbooking occurs; calculates the 
		percentage overbooked and the cabin load factor.

		returns: 
			list of tuples {cabin_load_factor, percent_overbooked}
		"""
		flights = self.f.getUniqueFlightsAndBookings()
		
		ans = []

		for booking_group, data in flights:
			
			AUTH = data['AUTH'].mean()
			CAP = data['CAP'].mean()
			if AUTH > CAP: # Overbooking occurs when AUTH > CAP

				flight = booking_group[:4]
				percent_overbooked = float(AUTH)/CAP
				cabin_load_factor = float(data['TOTALBKD'].mean()) / CAP
				ans.append((cabin_load_factor, percent_overbooked))

		return ans

	def interp(self, xvals, x, y):
		x, y = zip(*sorted(zip(x, y), key=lambda tup: tup[0]))
		return np.interp(xvals, x, y, left=0)
		
	def timeseries(self):
		"""
		Counts the number of flights that occur along a directed edge (unique
		org-des pairs) in self.entities and indexes the counts by their date
		
		returns:
			dictionary of {time, dictionary of {directed_edge, count}}
		"""
		flights = self.f.filterUniqueFlights(self.entities)
		time_series = {}
		for f, group in flights:
			local = f[2:]
			time_series[local] = time_series.get(local, {})
			time_series[local][f[0]] = time_series[local].get(f[0], 0) + 1

		return time_series

def main():
	pass

if __name__ == '__main__':
	main()