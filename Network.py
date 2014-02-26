# Kyle McConnaughay (2015) & Ryan Louie (2017)
# Data Science Project with ICF International
# Franklin W. Olin College of Engineering
# Spring 2014	

from FeatureFilter import FeatureFilter
from Utils import Utils

import pandas as pd

class Network():
	"""
	"""

	def __init__(self, nrows):
		self.f = FeatureFilter(nrows)

	def countEntitiesBetweenCities(self):
		"""

		"""
		flights = self.f.getFilterByOrgDes()
		network = {}
		for flight_path, group in flights:
			network[flight_path] = len(group)

		return network
															
	def countFlightsBetweenCities(self):
		flights = self.f.getFilterUniqueFlights()
		num_flights = {}
		for flight, group in flights:
			num_flights[flight[2:]] = num_flights.get(flight[2:], 0) + 1

		return num_flights

	def countTotalPassengersPerFlight(self):
		flights = self.f.getFilterUniqueFlightsAndBookings()
		total_booked = {}
		total_seats = {}
		utilization = {}

		for booking_group, data in flights:

			flight = booking_group[0:4]
			total_booked[flight] = total_booked.get(flight, 0) + data['TOTALBKD'].mean()
			total_seats[flight] = total_seats.get(flight, 0) + data['CAP'].mean()

		for booking_group, data in flights:
			flight = booking_group[0:4]
			utilization[flight] = float(total_booked[flight]) / total_seats[flight]
		
		return utilization

	def countCabinCapacityPerFlight(self):
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

	def countFinalCabinLoadFactor(self): # total booked / capacity per flight
		capacities = self.countCabinCapacityPerFlight()
		total_bookings = self.countTotalBookedPerFlight()
		cabin_load_factors = {}

		for flight in capacities.keys():
			total_cap = sum(capacities[flight].values())
			total_booked = sum(total_bookings[flight].values())
			cabin_load_factors[flight] = total_booked / total_cap

		return cabin_load_factors
		

	def countOverbookedAndCabinLoadFactor(self):
		""" Determines which flights  overbooking occurs; calculates the 
		percentage overbooked and the cabin load factor.

		returns: list of tuples (cabin_load_factor, percent_overbooked)
		"""
		flights = self.f.getFilterUniqueFlightsAndBookings()
		
		ans = []

		for booking_group, data in flights:
			
			AUTH = data['AUTH'].mean()
			CAP = data['CAP'].mean()
			if AUTH > CAP: # Overbooking occurs when AUTH > CAP

				flight = booking_group[:4]
				percent_overbooked = float(AUTH)/CAP
				cabin_load_factor = float(data['TOTALBKD'].mean())/CAP
				ans.append((cabin_load_factor, percent_overbooked))

		return ans	

	def timeseries(self):
		flights = self.f.filterUniqueFlights(self.entities)
		time_series = {}
		for flight, group in flights:
			location = flight[2:]
			time_series[location] = time_series.get(location, {})
			time_series[location][flight[0]] = time_series[location].get(flight[0], 0) + 1

		return time_series

def main():
	pass

if __name__ == '__main__':
	main()