# Kyle McConnaughay (2015) & Ryan Louie (2017)
# Data Science Project with ICF International
# Franklin W. Olin College of Engineering
# Spring 2014	

import csv
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


class Airport:
	def __init__(self, airport_code):
		pass

class Edge():
	def __init__(self):
		pass

class FeatureFilter():
	"""
		FeatureFilter is a low level class that handles data straight from the
		CSV file and groups rows according to various feature values.
	"""	

	def __init__(self, csvfile='Data/BKGDAT_Filtered.txt'):
		self.csvfile = csvfile

	def loadBookings(self, n):
		"""
		n: Number of lines to read from self.csvfile

		returns: Pandas DataFrame object with n rows of bookings
		"""

		if n == 'all':
			return pd.read_csv(self.csvfile)
		else:
			return pd.read_csv(self.csvfile, nrows=int(n))

	def filterByOrgDes(self, entities):
		"""
		entities: Pandas DataFrame object containing raw data from the CSV file

		returns: Pandas DataFrame object with bookings grouped by departure and
				 arrival locations (groups passengers by route traveled)
		"""
		return entities.groupby(['ORG', 'DES'], sort=False)

	def filterUniqueFlights(self, entities):
		"""
		entities: Pandas DataFrame object containing raw data from the CSV file

		returns: Pandas DataFrame object with bookings grouped into unique 
				 flight objects (groups passengers on a per-flight basis)
		"""
		return entities.groupby(['DATE', 'FLT', 'ORG', 'DES'], sort=False)

	def filterUniqueFlightsAndBookings(self, entities):
		return entities.groupby(['DATE', 'FLT', 'ORG', 'DES', 'BC'], sort=False)

class Network():
	"""
	"""

	def __init__(self, n):
		self.f = FeatureFilter()
		self.entities = self.f.loadBookings(n)
		self.util = Utils()

	def countEntitiesBetweenCities(self):
		"""

		"""
		flights = self.f.filterByOrgDes(self.entities)
		network = {}
		for flight_path, group in flights:
			network[flight_path] = len(group)

		return network
															
	def countFlightsBetweenCities(self):
		flights = self.f.filterUniqueFlights(self.entities)
		num_flights = {}
		for flight, group in flights:
			num_flights[flight[2:]] = num_flights.get(flight[2:], 0) + 1

		return num_flights

	def countTotalPassengersPerFlight(self):
		flights = self.f.filterUniqueFlightsAndBookings(self.entities)
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
		flights = self.f.filterUniqueFlightsAndBookings(self.entities)
		capacities = {}
		for booking_group, data in flights:
			flight = booking_group[0:4]
			bc = booking_group[4]
			cabin, rank = self.util.mapBookingClassToCabinHierarchy(bc)

			if flight not in capacities:
				capacities[flight] = {}

			capacities[flight][cabin] = data['CAP'].mean()

		return capacities
			

			

	def countMeanUtilization(self):
		pass
		

	def countOverbookedAndCabinLoadFactor(self):
		""" Determines which flights overbooking occurs; calculates the 
		percentage overbooked and the cabin load factor.

		returns: list of tuples (cabin_load_factor, percent_overbooked)
		"""
		flights = self.f.filterUniqueFlightsAndBookings(self.entities)
		
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
		# flights = self.f.filterByOrgDes(self.entities)
		# dictionary = {}
		# for flight_path, groupedByPath in flights: #group is a dataframe
		# 	groupedByDate = groupedByPath.groupby(['DATE'])
		# 	for date, groupByDate in groupedByDate:
		# 		print groupByDatek

class Visualizer():

	def __init__(self):
		pass

	def plotTimeSeries(self, time_series):
		"""
		time series is a dictionary of (dates: flight counts)
		"""
		sorted_keys = sorted(time_series.keys())
		print range(len(sorted_keys))
		print [time_series[date] for date in sorted_keys]

		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		ax.plot(range(len(sorted_keys)), [time_series[date] for date in sorted_keys])
		ax.xaxis.set_ticklabels(sorted_keys)
		for tick in ax.xaxis.get_major_ticks():
			tick.label.set_fontsize(8)
			tick.label.set_rotation('vertical')
		plt.show()

	def date2DayOfWeek(self, date):
		"""
		date : string 'm/d/yyyy' or 'mm/dd/yyyy'
		"""
		month, day, year = date.split('/')
		month, day, year = int(month), int(day), int(year)

		day = datetime.date(year, month, day) 
		return day.strftime("%A")

def summaryStatistics():
	num_records = 'all'
	n = Network(num_records)

	num_total_flights = len(n.f.filterUniqueFlights(n.entities))
	num_of_flights_between_cities = n.countFlightsBetweenCities()
	num_routes = len(num_of_flights_between_cities.keys())

	f = open('ICF Summary Statistics.txt', 'w')

	f.write("Total Number of Flights: " + str(num_total_flights) + "\n")
	f.write("Total Number of Directional Routes: " + str(num_routes) + "\n")
	f.writelines([str(citypath) + ': ' + str(num_flights) + "\n" for citypath, num_flights in num_of_flights_between_cities.items()])
	
	f.close()

def timeVsFlights():
	num_records = 'all'
	n = Network(num_records)
	x = n.timeseries()
	v = Visualizer()
	keys = x.keys()
	for i, key in enumerate(keys):
		print key, len(x[key])
	v.plotTimeSeries(x[keys[0]])

def overbookingVsCabinLoadFactor():
	num_records = 'all'
	n = Network(num_records)
	utilization = n.countOverbookedAndCabinLoadFactor()
	print utilization
	plt.plot(utilization)
	plt.xlabel('Cabin Load Factor (units?)')
	plt.ylabel('Overbooking (units?)')
	plt.show()

def testTransform():
	num_records = 'all'
	n = Network(num_records)
	flights_and_bookings = n.f.filterUniqueFlights(n.entities)

class Utils():
	def mapBookingClassToCabinHierarchy(self, bc):
		with open('Data/BC_Hierarchy.csv', 'r') as bc_file:
			reader = csv.reader(bc_file)
			for rank, booking_class, cabin in reader:
				if bc == booking_class:
					return cabin, rank

		raise Exception('Booking Class not found')

def main():
	num_records = 'all'
	n = Network(num_records)
	# utilizations = n.countMeanUtilization()
	# for flight, utilization in utilizations.items():
	# 	print flight, utilization
	# print n.entities.loc[:, 'TOTALBKD']
	print n.countCabinCapacityPerFlight()

	# x = n.countMeanUtilization()
	# for key, value in x.items():
	# 	print key, value

if __name__ == "__main__":
	num_records = '1000'
	n = Network(num_records)
	utilization = n.countOverbookedAndCabinLoadFactor()
	print utilization
	x, y = zip(*utilization)
	plt.scatter(x, y)
	plt.show()	

	# plt.plot(utilization)
	# plt.xlabel('Cabin Load Factor (units?)')
	# plt.ylabel('Overbooking (units?)')
	# plt.sho