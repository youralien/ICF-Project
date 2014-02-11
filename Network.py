# Kyle McConnaughay (2015) & Ryan Louie (2017)
# Data Science Project with ICF International
# Franklin W. Olin College of Engineering
# Spring 2014	

import pandas as pd 
import numpy as np 


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

	def __init__(self, csvfile='Data/BKGDAT.txt'):
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

class Network():
	"""
	"""

	def __init__(self, n):
		self.f = FeatureFilter()
		self.entities = self.f.loadBookings(n)

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

	def countMeanUtilization(self):
		flights = self.f.filterUniqueFlights(self.entities)
		utilization = {}
		for flight, group in flights:
			total_booked = group['TOTALBKD'].mean()
			capacity = group['CAP'].mean()
			utilization[flight] = float(total_booked) / capacity
		
		return utilization

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
		# 		print groupByDate



def main():
	num_records = 'all'
	n = Network(num_records)
	x = n.timeseries()
	# x = n.countMeanUtilization()
	# for key, value in x.items():
	# 	print key, value


if __name__ == "__main__":
	num_records = 'all'
	n = Network(num_records)
	x = n.timeseries()



