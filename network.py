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

class Network():
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
		"""
		return entities.groupby(['ORG', 'DES'], sort=False)

	def entitiesBetweenCities(self, entities):

		flights = self.filterByOrgDes(entities)
		network = {}
		for flight_path, group in flights:
			network[flight_path] = len(group)

		return network

	def filterUniqueFlights(self, entities):
		return entities.groupby(['DATE', 'FLT', 'ORG', 'DES'], sort=False)
															
	def countFlightsBetweenCities(self, entities):
		
		flights = self.filterUniqueFlights(entities)
		num_flights = {}
		for flight, group in flights:
			num_flights[flight[2:]] = num_flights.get(flight[2:], 0) + 1

		return num_flights

def main():
	num_records = 'all'
	n = Network()
	entities = n.loadBookings(num_records)
	print n.countFlightsBetweenCities(entities)

if __name__ == "__main__":
	main()



