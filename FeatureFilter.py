# Kyle McConnaughay (2015) & Ryan Louie (2017)
# Data Science Project with ICF International
# Franklin W. Olin College of Engineering
# Spring 2014	

import pandas as pd
from Utils import Utils

class FeatureFilter():
	"""
		FeatureFilter is a low level class that handles data straight from the
		CSV file and groups rows according to various feature values.
	"""	

	def __init__(self, n, csvfile='Data/BKGDAT_Filtered.txt'):
		self.csvfile = csvfile
		self.entities = self._loadBookings(n)

		self._filteredByOrgDes = None
		self._filteredByUniqueFlights = None
		self._filteredByUniqueFlightsAndBookings = None

	def getUniqueOrgDes(self, df=None):
		"""
		entities: Pandas DataFrame object containing raw data from the CSV file

		returns: Pandas GroupBy object with bookings grouped by departure and
				 arrival locations (groups passengers by route traveled)
		"""
		if not isinstance(df, pd.DataFrame):
			df = self.entities

		if self._filteredByOrgDes == None:
			self._filteredByOrgDes = self._filterByOrgDes(df)

		return self._filteredByOrgDes

	def getUniqueFlights(self, df=None):
		"""
		entities: Pandas DataFrame object containing raw data from the CSV file

		returns: Pandas GroupBy object with bookings grouped into unique 
				 flight objects (groups passengers on a per-flight basis)
		"""
		if not isinstance(df, pd.DataFrame):
			df = self.entities

		if self._filteredByUniqueFlights == None:
			self._filteredByUniqueFlights = self._filterUniqueFlights(df)

		return self._filteredByUniqueFlights

	def getUniqueFlightsAndBookings(self, df=None):
		"""
		entities: Pandas DataFrame object containing raw data from the CSV file

		returns: Pandas GroupBy object with booking grouped into unique flight
				 objects (groups passengers by their flight and booking class)
		"""
		if not isinstance(df, pd.DataFrame):
			df = self.entities

		if self._filteredByUniqueFlightsAndBookings == None:
			self._filteredByUniqueFlightsAndBookings = self._filterUniqueFlightsAndBookings(df)

		return self._filteredByUniqueFlightsAndBookings

	def getDrillDown(self, df=None, orgs=None, dests=None, flights=None, cabins=None, bcs=None, date_ranges=None):
		"""
		Filters a dataframe for rows that match the given features. 
		For instance, orgs=['DXB', 'DMM'], bcs=['B'] will return the dataframe
		that contains rows with flights that departed from DXB or DMM AND that
		were made in the 'B' booking class

		args:
			df: a pd.DataFrame object that the user wants to filter
			orgs: list of strings describing origin airports
			dests: list of strings describing destination airports
			flights: list of ints describing flight numbers
			cabins: list of strings describing cabins ('Y' or 'J')
			bcs: list of strings describing booking classes
			data_ranges: list of tuples of strings describing desired dates

		returns:
			pd.DataFrame instance
		"""

		if not isinstance(df, pd.DataFrame):
			df = self.entities.copy()

		false_mask = pd.Series(False, list(df.index))
		m = pd.Series(True, list(df.index))

		m = m & self._mask(false_mask, orgs, df.ORG)
		m = m & self._mask(false_mask, dests, df.DES)
		m = m & self._mask(false_mask, flights, df.FLT)
		m = m & self._mask(false_mask, bcs, df.BC)
		m = m & self._mask(false_mask, date_ranges, df.DATE)
		if cabins != None:
			cs = [Utils.mapCabinToBookingClass(cabin)[0] for cabin in cabins]
			m = m & self._mask(m, cs, df.BC)

		return df[m]

	def _mask(self, m, vals, column):
		"""
		Helper function for getDrillDown
		"""
		if vals != None:
			for val in vals:
				m = m | (column == val)

			return m
		else:
			return pd.Series(True, list(m.index))

	def _loadBookings(self, n):
		"""
		n: Number of lines to read from self.csvfile

		returns: Pandas DataFrame object with n rows of bookings
		"""

		if n == 'all':
			return pd.read_csv(self.csvfile)
		else:
			return pd.read_csv(self.csvfile, nrows=int(n))

	def _filterByOrgDes(self, df):
		"""
		entities: Pandas DataFrame object containing raw data from the CSV file

		returns: Pandas GroupBy object with bookings grouped by departure and
				 arrival locations (groups passengers by route traveled)
		"""
		return df.groupby(['ORG', 'DES'], sort=False)

	def _filterUniqueFlights(self, df):
		"""
		entities: Pandas DataFrame object containing raw data from the CSV file

		returns: Pandas GroupBy object with bookings grouped into unique 
				 flight objects (groups passengers on a per-flight basis)
		"""
		return df.groupby(['DATE', 'FLT', 'ORG', 'DES'], sort=False)

	def _filterUniqueFlightsAndBookings(self, df):
		"""
		entities: Pandas DataFrame object containing raw data from the CSV file

		returns: Pandas GroupBy object with booking grouped into unique flight
				 objects (groups passengers by their flight and booking class)
		"""
		return df.groupby(['DATE', 'FLT', 'ORG', 'DES', 'BC'], sort=False)

def main():
	pass

if __name__ == '__main__':
	main()