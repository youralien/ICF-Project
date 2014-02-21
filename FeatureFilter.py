import pandas as pd

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

	def getUniqueOrgDes(self):
		if self._filteredByOrgDes == None:
			self._filteredByOrgDes = self._filterByOrgDes()

		return self._filteredByOrgDes

	def getUniqueFlights(self):
		if self._filteredByUniqueFlights == None:
			self._filteredByUniqueFlights = self._filterUniqueFlights()

		return self._filteredByUniqueFlights

	def getUniqueFlightsAndBookings(self):
		if self._filteredByUniqueFlightsAndBookings == None:
			self._filteredByUniqueFlightsAndBookings = self._filterUniqueFlightsAndBookings()

		return self._filteredByUniqueFlightsAndBookings

	def _loadBookings(self, n):
		"""
		n: Number of lines to read from self.csvfile

		returns: Pandas DataFrame object with n rows of bookings
		"""

		if n == 'all':
			return pd.read_csv(self.csvfile)
		else:
			return pd.read_csv(self.csvfile, nrows=int(n))

	def _filterByOrgDes(self):
		"""
		entities: Pandas DataFrame object containing raw data from the CSV file

		returns: Pandas DataFrame object with bookings grouped by departure and
				 arrival locations (groups passengers by route traveled)
		"""
		return self.entities.groupby(['ORG', 'DES'], sort=False)

	def _filterUniqueFlights(self):
		"""
		entities: Pandas DataFrame object containing raw data from the CSV file

		returns: Pandas DataFrame object with bookings grouped into unique 
				 flight objects (groups passengers on a per-flight basis)
		"""
		return self.entities.groupby(['DATE', 'FLT', 'ORG', 'DES'], sort=False)

	def _filterUniqueFlightsAndBookings(self):
		return self.entities.groupby(['DATE', 'FLT', 'ORG', 'DES', 'BC'], sort=False)

def main():
	pass

if __name__ == '__main__':
	main()