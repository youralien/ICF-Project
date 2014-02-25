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

	def getUniqueOrgDes(self, df=None):
		if df == None:
			df = self.entities

		if self._filteredByOrgDes == None:
			self._filteredByOrgDes = self._filterByOrgDes(df)

		return self._filteredByOrgDes

	def getUniqueFlights(self, df=None):
		if df == None:
			df = self.entities

		if self._filteredByUniqueFlights == None:
			self._filteredByUniqueFlights = self._filterUniqueFlights(df)

		return self._filteredByUniqueFlights

	def getUniqueFlightsAndBookings(self, df=None):
		if df == None:
			df = self.entities

		if self._filteredByUniqueFlightsAndBookings == None:
			self._filteredByUniqueFlightsAndBookings = self._filterUniqueFlightsAndBookings(df)

		return self._filteredByUniqueFlightsAndBookings

	def getDrillDown(self, df=None, org=None, des=None, flight=None, cabin=None, bc=None, date_range=None):
		if df == None:
			df = self.entities.copy()

		if org != None: df = df[df.ORG == org]
		if des != None: df = df[df.DES == des]
		if flight != None: df = df[df.FLT == flight]
		if cabin != None: df = df[df.BC == self.utils.mapCabinToBookingClass(cabin)]
		if bc != None: df = df[df.BC == bc]
		# NEED TO FIGURE OUT HOW TO FILTER FOR A RANGE OF DATES

		# Call other functions we've already written using this new dataframe

		return df

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

		returns: Pandas DataFrame object with bookings grouped by departure and
				 arrival locations (groups passengers by route traveled)
		"""
		return df.groupby(['ORG', 'DES'], sort=False)

	def _filterUniqueFlights(self, df):
		"""
		entities: Pandas DataFrame object containing raw data from the CSV file

		returns: Pandas DataFrame object with bookings grouped into unique 
				 flight objects (groups passengers on a per-flight basis)
		"""
		return df.groupby(['DATE', 'FLT', 'ORG', 'DES'], sort=False)

	def _filterUniqueFlightsAndBookings(self, df):
		return df.groupby(['DATE', 'FLT', 'ORG', 'DES', 'BC'], sort=False)

def main():
	pass

if __name__ == '__main__':
	main()