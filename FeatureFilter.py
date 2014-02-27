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
		if not isinstance(df, pd.DataFrame):
			df = self.entities

		if self._filteredByOrgDes == None:
			self._filteredByOrgDes = self._filterByOrgDes(df)

		return self._filteredByOrgDes

	def getUniqueFlights(self, df=None):
		if not isinstance(df, pd.DataFrame):
			df = self.entities

		if self._filteredByUniqueFlights == None:
			self._filteredByUniqueFlights = self._filterUniqueFlights(df)

		return self._filteredByUniqueFlights

	def getUniqueFlightsAndBookings(self, df=None):
		if not isinstance(df, pd.DataFrame):
			df = self.entities

		if self._filteredByUniqueFlightsAndBookings == None:
			self._filteredByUniqueFlightsAndBookings = self._filterUniqueFlightsAndBookings(df)

		return self._filteredByUniqueFlightsAndBookings

	def getDrillDown(self, df=None, orgs=None, dests=None, flights=None, cabins=None, bcs=None, date_ranges=None):
		if not isinstance(df, pd.DataFrame):
			df = self.entities.copy()

		m = pd.Series(True, list(df.index))

		m = self._mask(m, orgs, df.ORG)
		m = self._mask(m, dests, df.DES)
		m = self._mask(m, flights, df.FLT)
		m = self._mask(m, bcs, df.BC)
		m = self._mask(m, date_ranges, df.DATE)
		if cabins != None:
			m = self._mask(m, [Utils.mapCabinToBookingClass(cabin) for cabin in cabins], df.BC)

		return df[m]

		# if orgs != None: 
		# 	for org in orgs
		# if des != None: df = df[df.DES == des]
		# if flight != None: df = df[df.FLT == flight]
		# if bcs != None:

		# 	for bc in bcs:
		# 		| df.BC == bc
		# 	df = df[df.BC == bc]
		# if cabin != None: df = df[df.BC == self.utils.mapCabinToBookingClass(cabin)]


		return df

	def _mask(self, m, vals, column):
		"""
		Helper function for getDrillDown
		"""
		if vals != None:
			for val in vals:
				print column == val
				print m & (column == val)
				m = m & (column == val)

		print m

		return m


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