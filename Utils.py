# Kyle McConnaughay (2015) & Ryan Louie (2017)
# Data Science Project with ICF International
# Franklin W. Olin College of Engineering
# Spring 2014	

import csv

class Utils():
	"""
	Utility functions used by FeatureFilter, Network, and Visualizer
	"""

	@staticmethod
	def mapBookingClassToCabinHierarchy(bc):
		"""
		args:
			bc: string of the booking class that is being looked up

		returns:
			tuple of (cabin letter, rank in the booking class hierarchy)
		"""
		with open('Data/BC_Hierarchy.csv', 'r') as bc_file:
			reader = csv.reader(bc_file)
			for rank, booking_class, cabin in reader:
				if bc == booking_class:
					return cabin, rank

		raise Exception('Booking Class not found')

	@staticmethod
	def mapCabinToBookingClass(cabin):
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

def main():
	pass
	
if __name__ == '__main__':
	main()