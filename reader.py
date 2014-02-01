# Kyle McConnaughay (2015) & Ryan Louie (2017)
# Data Science Project with ICF International
# Franklin W. Olin College of Engineering
# Spring 2014

import csv

class DataReader():
	"""

	"""
	def __init__(self, filepath='Data/BKGDAT.txt', delimiter=',', quotechar='"', num_data=10):
		self.filepath = filepath;
		self.delim = delimiter
		self.qchar = quotechar
		self.num_data = num_data;

	# Returns a tuple of lists; the first is a list of the headers in a CSV file and the 
	def getData(self):
		data = []
		header = []

		with open(self.filepath, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=self.delim, quotechar=self.qchar)
			header = reader.next()
			for i in range(self.num_data):
				data.append(reader.next())

		return header, data

if __name__ == "__main__":
	dr = DataReader(num_data=20)
	header, data = dr.getData()
	print data


	# visualizations, patterns in audience/destination, how are flights performing for different endpoints, bookings, etc.