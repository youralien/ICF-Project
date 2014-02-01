import csv

class DataReader():
	def __init__(self, filepath='Data/BKGDAT.txt', delimiter=',', quotechar='"', num_data=10):
		self.filepath = filepath;
		self.delimiter = delimiter
		self.quotechar = quotechar
		self.num_data = num_data;

	def getData(self):
		data = []
		header = []

		with open(self.filepath, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=self.delimiter, quotechar=self.quotechar)
			header = reader.next()
			for i in range(self.num_data):
				data.append(reader.next())

		return header, data

if __name__ == "__main__":
	dr = DataReader()
	header, data = dr.getData()
	print data


	# visualizations, patterns in audience/destination, how are flights performing for different endpoints, bookings, etc.