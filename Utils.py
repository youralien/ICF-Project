import csv

class Utils():
	def mapBookingClassToCabinHierarchy(self, bc):
		with open('Data/BC_Hierarchy.csv', 'r') as bc_file:
			reader = csv.reader(bc_file)
			for rank, booking_class, cabin in reader:
				if bc == booking_class:
					return cabin, rank

		raise Exception('Booking Class not found')

	def date2DayOfWeek(self, date):
		"""
		date : string 'm/d/yyyy' or 'mm/dd/yyyy'
		"""
		month, day, year = date.split('/')
		month, day, year = int(month), int(day), int(year)

		day = datetime.date(year, month, day) 
		return day.strftime("%A")

def main():
	pass

if __name__ == '__main__':
	main()