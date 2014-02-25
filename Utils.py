import csv

class Utils():
	@staticmethod
	def mapBookingClassToCabinHierarchy(bc):
		with open('Data/BC_Hierarchy.csv', 'r') as bc_file:
			reader = csv.reader(bc_file)
			for rank, booking_class, cabin in reader:
				if bc == booking_class:
					return cabin, rank

		raise Exception('Booking Class not found')

	@staticmethod
	def mapCabinToBookingClass(cabin):
		"""
			cabin can be either 'Y' or 'J' for economy or business class
		"""
		with open('Data/BC_Hierarchy.csv', 'r') as bc_file:
			reader = csv.reader(bc_file)
			return [bc for r, bc, c in reader if c == cabin]

	@staticmethod
	def date2DayOfWeek(date):
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