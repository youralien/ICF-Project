import matplotlib.pyplot as plt

class Visualizer():

	def __init__(self):
		pass

	def plotTimeSeries(self, time_series):
		"""
		time series is a dictionary of (dates: flight counts)
		"""
		sorted_keys = sorted(time_series.keys())
		print range(len(sorted_keys))
		print [time_series[date] for date in sorted_keys]

		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		ax.plot(range(len(sorted_keys)), [time_series[date] for date in sorted_keys])
		ax.xaxis.set_ticklabels(sorted_keys)
		for tick in ax.xaxis.get_major_ticks():
			tick.label.set_fontsize(8)
			tick.label.set_rotation('vertical')
		plt.show()

	def 
	def overbookingVsCabinLoadFactor(self, network):
		utilization = network.countOverbookedAndCabinLoadFactor()
		print utilization
		plt.plot(utilization)
		plt.xlabel('Cabin Load Factor (units?)')
		plt.ylabel('Overbooking (units?)')
		plt.show()

	def timeVsFlights(self, network):
		x = network.timeseries()
		v = Visualizer()
		keys = x.keys()
		for i, key in enumerate(keys):
			print key, len(x[key])
		v.plotTimeSeries(x[keys[0]])

	def summaryStatistics(self):
		num_records = 'all'
		n = Network(num_records)

		num_total_flights = len(n.f.filterUniqueFlights(n.entities))
		num_of_flights_between_cities = n.countFlightsBetweenCities()
		num_routes = len(num_of_flights_between_cities.keys())

		f = open('ICF Summary Statistics.txt', 'w')

		
		f.write("Total Number of Flights: " + str(num_total_flights) + "\n")
		
		f.write("Total Number of Directional Routes: " + str(num_routes) + "\n")

		f.writelines([str(citypath) + ': ' + str(num_flights) + "\n" for citypath, num_flights in num_of_flights_between_cities.items()])
		
		f.close()

def main():
	pass

if __name__ == '__main__':
	main()