import matplotlib.pyplot as plt
import thinkplot
import thinkstats2
from math import isnan, isinf
import numpy as np

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

	def overbookingVsCabinLoadFactor(self, network):
		"""
		Investigates how overbooking, which varies with time, also varies 
		depending on the final cabin load factor.

		Overbooking is defined as AUTH / CAP, where AUTH varies with time.

		Final Cabin Load Factor is defined as TOTALBKD / CAP. 
		"""

		pass

	def timeVsFlights(self, network):
		x = network.timeseries()
		v = Visualizer()
		keys = x.keys()
		for i, key in enumerate(keys):
			print key, len(x[key])
		v.plotTimeSeries(x[keys[0]])

	def bookingCurves(self, network, org=None, des=None, flight=None,	
		cabin=None, bc=None, date_range=None):
		""" Plots booking curves for some subset of the data.
		
		A booking curve tracks the number of seats booked over time, starting 
		from the opening of ticket sales and ending close to departure
	
		"""

		df = network.f.getDrillDown(orgs=['DMM', 'DXB'], dests=['DXB', 'DMM'], bcs=['B'], flights=[101, 117])
		print df
		fltbk = network.f.getUniqueFlightsAndBookings(df)
		plt.figure()
		for g, d in fltbk:
			# print list(d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])
			BKD = list(d.sort(columns='KEYDAY', ascending=False)['BKD'])
			KEYDAY = list(-d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])

			ID = d['DATE'].first
			# print "ID ", ID
			# print "typeID: ", type(ID)
			BC = d['BC'].first
				
			plt.plot(KEYDAY, BKD)
			

		plt.title("Flight Number") 
		plt.xlabel('-KEYDAY')
		plt.ylabel('BKD')
		plt.show()
	
	def CDFCabinLoadFactor(self, network):
		""" Plots a Cumulative Distrubtion Function for Cabin Load Factor.

		Cabin Load Factor is defined as the ratio between the total number of 
		passengers booked for a flight and the total capacity of the plane.

		"""

		clf_list =  network.countFinalCabinLoadFactor().values()
		
		# remove any invalid floats 'nan' or 'inf'
		clf_list[:] = [e for e in clf_list if not isnan(e) and not isinf(e)]

		clf_cdf = thinkstats2.MakeCdfFromList(clf_list)
		thinkplot.Cdf(clf_cdf)
		thinkplot.show(title='Fraction of the cabin filled for flights at departure', 
						xlabel='Cabin Load Factor',
						ylabel='CDF'
						)

	def summaryStatistics(self, network):
		""" Creates a text file of summary statistics that may be useful
		to presenting to preliminary meetings with our advisors/sponsors.

		"""
		
		num_total_flights = len(network.f.filterUniqueFlights(n.entities))
		num_of_flights_between_cities = network.countFlightsBetweenCities()
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