# Kyle McConnaughay (2015) & Ryan Louie (2017)
# Data Science Project with ICF International
# Franklin W. Olin College of Engineering
# Spring 2014	

import pandas as pd
import matplotlib.pyplot as plt
import thinkplot
import thinkstats2
from math import isnan, isinf
import numpy as np
from Utils import Utils

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

	def timeVsFlights(self, network):
		x = network.timeseries()
		v = Visualizer()
		keys = x.keys()
		for i, key in enumerate(keys):
			print key, len(x[key])
		v.plotTimeSeries(x[keys[0]])

	def authCurves(self, network, orgs=None, dests=None, flights=None, 
					cabins=None, bcs=None, date_ranges=None):
		""" Plots AUTH curves for some subset of the data.
		
		AUTH is stated at the level of a cabin-booking class.  AUTH changes
		with time starting from the opening of ticket sales and ending close 
		to departure.   Note that you only have to look at two booking 
		classes (BC) for purpose of overbooking: Y class for Y cabin and J 
		class for J cabin. This is because those classes always have the
		maximum AUTH among all classes in a cabin at a given point of time 
		(they are at the top in hierarchy). 
	
		"""

		df = network.f.getDrillDown(orgs=orgs, dests=dests, flights=flights,
							cabins=cabins, bcs=bcs, date_ranges=date_ranges)

		fltbk = network.f.getUniqueFlightsAndBookings(df)

		plt.figure()
		for g, d in fltbk:
			AUTH = np.array(d.sort(columns='KEYDAY', ascending=False)['AUTH'])
			KEYDAY = np.array(-d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])

			plt.plot(KEYDAY, AUTH)

		title = Utils.createTitleForFeatures(orgs,dests,flights,cabins,bcs,date_ranges)
		plt.title(title)
		plt.xlabel('-KEYDAY')
		plt.ylabel('AUTH')
		plt.show()

	def overbookingCurves(self, network, orgs=None, dests=None, flights=None, 
						cabins=None, bcs=None, date_ranges=None, normalized=True):
		""" Plots overbooking curves for some subset of the data.
		
		Overbooking is defined where AUTH > CAP.  We plot overbooking as a 
		ratio between AUTH and CAP.  Overbooking varies with time.
	
		"""
		df = network.f.getDrillDown(orgs=orgs, dests=dests, flights=flights,
							cabins=cabins, bcs=bcs, date_ranges=date_ranges)

		fltbk = network.f.getUniqueFlightsAndBookings(df)

		plt.figure()
		
		if normalized:
			for g, d in fltbk:
				# normalized AUTH == OVERBOOKED
				AUTH = np.array(d.sort(columns='KEYDAY', ascending=False)['AUTH'])
				
				# ignore time series that are not overbooked
				if not Utils.isOverbooked(AUTH):
					continue

				KEYDAY = np.array(-d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])
				
				plt.plot(KEYDAY, AUTH)
		else:
			for g, d in fltbk:
				AUTH = np.array(d.sort(columns='KEYDAY', ascending=False)['AUTH'])
				CAP = float(d.iloc[0]['CAP'])
				OVRBKD = AUTH/CAP

				# ignore time series that are not overbooked
				if not Utils.isOverbooked(OVRBKD):
					continue

				KEYDAY = np.array(-d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])
				
				plt.plot(KEYDAY, OVRBKD)

		title = Utils.createTitleForFeatures(orgs,dests,flights,cabins,bcs,date_ranges)
		plt.title(title)
		plt.xlabel('-KEYDAY')
		plt.ylabel('Percentage Overbooked: AUTH / CAP')
		plt.show()

	def overbookingVsCabinLoadFactor(self, network, orgs=None, dests=None, flights=None, 
									cabins=None, bcs=None, date_ranges=None, 
									normalized=True, subplots=True):
		""" Plots how overbooking varies with Cabin load factor.  Final Cabin Load Factor
		for a particular flight booking class is binned into three separate categories:
		
		Overbooked: CLF > 1
		Underbooked: CLF < .8
		Optimumly booked: .8 < CLF < 1
		
		"""
		df = network.f.getDrillDown(orgs=orgs, dests=dests, flights=flights,
							cabins=cabins, bcs=bcs, date_ranges=date_ranges)

		fltbk = network.f.getUniqueFlightsAndBookings(df)
		# TODO: allow for countFinalCabinLoadFactor to use normalized data		
		CLF_dict = network.countFinalCabinLoadFactor()

		plt.figure()
		# preparing to capture the legend handles
		legend_over = None
		legend_under = None
		legend_optimum = None

		if normalized:
			for g, d in fltbk:
				# normalized AUTH == OVERBOOKED
				AUTH = np.array(d.sort(columns='KEYDAY', ascending=False)['AUTH'])
				
				# ignore time series that are not overbooked
				if not Utils.isOverbooked(AUTH):
					continue

				KEYDAY = np.array(-d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])
				DATE = d.iloc[0]['DATE']
				FLT = d.iloc[0]['FLT']
				ORG = d.iloc[0]['ORG']
				DES = d.iloc[0]['DES']
				
				#TODO: See CLF_dict (above)
				CABIN_LOAD_FACTOR = CLF_dict[(DATE, FLT, ORG, DES)]

				if CABIN_LOAD_FACTOR > 1:
					plt.plot(KEYDAY, AUTH, 'r')
				elif CABIN_LOAD_FACTOR < .95: 
					plt.plot(KEYDAY, AUTH, 'y')
				else:
					plt.plot(KEYDAY, AUTH, 'g')
		else:
			for g, d in fltbk:
				
				AUTH = np.array(d.sort(columns='KEYDAY', ascending=False)['AUTH'])
				CAP = float(d.iloc[0]['CAP'])
				OVRBKD = AUTH/CAP

				# ignore time series that are not overbooked
				if not Utils.isOverbooked(OVRBKD):
					continue

				KEYDAY = np.array(-d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])
				DATE = d.iloc[0]['DATE']
				FLT = d.iloc[0]['FLT']
				ORG = d.iloc[0]['ORG']
				DES = d.iloc[0]['DES']

				CABIN_LOAD_FACTOR = CLF_dict[(DATE, FLT, ORG, DES)]

				


				if CABIN_LOAD_FACTOR > 1:
					plt.subplot(311) if subplots else None
					if not legend_over:
						legend_over, = plt.plot(KEYDAY, OVRBKD , 'r')
					else:
						plt.plot(KEYDAY, OVRBKD , 'r')
				elif CABIN_LOAD_FACTOR < .95: 
					plt.subplot(313) if subplots else None
					if not legend_under:
						legend_under, = plt.plot(KEYDAY, OVRBKD, 'y')
					else:
						plt.plot(KEYDAY, OVRBKD, 'y')
				else:
					plt.subplot(312) if subplots else None
					if not legend_optimum:
						legend_optimum, = plt.plot(KEYDAY, OVRBKD, 'g')
					else:
						plt.plot(KEYDAY, OVRBKD, 'g')

		title = Utils.createTitleForFeatures(orgs,dests,flights,cabins,bcs,date_ranges)
		plt.subplot(311) if subplots else None
		plt.title(title)
		plt.xlabel('-KEYDAY')
		plt.ylabel('Percentage Overbooked: AUTH / CAP')

		plt.legend([legend_over, legend_under, legend_optimum], ["Cabin Load Factor > 1", "Cabin Load Factor < .95", "Optimum Cabin Load Factor"])
		plt.show()

	def bookingCurves(self, network, orgs=None, dests=None, flights=None, 
						cabins=None, bcs=None, date_ranges=None):
		""" Plots booking curves for some subset of the data.
		
		A booking curve tracks the number of seats booked over time, starting 
		from the opening of ticket sales and ending close to departure
	
		"""

		df = network.f.getDrillDown(orgs=orgs, dests=dests, flights=flights,
							cabins=cabins, bcs=bcs, date_ranges=date_ranges)
		fltbk = network.f.getUniqueFlightsAndBookings(df)

		plt.figure()
		for g, d in fltbk:
			BKD = list(d.sort(columns='KEYDAY', ascending=False)['BKD'])
			KEYDAY = list(-d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])

			ID = d['DATE'].first
			BC = d['BC'].first
				
			plt.plot(KEYDAY, BKD)
			
		title = Utils.createTitleForFeatures(orgs,dests,flights,cabins,bcs,date_ranges)
		plt.title(title)
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
		thinkplot.show(title='Fraction of the cabin filled at departure', 
					   xlabel='Cabin Load Factor',
					   ylabel='CDF')

	def stackedBookingCurve(self, network, orgs=None, dests=None, 
		                     flights=None, cabins=None, bcs=None, 
		                     date_ranges=None):
		"""
		Generate a summative booking curve for a given flight. In order for this
		function to work properly the arguments must specify one specific flight
		(or a subset of the booking classes on a specific flight). Additionally,
		the network must have been create using a normalized data set.
		"""

		first_flights = network.f.getDrillDown(orgs=orgs, dests=dests, 
											   flights=flights, cabins=cabins, 
											   bcs=bcs, date_ranges=date_ranges)
		groupedByBookings = network.f.getUniqueFlightsAndBookings(first_flights)
		xvals = np.linspace(-1, 0, 101)
		interps = None
		labels = [g[4] for g, d in groupedByBookings]

		for g, d in groupedByBookings:
			keydays = -d['KEYDAY']
			booked = d['BKD']
			yvals = network.interp(xvals, keydays, booked)
			if interps == None:
				interps = yvals
			else:
				interps = np.vstack((interps, yvals))

		# interps is my matrix
		m, n = interps.shape
		interps_sum = np.zeros((m,n))
		for i in range(m-1):
			for j in range(i+1, m):
				interps_sum[j] += interps[i]

		for i in range(m):
			plt.plot(xvals, interps_sum[i])

		plt.legend(labels, loc=6, prop={'size': 14})
		plt.title('Summative Booking Curve\n' + Utils.createTitleForFeatures(orgs, dests, flights, cabins, bcs, date_ranges))
		plt.xlabel('Normalized Keyday')
		plt.ylabel('Normalized Booked')
		plt.show()

	def numFlightsByDayOfWeek(self, network):
		flights = network.f.getUniqueFlights()
		day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
		num_flights = np.zeros(7)

		for group, df in flights:
			date = group[0]
			day = Utils.date2DayOfWeek(date)
			num_flights[day_names.index(day)] += 1

		fracs = [flight / num_flights.sum() for flight in num_flights]
	
		plt.pie(fracs, labels=day_names, autopct='%1.1f%%', colors=("b","g","r","y", "c", "w", "m"))
		plt.title('Total Number of Flights by Day of Week')
		plt.show()

	def numPassengersByDayOfWeek(self, network):
		flights = network.countTotalBookedPerFlight()
		day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
		num_peeps = np.zeros(7)

		for key, val in flights.items():
			date = key[0]
			day = Utils.date2DayOfWeek(date)
			num_peeps[day_names.index(day)] += sum(val.values())

		fracs = [val / num_peeps.sum() for val in num_peeps]

		plt.pie(fracs, labels=day_names, autopct='%1.1f%%', colors=("b","g","r","y", "c", "w", "m"))
		plt.title('Total Number of Passengers by Day of Week')
		plt.show()


	def summaryStatistics(self, network):
		""" Creates a text file of summary statistics that may be useful
		to presenting to preliminary meetings with our advisors/sponsors.

		"""

		flights = network.countTotalBookedPerFlight()
		networkData = [sum(flights[key].values()) for key in flights.keys()]
		networkSeries = pd.Series(networkData).describe()
		edgeData = {}

		for flight, data in flights.items():
			org_des = flight[2:]
			edgeData[org_des] = edgeData.get(org_des, [])
			edgeData[org_des].append(sum(data.values()))

		with open('ICF_Summary_Statistics.txt', 'w') as f:
			f.write('Network Summary\n')
			Utils.writeSeriesToFile(f, networkSeries, indent='	')

			f.write('\nRoute Summaries\n\n')
			for org_des, booked in edgeData.items():
				f.write(org_des[0] + ' -> ' + org_des[1] + '\n')
				statsSeries = pd.Series(booked).describe()
				Utils.writeSeriesToFile(f, statsSeries, indent='	')
				f.write('\n')

def main():
	pass

if __name__ == '__main__':
	main()
