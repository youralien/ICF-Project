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

		plt.title('Orgs DXB, Dests DMM, bcs Y and J, flights 101 for first 100,000 rows')
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
				KEYDAY = np.array(-d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])
				
				plt.plot(KEYDAY, AUTH)
		else:
			for g, d in fltbk:
				AUTH = np.array(d.sort(columns='KEYDAY', ascending=False)['AUTH'])
				CAP = d.iloc[0]['CAP']
				KEYDAY = np.array(-d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])
				
				plt.plot(KEYDAY, AUTH/float(CAP))

		plt.title('Orgs DXB, Dests DMM, bcs Y and J, flights 101 for first 100,000 rows')
		plt.xlabel('-KEYDAY')
		plt.ylabel('Percentage Overbooked: AUTH / CAP')
		plt.show()

	def overbookingVsCabinLoadFactor(self, network, orgs=None, dests=None, flights=None, 
							cabins=None, bcs=None, date_ranges=None, normalized=True):
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
				KEYDAY = np.array(-d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])
				DATE = d.iloc[0]['DATE']
				FLT = d.iloc[0]['FLT']
				ORG = d.iloc[0]['ORG']
				DES = d.iloc[0]['DES']
				
				#TODO: See CLF_dict (above)
				CABIN_LOAD_FACTOR = CLF_dict[(DATE, FLT, ORG, DES)]

				if CABIN_LOAD_FACTOR > 1:
					plt.plot(KEYDAY, AUTH, 'x-')
				elif CABIN_LOAD_FACTOR < .8: 
					plt.plot(KEYDAY, AUTH, 'o-')
				else:
					plt.plot(KEYDAY, AUTH, '-^')
		else:
			for g, d in fltbk:
				
				AUTH = np.array(d.sort(columns='KEYDAY', ascending=False)['AUTH'])
				CAP = float(d.iloc[0]['CAP'])
				KEYDAY = np.array(-d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])
				DATE = d.iloc[0]['DATE']
				FLT = d.iloc[0]['FLT']
				ORG = d.iloc[0]['ORG']
				DES = d.iloc[0]['DES']

				CABIN_LOAD_FACTOR = CLF_dict[(DATE, FLT, ORG, DES)]

				


				if CABIN_LOAD_FACTOR > 1:
					if not legend_over:
						legend_over, = plt.plot(KEYDAY, AUTH/CAP , 'x-')
					else:
						plt.plot(KEYDAY, AUTH/CAP , 'x-')
				elif CABIN_LOAD_FACTOR < .8: 
					if not legend_under:
						legend_under, = plt.plot(KEYDAY, AUTH/CAP, 'o-')
					else:
						plt.plot(KEYDAY, AUTH/CAP, 'o-')
				else:
					if not legend_optimum:
						legend_optimum, = plt.plot(KEYDAY, AUTH/CAP, '^-')
					else:
						plt.plot(KEYDAY, AUTH/CAP, '^-')

		plt.title('Orgs DXB, Dests DMM, bcs Y and J, flights 101 for first 100,000 rows')
		plt.xlabel('-KEYDAY')
		plt.ylabel('Percentage Overbooked: AUTH / CAP')

		plt.legend([legend_over, legend_under, legend_optimum], ["Cabin Load Factor > 1", "Cabin Load Factor < .8", ".8 < Cabin Load Factor < 1"])
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
			
		plt.title("DrillDown: Origin DXB DMM, Destination DMM DXB, BC B, FLT 101 for first 100,000 rows") 
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

	def stackedBookingCurves(self):
		pass

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
