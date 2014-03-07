# Kyle McConnaughay (2015) & Ryan Louie (2017)
# Data Science Project with ICF International
# Franklin W. Olin College of Engineering
# Spring 2014

import numpy as np
import matplotlib.pyplot as plt

from FeatureFilter import FeatureFilter
from Utils import Utils
from Visualizer import Visualizer
from Network import Network

def main():

	num_records = 1500
	filename = 'Data/Normalized_BKGDAT_Filtered.txt'
	n = Network(num_records, filename)
	v = Visualizer()
	
	first_flight = n.f.getDrillDown(flights=[101], orgs=['DXB'], dests=['DMM'], date_ranges=['1/1/2013'])
	lala = n.f.getUniqueFlightsAndBookings(first_flight)
	xvals = np.linspace(-1, 0, 101)
	interps = None
	for g, d in lala:
		keydays = -d['KEYDAY']
		booked = d['BKD']
		yvals = n.interp(xvals, keydays, booked)
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

	# twentyfifths = []
	# medians = []
	# seventyfifths = []

	# for i in interps.transpose():
	# 	medians.append(np.median(i))
		# twentyfifths.append(np.percentile(i, 25))
		# seventyfifths.append(np.percentile(i, 75))

	# plt.plot(xvals, medians)
	# plt.plot(xvals, twentyfifths)
	# plt.plot(xvals, seventyfifths)
	plt.show()
	
	# print first_flight


	# x, y = zip(*sorted(zip(x, y), key=lambda tup: tup[0]))

	# yvals = np.interp(xvals, x, y, left=0)
	# yvals = n.interp(xvals, x, y)

	# plt.plot(x, y, 'o')
	# plt.plot(xvals, yvals, '-x')
	# plt.show()

def overbookingVsCabinLoadFactor():

	num_records = 10000
	filename = 'Data/BKGDAT_Filtered.txt'
	n = Network(num_records, filename)
	
	v = Visualizer()
	v.overbookingVsCabinLoadFactor(n, orgs=['DXB', 'DMM'],
									 dests=['DMM', 'DXB'], 
									 bcs=['Y'], 
									 #date_ranges=['1/1/2013'], 
									 normalized=False)

if __name__ == '__main__':
	
	overbookingVsCabinLoadFactor()
	

