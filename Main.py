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

	num_records = 10000
	filename = 'Data/BKGDAT_Filtered.txt'
	n = Network(num_records, filename)
	flights = n.getDrillDown(orgs=['DXB'], dests=['DMM'])
	sorted_flights = n.f.getUniqueFlightsAndBookings(flights)

	for g, df in sorted_flights.groupby('BC'):
		print g, df
		brea


if __name__ == '__main__':
	# overbookingVsCabinLoadFactor()
	main()
	

