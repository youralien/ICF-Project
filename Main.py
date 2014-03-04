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

	num_records = 500
	filename = 'Data/Normalized_BKGDAT_Filtered.txt'
	n = Network(num_records, filename)
	v = Visualizer()
	
	first_flight = n.f.getDrillDown(flights=[101], orgs=['DXB'], dests=['DMM'], bcs=['B'])
	
	xvals = np.linspace(-1, 0, 101)
	x = -first_flight['KEYDAY']
	y = first_flight['BKD']

	# x, y = zip(*sorted(zip(x, y), key=lambda tup: tup[0]))

	# yvals = np.interp(xvals, x, y, left=0)
	yvals = n.interp(xvals, x, y)

	plt.plot(x, y, 'o')
	plt.plot(xvals, yvals, '-x')
	plt.show()

if __name__ == '__main__':
	main()
	

