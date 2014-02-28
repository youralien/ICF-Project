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

	num_records = 100000
	n = Network(num_records)
	v = Visualizer()
	# v.CDFCabinLoadFactor(n)
	v.overbookingCurves(n)

if __name__ == '__main__':
	main()
	# num_records = 'all'
	# n = Network(num_records)
	# v = Visualizer()

	# booked = n.countTotalBookedPerFlight()

	

