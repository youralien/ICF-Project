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

	num_records = 'all'
	filename = 'Data/Normalized_BKGDAT_Filtered_ZeroTOTALBKD.txt'
	n = Network(num_records, filename)
	v = Visualizer()
	v.stackedBookingCurve(n, orgs=['DXB'], dests=['DMM'], date_ranges=['1/1/2013'], flights=[101])

def overbookingVsCabinLoadFactor():
	num_records = 'all'
	filename = 'Data/BKGDAT_Filtered.txt'
	n = Network(num_records, filename)
	
	v = Visualizer()
	v.overbookingVsCabinLoadFactor(n, orgs=['DXB'],
									 dests=['DMM'], 
									 bcs=['Y'], 
									 #date_ranges=['1/1/2013'], 
									 normalized=False,
									 subplots=True
									 )

if __name__ == '__main__':
	# overbookingVsCabinLoadFactor()
	main()
	

