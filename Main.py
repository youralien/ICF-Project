import numpy as np
import matplotlib.pyplot as plt

from FeatureFilter import FeatureFilter
from Utils import Utils
from Visualizer import Visualizer
from Network import Network

def main():

	num_records = 'all'
	n = Network(num_records)
	v = Visualizer()
	v.plotCDFCabinLoadFactor(n)
	# v.bookingCurves(n,org=['DXB'])

if __name__ == '__main__':
	main()

	

