import numpy as np
import matplotlib.pyplot as plt

from FeatureFilter import FeatureFilter
from Utils import Utils
from Visualizer import Visualizer
from Network import Network

def main():

	num_records = 1000
	n = Network(num_records)
	fltbk = n.f.getUniqueFlightsAndBookings()

	for g, d in fltbk:
		print list(d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])
		BKD = list(d.sort(columns='KEYDAY', ascending=False)['BKD'])
		KEYDAY = list(-d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])

		ID = d['DATE'].first
		print "ID ", ID
		print "typeID: ", type(ID)
		BC = d['BC'].first
	
		# plt.figure()	
		# plt.plot(KEYDAY, BKD)
		# plt.title("Flight Number") 
		# plt.xlabel('-KEYDAY')
		# plt.ylabel('BKD')
		# plt.show()
		# break

if __name__ == '__main__':
	main()

	

