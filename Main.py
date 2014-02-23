import numpy as np
import matplotlib.pyplot as plt

from FeatureFilter import FeatureFilter
from Utils import Utils
from Visualizer import Visualizer
from Network import Network

def main():
	n = Network(1000)
	fltbk = n.f.getUniqueFlightsAndBookings()
	
	for g, d in fltbk:
		print list(d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])
		BKD = list(d.sort(columns='KEYDAY', ascending=False)['BKD'])
		KEYDAY = list(-d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])

		plt.plot(KEYDAY, BKD)
		plt.show()
		break

if __name__ == '__main__':
	main()	
	orgdes = n.f.getUniqueOrgDes()
	print type(orgdes)
	n = Network(1000)


