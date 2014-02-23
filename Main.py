import numpy as np
import matplotlib.pyplot as plt

from FeatureFilter import FeatureFilter
from Utils import Utils
from Visualizer import Visualizer
from Network import Network

def main():
	num_records = 1000
	n = Network(num_records)
	# fltbk = n.f.getUniqueFlightsAndBookings()
	
	# for g, d in fltbk:
	# 	print list(d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])
	# 	BKD = list(d.sort(columns='KEYDAY', ascending=False)['BKD'])
	# 	KEYDAY = list(-d.sort(columns='KEYDAY', ascending=False)['KEYDAY'])

	# 	plt.plot(KEYDAY, BKD)
	# 	plt.show()
	# 	break
	x = n.f.getDrillDown(org='DMM', des='DXB')

if __name__ == '__main__':
	main()	
	# n = Network(1000)
	# orgdes = n.f.getUniqueOrgDes()
	# print type(orgdes)
	# (1/24/2013,164,"RUH","DXB"): 0.19069767441860466



