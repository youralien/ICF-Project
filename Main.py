from Network import Network
from Visualizer import Visualizer
from Utils import Utils
from FeatureFilter import FeatureFilter

def main():
	n = Network(1000)
	print n.countCabinCapacityPerFlight()	

if __name__ == '__main__':
	main()