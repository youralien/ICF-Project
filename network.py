# Kyle McConnaughay (2015) & Ryan Louie (2017)
# Data Science Project with ICF International
# Franklin W. Olin College of Engineering
# Spring 2014	

import pandas as pd 
import numpy as np 


class Airport:
	def __init__(self, airport_code):
		pass

class Edge():
	def __init__(self):
		pass

def LoadBookings(n=10000):
	"""
	n: number of bookings to load from file

	returns: Bookings Data Frame
	"""
	if n =='all':
		return pd.read_csv("Data/BKGDAT.txt")
	else:
		return pd.read_csv("Data/BKGDAT.txt", nrows = n)

def FilterByOriginDestination(df):
	"""
	df: bookings, a pandas data frame

	returns: Bookings grouped by Origin and Destination
	"""
	return df.groupby(['ORG', 'DES'], sort = False)

def CreateNetwork(df_groupby):
	"""
	df_groupby: dataframe groupby object

	returns: a network, in the form of a dictionary, mapping a particular 
	flight path ('ORG', 'DES') to number of people who traveled on the path. 
	"""
	network = {}
	for flight_path, group in df_groupby:
		network[flight_path] = len(group)
	return network

if __name__ == "__main__":
	bookings = FilterByOriginDestination(LoadBookings())
	flight_network = CreateNetwork(bookings)
	print flight_network

