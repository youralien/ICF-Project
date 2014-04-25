import pandas as pd
import numpy as np

from FeatureFilter import FeatureFilter
from Utils import Utils
from AirportCodes import AirportCodes

def KFoldSplit(X, y, identifiers, n_folds):
	pass

def encodeFlights(flights, interp_params, cat_encoding):
	data = [encodeFlight(flt, flt_df, interp_params, cat_encoding) for flt, flt_df in flights]
	X, y, identifiers = zip(*data)
	return X, y, identifiers

def encodeFlight(flt, df, interp_params, cat_encoding):
	"""
	args:
		interp_params: tuple of (start, stop, number_of_points) to use in
					   interpolate
		cat_encoding: tuple of (bin_size, date_reduction) specifying how 
					  compressed the BC and day of week categorical features
					  should be in the final feature matrix
	returns:
		tuple of (features, targets, flight IDs) suitable for use in training
		and graph generation
	"""
	X = None
	y = None
	identifiers = None
	bc_groupby = df.groupby('BC')
	bc_groupby = sortBCGroupby(bc_groupby)

	for bc, bc_df in bc_groupby:
		# Unpack relevant columns of the dataframe
		keyday = -1 * bc_df['KEYDAY']
		bkd = bc_df['BKD']
		auth = bc_df['AUTH']
		avail = bc_df['AVAIL']
		cap = bc_df['CAP']

		# Stack the numerical and categorical data into a feature matrix
		nums = encodeNumericalData(interp_params, keyday, bkd, auth, avail, cap)
		cats = encodeCategoricalData(flt, bc, len(nums), cat_encoding)
		features = hStackMatrices(cats, nums)

		# Save the new features in the X and y sets
		X = vStackMatrices(X, features)
		y = hStackMatrices(y, delta_bkd)
		identifiers = vStackMatrices(identifiers, np.array(flt+(bc,)))

		return X, y, identifiers

def encodeNumericalData(interp_params, keyday, bkd, auth, avail, cap):
	keyday, bkd, auth, avail = Utils.sortByIndex(keyday, bkd, auth, avail)
	keyday, bkd, auth, avail, cap = filterDataForKeyDay(
		keyday, bkd, auth, avail, cap)
	keyday, bkd, auth, avail, cap = interpolateFlight(
		interp_params, keyday, bkd, auth, avail, cap)

	# Create any other features
	delta_bkd = np.diff(bkd)
	cabin_load_factor = bkd / cap

	# Stack the numerical data into a feature matrix
	nums = [each[:-1] for each in [keyday, bkd, auth, avail, cap, clf]]
	nums = np.column_stack(nums)

	return nums

def interpolateFlight(interp_params, keyday, bkd, auth, avail, cap):
	start, stop, num_points = interp_params
	keyday_vals = np.linspace(start, stop, num_points)
	keyday, bkd, auth, avail = interpolate(
		keyday_vals, keyday, bkd, auth, avail)

	cap = float(cap.iget(0))
	cap = np.array([cap] * len(keyday))

	return keyday, bkd, auth, avail, cap

def encodeCategoricalData():
	pass

def sortBCGroupby(groupby):
	tups = [(bc, bc_df) for bc, bc_df in groupby]
	return sorted(tups, key=lambda tup: Utils.compareBCs(tup[0]))

def interpolate(keyday_vals, keydays, *args):
    interps = [np.interp(keyday_vals, keydays, arg, left=0) for arg in args]
    return interps

def filterDataForKeyDay(time, keydays, *args):
    index = next((i for i, k in enumerate(keydays) if k > time), keydays[0])
    filtered_keydays = keydays[index:]
    filtered_args = [arg[index:] for arg in args]
    return [filtered_keydays] + filtered_args

def vStackMatrices(x, new_x):
    return stackMatrices(x, new_x, np.vstack)

def hStackMatrices(x, new_x):
    return stackMatrices(x, new_x, np.hstack)

def colStackMatrices(x, new_x):
    return stackMatrices(x, new_x, np.column_stack)

def stackMatrices(x, new_x, fun):
    if x is None:
        x = new_x
    else: 
        x = fun((x, new_x))

    return x

def main():
	# Set parameters for opening the data
	# num_records = 'all'
	num_records = 'all'
	csvfile = "Data/BKGDAT_ZeroTOTALBKD.txt"

	# Set parameters for filtering the data
	market = AirportCodes.London
	orgs=[AirportCodes.Dubai, market]
	dests=[AirportCodes.Dubai, market]
	cabins=["Y"]

	# Get the data, filter it, and group it by flight
	f = FeatureFilter(num_records, csvfile)
	data = f.getDrillDown(orgs=orgs, dests=dests, cabins=cabins)
	unique_flights = f.getUniqueFlights(data)

	# Encode the flights
	x, y, i = encodeFlights(unique_flights, None)
	print type(x), type(y), type(i)

if __name__ == '__main__':
	main()