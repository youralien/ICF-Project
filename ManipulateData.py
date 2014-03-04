# Kyle McConnaughay (2015) & Ryan Louie (2017)
# Data Science Project with ICF International
# Franklin W. Olin College of Engineering
# Spring 2014	

import pandas as pd
from pandas.util.testing import assert_series_equal

def RemoveHourMinuteSecond(oldfilename, newfilename):

	oldfile = open(oldfilename, 'r')
	newfile = open(newfilename, 'w')

	for oldline in oldfile:
		newline = oldline.replace(' 0:00:00', '')
		newfile.write(newline)

	oldfile.close()
	newfile.close()

def NormalizeData(oldfilename, newfilename):
	oldfile = open(oldfilename, 'r')
	max_keydays = {}
	header = True
	for oldline in oldfile:
		if header:
			header = False
			continue
		tokens = oldline.strip().split(',')
		keyday = float(tokens[9])
		flight = tuple(tokens[:4])
		max_keydays[flight] = max(max_keydays.get(flight, 0), keyday)

	oldfile.close()

	oldfile = open(oldfilename, 'r')
	newfile = open(newfilename, 'w')

	precision = 2

	header = True
	index = 1
	for oldline in oldfile:
		if header:
			header = False
			newfile.write(oldline)
			continue

		tokens = oldline.strip().split(',')
		date = tokens[0]
		flt = tokens[1]
		org = tokens[2]
		des = tokens[3]
		cap = float(tokens[4])
		bc = tokens[5]
		bkd = int(tokens[6])
		avail = int(tokens[7])
		auth = int(tokens[8])
		keyday = float(tokens[9])
		totalbkd = int(tokens[10])
		
		if cap == 0:
			continue

		flight = tuple(tokens[:4])

		norm_bkd = round(bkd / cap, precision)
		norm_avail = round(avail / cap, precision)
		norm_auth = round(auth / cap, precision)
		norm_keyday = round(keyday / max_keydays[flight], precision)
		norm_totalbkd = round(totalbkd / cap, precision)
		
		new_tokens = tokens[:6 ]+ [str(norm_bkd), str(norm_avail), str(norm_auth), str(norm_keyday), str(norm_totalbkd)]
		newline = ','.join(new_tokens) + '\n'
		newfile.write(newline)

		print index
		index += 1

	oldfile.close()
	newfile.close()


def CopyNRowsOfData(oldfilename, n):
	oldfile = open(oldfilename, 'r')
	newfile = open('Data/Moved.txt', 'w')

	for i, row in enumerate(oldfile):
		newfile.write(row)
		if i == n:
			break

	oldfile.close()
	newfile.close()


if __name__ == "__main__":
	# RemoveHourMinuteSecond('Data/BKGDAT.txt', 'Data/BKGDAT_Filtered.txt')
	NormalizeData('Data/BKGDAT_Filtered.txt', 'Data/Normalized_BKGDAT_Filtered.txt')
	# CopyNRowsOfData('Data/Normalized_BKGDAT_Filtered.txt', 1000)
