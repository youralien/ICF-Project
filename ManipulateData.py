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
	df = pd.read_csv(oldfilename, nrows=1000)

	old_row = df.loc[1, ['DATE', 'FLT', 'ORG', 'DES', 'BC']]
	old_index = 0
	max_keyday = -1

	for i in range(len(df)):
		capacity = df.loc[i, 'CAP']
		df.loc[i, 'BKD'] = float(df.loc[i, 'BKD']) / capacity
		df.loc[i, 'AUTH'] = float(df.loc[i, 'AUTH']) / capacity
		df.loc[i, 'TOTALBKD'] = float(df.loc[i, 'TOTALBKD']) / capacity
		df.loc[i, 'AVAIL'] = float(df.loc[i, 'AVAIL']) / capacity

		max_keyday = max(max_keyday, df.loc[i, 'KEYDAY'])
		
		if not (old_row == df.loc[i, ['DATE', 'FLT', 'ORG', 'DES', 'BC']]).all():
			for j in range(old_index, i):
				df.loc[j, 'KEYDAY'] = float(df.loc[j, 'KEYDAY']) / max_keyday

			old_index = i
			max_keyday = df.loc[i, 'KEYDAY']
			old_row = df.loc[i, ['DATE', 'FLT', 'ORG', 'DES', 'BC']]

	df.to_csv(newfilename, index=False)


if __name__ == "__main__":
	# RemoveHourMinuteSecond('Data/BKGDAT.txt', 'Data/BKGDAT_Filtered.txt')
	NormalizeData('Data/BKGDAT_Filtered.txt', 'Data/Normalized_BKGDAT_Filtered.txt')