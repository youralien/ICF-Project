def RemoveHourMinuteSecond(oldfilename, newfilename):

	oldfile = open(oldfilename, 'r')
	newfile = open(newfilename, 'w')

	for oldline in oldfile:
		newline = oldline.replace(' 0:00:00', '')
		newfile.write(newline)

	oldfile.close()
	newfile.close()
	

if __name__ == "__main__":
	RemoveHourMinuteSecond('Data/BKGDAT.txt', 'Data/BKGDAT_Filtered.txt')
	