import matplotlib.pyplot at plt

import thinkstats2
import thinkplot


def CDFofCLF(norm_data, title, xlabel, ylabel):
	"""
	Displays a Cumulative Distribution Function (CDF) of 
	Cabin Load Factor (CLF)
	args:
		norm_data: normalized dataframe for aggregate or separate markets. 
		title: plot title, a string
		xlabel: plot xlabel, a string
		ylabel, plot ylabel, a string
	"""
	unique_flights = f.getUniqueFlights(norm_data)

	CLFs = [d.iget(0) for g, d in unique_flights.TOTALBKD]

	cdf_of_clf = thinkstats2.MakeCdfFromList(CLFs)
	
	thinkplot.Cdf(cdf_of_clf)
	thinkplot.Show(	title='Economy CLF for all Markets from January - March 2013',
               		xlabel='Cabin Load Factor',
                   	ylabel='CDF')

def main():
	pass

if __name__ == '__main__':
	main()