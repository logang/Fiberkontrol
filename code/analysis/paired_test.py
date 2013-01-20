import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""
This class is for comparing the effect two different trial conditions (i.e. novel object interaction vs. social interaction)
across a cohort of mice.
The score used to measure the effect of each condition is determined by which files are passed in to the class.
An example of a score is the maximum fluorescence value within an interaction event, or the area under
the curve within a window surrounding the event.

"""
class PairAnalyze( object ):

		def __init__(self, options):
				"""
				Initialize the FiberAnalyze class using options object from OptionsParser.
				"""
				# values from option parser
				self.output_path = options.output_path
				self.n_path = options.novel_path #novel_path can represent any
				self.s_path = options.social_path

		def Load( self ):
			self.n_files = []
			n_listing = os.listdir(self.n_path)
			for infile in n_listing:
				try:
					self.n_files.append(np.load(self.n_path + '/' + infile))
					print infile
				except:
					print "FILE DOESN'T WORK: ", infile

			self.s_files = []
			s_listing = os.listdir(self.s_path)
			for infile in s_listing:
				try:
					self.s_files.append(np.load(self.s_path + '/' + infile))
					print infile
				except:
					print "FILE DOESN'T WORK: ", infile


			self.n_scores = [] #An array where each entry is an array holding the scores for all events of a single mouse's trial
			self.s_scores = []
			self.n_start_times = [] #An array where each entry is an array holding the start time of all events of a single mouse's trial
			self.s_start_times = []
			self.n_end_times = [] # An array where each entry is an array holding the end time of all events of a single mouse's trial
			self.s_end_times = []
			self.n_windows = [] # An array where each entry is the exact length (in s) of the window used to calculate the score (i.e. area under curve)
			self.s_windows = []

			for f in self.n_files:
				self.n_scores.append(f['scores'])
				self.n_start_times.append(f['event_times'])
				self.n_end_times.append(f['end_times'])
				self.n_windows.append(f['window_size'])

			for f in self.s_files:
				self.s_scores.append(f['scores'])
				self.s_start_times.append(f['event_times'])
				self.s_end_times.append(f['end_times'])
				self.s_windows.append(f['window_size'])


		def CompareEvent( self, index, test="ttest"):
			"""
			Compare the scores of the index'th event across paired test conditions, with the
			null hypothesis that the mean score of the two test conditions do not differ.
			Example: If index = 0, compare the scores of the first event between condition s and n.
			Return a measure of significance of the difference between the two test conditions
			as determined by the specified statistical test.
			"""


			n_score_list = []
			s_score_list = []

			for f in self.n_scores:
				n_score_list.append(f[index]) 

			for f in self.s_scores:
				s_score_list.append(f[index])  

			print "n_score_list ", n_score_list
			print "s_score_list ", s_score_list

			if test == "ttest":
				[tvalue, pvalue] = stats.ttest_rel(s_score_list, n_score_list)
				print "normalized area tvalue: ", tvalue, " normalized area pvalue: ", pvalue
			if test == "wilcoxon":
				[zstatistic, pvalue] = stats.wilcoxon(s_score_list, n_score_list)
				print "normalized area zstatistic: ", zstatistic, " normalized area pvalue: ", pvalue



		def EventLength_vs_Score( self, scores, start_times, end_times, just_first=False, title=None ):
			"""
			Produces a scatter plot of the length of each event vs the score (i.e. area under the curve) 
			associated with that event.

			just_first parameter indicates whether to only include the first interaction event of a trial
			"""

			event_starts = []
			event_ends = []
			event_scores = []

			event_lengths = []


			for f in scores:
				if(just_first):
					event_scores.append(f[0])
				else:
					for d in f:
						event_scores.append(d)


			for f in start_times:
				if np.shape(f) != ():
					if(just_first):
						event_starts.append(f[0])
					else:
						for i in range(len(f)):
							event_starts.append(f[i])
				else:
						event_starts.append(f)


			for f in end_times:
				if np.shape(f) != ():
					if(just_first):
						event_ends.append(f[0])
					else:
						for i in range(len(f)):
							event_ends.append(f[i])
				else:
						event_ends.append(f)

			event_lengths = np.array(event_ends) - np.array(event_starts)

			print "median length", np.median(event_lengths)
			print "stdev length", np.sqrt(np.var(event_lengths))

			print "mean length", np.mean(event_lengths)

			A = np.array([event_lengths, np.ones(len(event_lengths))])
			w = np.linalg.lstsq(A.T, event_scores)[0]

			plt.figure()
			plt.plot(event_lengths, w[0]*event_lengths + w[1], 'r-', event_lengths, event_scores, 'o')
			plt.ylim([0, 1.0])
			plt.xlabel('Interaction Time [s]')
			plt.ylabel('Sharpness of first peak: ' r'$\frac{\sum\delta F/F}{\max(peak)}}$ over window of 1s')
			if title is not None:
				plt.title(title)

def test_PairAnalyze(options):
	"""
	Test the PairAnalyze class.
	"""

	PA = PairAnalyze( options )
	PA.Load()
	PA.CompareEvent(0)
	PA.CompareEvent(0, "wilcoxon" )
	print "Novel interaction time:"
	PA.EventLength_vs_Score(PA.n_scores, PA.n_start_times, PA.n_end_times, just_first=True, title="Novel")
	print "Social interaction time:"
	PA.EventLength_vs_Score(PA.s_scores, PA.s_start_times, PA.s_end_times, just_first=True, title="Social")
	plt.show()



if __name__ == "__main__":

		# Parse command line options
		from optparse import OptionParser

		parser = OptionParser()
		parser.add_option("-n", "--novel-path", dest="novel_path", default=None,
                      help="Specify the path to the folder containing the cohort of novel object trials (or test condition #1).")
		parser.add_option("-s", "--social-path", dest="social_path", default=None,
                      help="Specify the path to the folder containing the cohort of social interaction trials (or test condition #2).")
		parser.add_option("-o", "--output-path", dest="output_path", default=None,
                      help="Specify the ouput path.")

		(options, args) = parser.parse_args()

		test_PairAnalyze(options)