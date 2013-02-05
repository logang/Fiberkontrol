import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scipy as sp

"""
This class is for comparing the effect two different trial conditions (i.e. novel object interaction vs. social interaction)
across a cohort of mice.
The score used to measure the effect of each condition is determined by which  are passed in to the class.
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
				self.p_path = options.p_path

		def Load( self ):

			self.n_scores = [] #An array where each entry is an array holding the scores for all events of a single mouse's trial
			self.s_scores = []
			self.n_start_times = [] #An array where each entry is an array holding the start time of all events of a single mouse's trial
			self.s_start_times = []
			self.n_end_times = [] # An array where each entry is an array holding the end time of all events of a single mouse's trial
			self.s_end_times = []
			self.n_windows = [] # An array where each entry is the exact length (in s) of the window used to calculate the score (i.e. area under curve)
			self.s_windows = []
			self.n_length_times = []
			self.s_length_times = []

			self.n_aux = [] #In case an auxiliary variable is provided
			self.s_aux = []
			self.aux_included = True


			if self.p_path is not None:
				self.p_files = []
				p_listing = os.listdir(self.p_path)
				for infile in p_listing:
					try:
						self.p_files.append(np.load(self.p_path + '/' + infile))
						print infile
					except:
						print "FILE DOESN'T WORK: ", infile


				for f in self.p_files:
					print "len(f['before']", len(f['before'])
					self.n_scores.append(f['before'][0:min(8, len(f['before']) -1 )])
					self.s_scores.append(f['after'][0:min(8, len(f['after']) -1 )])

					#self.n_start_times.append(f['event_times'])
					#self.n_end_times.append(f['end_times'])
					#self.n_windows.append(f['window_size'])
					#self.n_length_times.append(np.array(f['end_times']) - np.array(f['event_times']))
					# try:
					# 	self.n_aux.append(f['aux'])

					# self.s_start_times.append(f['event_times'])
					# self.s_end_times.append(f['end_times'])
					# self.s_windows.append(f['window_size'])
					# self.s_length_times.append(np.array(f['end_times']) - np.array(f['event_times']))
					# try:
					# 	self.s_aux.append(f['aux'])
					# except:
					# 	aux_included = False

			elif self.n_path is not None:
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

				for f in self.n_files:
					self.n_scores.append(f['scores'])
					self.n_start_times.append(f['event_times'])
					self.n_end_times.append(f['end_times'])
					self.n_windows.append(f['window_size'])
					self.n_length_times.append(np.array(f['end_times']) - np.array(f['event_times']))
					try:
						self.n_aux.append(f['aux'])
					except:
						aux_included = False

				for f in self.s_files:
					self.s_scores.append(f['scores'])
					self.s_start_times.append(f['event_times'])
					self.s_end_times.append(f['end_times'])
					self.s_windows.append(f['window_size'])
					self.s_length_times.append(np.array(f['end_times']) - np.array(f['event_times']))
					try:
						self.s_aux.append(f['aux'])
					except:
						aux_included = False


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
				if np.shape(f) != ():
					n_score_list.append(f[index]) 
				else:
					n_score_list.append(f)

			for f in self.s_scores:
				if np.shape(f) != ():
					s_score_list.append(f[index]) 
				else:
					s_score_list.append(f) 

			print "  n      s   "
			for i in range(len(n_score_list)):
				print "{} \t {}".format(str(n_score_list[i]), str(s_score_list[i]))

			if test == "ttest":
				[tvalue, pvalue] = stats.ttest_rel(s_score_list, n_score_list)
				print "normalized area tvalue: ", tvalue, " normalized area pvalue: ", pvalue
			if test == "wilcoxon":
				[zstatistic, pvalue] = stats.wilcoxon(s_score_list, n_score_list)
				print "normalized area zstatistic: ", zstatistic, " normalized area pvalue: ", pvalue



		def EventLength_vs_Score( self, scores, start_times, end_times, length_times, just_first=False, title=None, out_path=None ):
			"""
			Produces a scatter plot of the length of each event vs the score (i.e. area under the curve) 
			associated with that event.

			just_first parameter indicates whether to only include the first interaction event of a trial
			"""

			event_starts = []
			event_ends = []
			event_scores = []
			event_lengths = []

			trial_labels = []
			time_from_first_event = []

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

			for i in range(len(start_times)):
				f = start_times[i]
				if np.shape(f) != ():
					if(just_first):
						trial_labels.append(i)
					else:
						for j in range(len(f)):
							trial_labels.append(i)
				else:
					trial_labels.append(i)

			for i in range(len(start_times)):
				f = start_times[i]
				if np.shape(f) != ():
					if(just_first):
						time_from_first_event.append(0)
					else:
						for j in range(len(f)):
							time_from_first_event.append(f[j] - f[0])
				else:
					time_from_first_event.append(0) 




			event_lengths = np.array(event_ends) - np.array(event_starts)
			event_lengths = event_lengths[event_lengths>=0]
			event_lengths = event_lengths[event_lengths<10]
			event_scores = [event_scores[i] for i in range(len(event_lengths)) if event_lengths[i]>=0]
			#event_scores = event_scores[event_lengths>=0]

			print "median length", np.median(event_lengths)
			print "stdev length", np.sqrt(np.var(event_lengths))
			print "mean length", np.mean(event_lengths)

			A = np.array([event_lengths, np.ones(len(event_lengths))])
			w = np.linalg.lstsq(A.T, event_scores)[0]

			plt.figure()
			plt.plot(event_lengths, w[0]*event_lengths + w[1], 'r-', event_lengths, event_scores, 'o')
			plt.ylim([min(0, np.min(event_scores)), 1.0])
			plt.xlabel('Interaction Time [s]')
			plt.ylabel('Sharpness of first peak: ' r'$\frac{\sum\delta F/F}{\max(peak)}}$ over window of 1s')
			if title is not None:
				plt.title(title)
			if out_path is not None:
				plt.savefig(out_path + title + "_duration_vs_score.pdf")
			else:
				plt.show()


			plt.figure()
			h = plt.hist(event_lengths, bins=np.max(event_lengths)*20)
			print h
			plt.xlabel('Interaction Time [s]')
			plt.ylabel('Number of interaction events')
			plt.text(2, np.max(h[0]) - 5, 'mean=' + "{0:.2f}".format(np.mean(event_lengths)) + ' stdev=' + "{0:.2f}".format(np.sqrt(np.var(event_lengths))))
			plt.xlim([0, 6])
			if title is not None:
				plt.title(title)
			if out_path is not None:
				plt.savefig(out_path + title + "_event_duration_hist.pdf")
			else:
				plt.show()

			event_intervals = np.array(event_starts[1:]) - np.array(event_starts[0:-1])

			time_from_first_event = [time_from_first_event[i] for i in range(len(event_intervals)) if event_intervals[i]>0]
			trial_labels = [trial_labels[i] for i in range(len(event_intervals)) if event_intervals[i]>0]

			event_intervals = event_intervals[event_intervals > 0]
			print len(time_from_first_event), len(event_intervals)
			plt.figure()
			p = plt.scatter(np.log(time_from_first_event), np.log(np.array(event_intervals)), c=trial_labels)
			plt.xlabel('log time of event t since first event in a trial [s]')
			plt.ylabel('log interval between event t and t+1 [s]')
			if title is not None:
				plt.title(title)
			plt.show()



			# print plt.cm.jet(np.arange(len(length_times)))
			# print length_times
			# colors = ['r', 'g', 'b', 'k', 'y', 'purple', 'orange', 'salmon', 'cyan', 'magenta', 'burlywood', 'chartreuse']

			# plt.figure()
			# plt.hist(length_times, bins=np.max(np.max(length_times))*20, histtype='stepfilled', stacked=True, fill=True)
			# plt.xlabel('Interaction Time [s]')
			# plt.ylabel('Number of interaction events')
			# plt.xlim([0, 6])
			# if title is not None:
			# 	plt.title(title)


		def eNegX(self, p, x):
			x0, y0, c, k=p
			#Set c=1 to normalize all of the trials, since we
			# are only interested in the rate of decay
			y = (1 * np.exp(-k*(x-x0))) + y0
			return y

		def eNegX_residuals(self, p, x, y):
			return y - self.eNegX(p, x)

		def fit_exponential(self, x, y):
			# Because we are optimizing over a nonlinear function
			# choose a number of possible starting values of (x0, y0, c, k)
			# and use the results from whichever produces the smallest 
			# residual
			kguess = [0, 0.1, 0.5, 1.0, 100, 500, 1000]
			max_r2 = -1
			maxvalues = ()
			for kg in kguess:
				p_guess=(np.min(x), 0, 1, kg)
				p, cov, infodict, mesg, ier = sp.optimize.leastsq(
					self.eNegX_residuals, p_guess, args=(x, y), full_output=1)

				x0,y0,c,k=p 
				# print('''Reference data:\  
				# 			x0 = {x0}
				# 			y0 = {y0}
				# 			c = {c}
				# 			k = {k}
				# 			'''.format(x0=x0,y0=y0,c=c,k=k))

				numPoints = np.floor((np.max(x) - np.min(x))*100)
				xp = np.linspace(np.min(x), np.max(x), numPoints)
				#pxp = np.exp(-1*xp)
				pxp = self.eNegX(p, xp)
				yxp = self.eNegX(p, x)

				sstot = np.sum(np.multiply(y - np.mean(y), y - np.mean(y)))
				sserr = np.sum(np.multiply(y - yxp, y - yxp))
				r2 = 1 - sserr/sstot
				if max_r2 == -1:
					maxvalues = (xp, pxp, x0, y0, c, k, r2)
				if r2 > max_r2:
						max_r2 = r2
						maxvalues = (xp, pxp, x0, y0, c, k, r2)

			return maxvalues



		def GroupScoresByBout(self, scores, start_times=None, index=-1):
			"""
			Input: scores = [[scores from trial 1], [scores from trial 2], [scores from trial 3]...]
						 start_times = [[event times from trial 1], [event times from trial 2], [event times from trial 3]...]
						 index: if you wish to use only a single bout number of the specified index

			Returns: 
						 scores_per_bout = [[scores from the first bout across trials], [scores from the second bout across trials]...]
						 all_bouts = [scores from first trial, scores from second trial, scores from third trial] (all in on array)
						 all_bout_indices = [the indices to match the above array]
						 all_bout_times = [the times to match the data points in all_bouts]
						 all_bout_labels = [the trial number of each data point in all_bouts]
			 """

			scores_per_bout = []
			all_bouts = []
			all_bout_indices = []
			all_bout_times = []
			all_bout_labels = []

			for j in range(len(scores)):
				f = scores[j]
				if start_times is not None:
					t = start_times[j]
				if(index != -1):
					event_scores.append(f[index])
				else:
					for i in range(len(f)):
						all_bouts.append(f[i])
						all_bout_indices.append(i+1)
						if start_times is not None:
							all_bout_times.append(t[i] - t[0])
						all_bout_labels.append(j)
						if i>=len(scores_per_bout):
							scores_per_bout.append([f[i]])
						else:
							scores_per_bout[i].append(f[i])

			return (scores_per_bout, all_bouts, all_bout_indices, all_bout_times, all_bout_labels)


		def SummarizeBouts(self, scores_per_bout):

			bout_avgs = np.zeros(len(scores_per_bout))
			bout_stderr = np.zeros(len(scores_per_bout))
			for i in range(len(scores_per_bout)):
				bout_avgs[i] = np.mean(scores_per_bout[i])
				bout_stderr[i] = np.sqrt(np.var(scores_per_bout[i])/len(scores_per_bout[i]))

			return (bout_avgs, bout_stderr)

		def AverageScores(self, scores, start_times, title=None, score_type=None, index=-1, out_path=None, bucket='number'):
			"""
			Combines interaction bouts across multiple trials based on
			which number bout it is within a trial.
			If bucket='number', simply use the bout number.
			Plots the average score for each bout.
			"""

			scores_per_bout, all_bouts, all_bout_indices, all_bout_times, all_bout_labels = self.GroupScoresByBout(scores, start_times, index)

			bout_avgs, bout_stderr = self.SummarizeBouts(scores_per_bout)

			bout_numbers = range(1, len(bout_avgs) +1)
			fig = plt.figure()	
			ax = fig.add_subplot(111)
			#ax.plot(bout_numbers, bout_avgs, 'o')
			ax.errorbar(bout_numbers, bout_avgs, yerr=1.96*bout_stderr, fmt='o')
		#	ax.plot(all_bout_indices, all_bouts, 'o', alpha=0.3)
			plt.xlabel('Bout number')
			plt.ylabel('Average ' + score_type + ' across all trials [dF/F]')
			ax.set_ylim([min(0, np.min(bout_avgs)), 1])
			if title is not None:
				plt.title(title)

			try:
				text_x = np.min(bout_numbers) + 3
				text_y = np.max(bout_avgs) 
				xp, pxp, x0, y0, c, k, r2 = self.fit_exponential(bout_numbers, bout_avgs + 1)
				ax.plot(xp, pxp-1)
				ax.text(min(200, text_x), text_y + 0.25, "y = c*exp(-k*(x-x0)) + y0")
				ax.text(min(200, text_x), text_y + 0.20, "k = " + "{0:.2f}".format(k) + ", c = " + "{0:.2f}".format(c) + 
																				", x0 = " + "{0:.2f}".format(x0) + ", y0 = " + "{0:.2f}".format(y0) )
				ax.text(min(200, text_x), text_y + 0.15, "r^2 = " + str(r2))
			except:
					print "Exponential Curve fit did not work"

			if out_path is not None:
				plt.savefig(out_path + title + "_" + score_type + "_average_vs_bout_number.pdf")
			else:
				plt.show()

			fig = plt.figure()	
			ax = fig.add_subplot(111)
			ax.scatter(all_bout_times, all_bouts, c=all_bout_labels, linewidth=.1, cmap=plt.cm.jet, marker='o', alpha=0.8)
			plt.xlabel('Time since first bout [s]')
			plt.ylabel('Fluorescence (' + score_type + ') [dF/F]')
			ax.set_ylim([min(0, np.min(all_bouts)), 1])
			ax.set_xlim([-20, np.max(all_bout_times)])
			if title is not None:
				plt.title(title)


			text_x = np.min(all_bout_times) + 100
			text_y = np.max(all_bouts) - 0.3
			xp, pxp, x0, y0, c, k, r2 = self.fit_exponential(all_bout_times, np.array(all_bouts) + 1)
			ax.plot(xp, pxp-1, linewidth=2.0, color='k')
			ax.text(min(200, text_x), text_y + 0.25, "y = c*exp(-k*(x-x0)) + y0")
			ax.text(min(200, text_x), text_y + 0.20, "k = " + "{0:.2f}".format(k) + ", c = " + "{0:.2f}".format(c) + 
																			", x0 = " + "{0:.2f}".format(x0) + ", y0 = " + "{0:.2f}".format(y0) )
			ax.text(min(200, text_x), text_y + 0.15, "r^2 = " + str(r2))

			if out_path is not None:
				plt.savefig(out_path + title + "_" + score_type + "_vs_time.pdf")
			else:
				plt.show()



				#This does not currently work
			def CompareAllPeaks(self):
				if (self.peaks_included):

					n_avg_peaks = []
					s_avg_peaks = []

					s_counts = []
					n_counts = []

					for peak in self.s_peaks:
						for i in range(len(peak)):
							if i >= len(s_avg_peaks):
								s_avg_peaks.append(peak[i])
								s_count.append(1)
							else:
								s_avg_peaks[i] = s_avg_peaks[i] + peak[i]
								s_counts[i] = s_counts[i] + 1

					plt.plot(np.range(len(s_avg_peaks)), s_avg_peaks)


		def ScoreHistogram(self, scores, title=None, score_type=None, out_path=None):
			print scores
			all_scores = []
			for f in scores:
				if np.shape(f) != ():
					for d in f:
						all_scores.append(d) 
				else:
					all_scores.append(f)


			plt.figure()
			h = plt.hist(all_scores, bins=(np.max(all_scores) - np.min(all_scores))*30)
			plt.xlabel("Fluorescence (" + score_type + ") [dF/F]")
			plt.ylabel('Number of occurences')
			plt.text(2, np.max(h[0]) - 5, 'mean=' + "{0:.2f}".format(np.mean(all_scores)) + ' stdev=' + "{0:.2f}".format(np.sqrt(np.var(all_scores))))
			plt.xlim([min(-.4, np.min(all_scores)), max(1, np.max(all_scores))])
			if title is not None:
				plt.title(title)
			if out_path is not None:
				plt.savefig(out_path + title + "_" + score_type + "_hist.pdf")
			else:
				plt.show()			


		def CombinedAverage(self, n_scores, n_start_times, s_scores=None, s_start_times=None, 
										titlen=None, titles=None, score_type=None, out_path=None):

			n_scores_per_bout, n_all_bouts, n_all_bout_indices, n_all_bout_times, n_all_bout_labels= self.GroupScoresByBout(n_scores, n_start_times)
			if s_scores is not None:
				s_scores_per_bout, s_all_bouts, s_all_bout_indices, s_all_bout_times, s_all_bout_labels= self.GroupScoresByBout(s_scores, s_start_times)


			n_bout_avgs, n_bout_stderr = self.SummarizeBouts(n_scores_per_bout)
			if s_scores is not None:
				s_bout_avgs, s_bout_stderr = self.SummarizeBouts(s_scores_per_bout)


			n_bout_numbers = range(1, len(n_bout_avgs) +1)
			if s_scores is not None:
				s_bout_numbers = range(1, len(s_bout_avgs) +1)


			print n_scores_per_bout
			print n_bout_avgs
			print np.max(n_bout_avgs)

			##PLOTTING##
			fig = plt.figure()	
			ax = fig.add_subplot(111)
			ax.errorbar(n_bout_numbers, n_bout_avgs, yerr=1.96*n_bout_stderr, fmt='ob')
			if s_scores is not None:
				ax.errorbar(s_bout_numbers, s_bout_avgs, yerr=1.96*s_bout_stderr, fmt='^g')
			plt.xlabel('Bout number')
			plt.ylabel('Average ' + score_type + ' across all trials [dF/F]')
			if s_scores is not None:
				ax.set_ylim([min(np.min(s_bout_avgs), min(0, np.min(n_bout_avgs))), max(np.max(n_bout_avgs), np.max(s_bout_avgs)*(1.5))])
				ax.set_xlim([min(np.min(s_bout_numbers), min(0, np.min(n_bout_numbers))), max(np.max(n_bout_numbers), np.max(s_bout_numbers)*(1.1))])

			else:
				ax.set_ylim([min(0, np.min(n_bout_avgs)), np.max(n_bout_avgs)*2])
				ax.set_xlim([min(0, np.min(n_bout_numbers)), np.max(n_bout_numbers)*(1.1)])

			if titlen is not None and titles is not None:
				plt.title("")

			n_xp, n_pxp, n_x0, n_y0, n_c, n_k, n_r2 = self.fit_exponential(n_bout_numbers, n_bout_avgs + 1)
			if s_scores is not None:
				s_xp, s_pxp, s_x0, s_y0, s_c, s_k, s_r2 = self.fit_exponential(s_bout_numbers, s_bout_avgs + 1)
				text_x = max(np.min(s_bout_numbers), np.min(n_bout_numbers) + 3)
				text_y = max(np.max(n_bout_avgs), np.max(s_bout_avgs))
			else:
				text_x = np.min(n_bout_numbers) + 3
				text_y = np.max(n_bout_avgs)	
			nplot, = plt.plot(n_xp, n_pxp-1, 'b', linewidth=2)
			if s_scores is not None:
				splot, = plt.plot(s_xp, s_pxp-1, 'g', linewidth=2)
				#plt.legend([nplot, splot], ["Novel object: decay rate = " + "{0:.2f}".format(n_k) + r", $r^2 = $" + "{0:.2f}".format(n_r2), 
				#												"Social interaction: decay rate = " + "{0:.2f}".format(s_k) + r", $r^2 = $" + "{0:.2f}".format(s_r2)])
				plt.legend([nplot, splot], ["Before: decay rate = " + "{0:.2f}".format(n_k) + r", $r^2 = $" + "{0:.2f}".format(n_r2), 
																"After: decay rate = " + "{0:.2f}".format(s_k) + r", $r^2 = $" + "{0:.2f}".format(s_r2)])

			else:
				plt.legend([nplot], ["Sucrose response: decay rate = " + "{0:.2f}".format(n_k) + r", $r^2 = $" + "{0:.2f}".format(n_r2)])
			# ax.text(min(200, text_x), text_y + 0.25, "y = c*exp(-k*(x-x0)) + y0")
			# ax.text(min(200, text_x), text_y + 0.20, "novel k = " + "{0:.2f}".format(n_k) + ", r^2 = " + "{0:.2f}".format(n_r2))
			# ax.text(min(200, text_x), text_y + 0.15, "social k = " + "{0:.2f}".format(s_k) + ", r^2 = " + "{0:.2f}".format(s_r2))
			# except:
			# 		print "Exponential Curve fit did not work"

			if out_path is not None:
				print "saving fig: ", out_path + titles + "_" + titlen + "combined_average_vs_bout_number.pdf"
				plt.savefig(out_path + titles + "_" + titlen + "combined_average_vs_bout_number.pdf")
			else:
				plt.show()
			pass



			##TIME SINCE FIRST EVENT PLOT, NOT FROM AVERAGES###
			fig = plt.figure()	
			ax = fig.add_subplot(111)
			print n_all_bout_times
			if n_all_bout_times != []:
				ax.scatter(n_all_bout_times, n_all_bouts, c=n_all_bout_labels, linewidth=.1, marker='o', alpha=0.8)
			if s_scores is not None and s_all_bout_times !=[]:
				ax.scatter(s_all_bout_times, s_all_bouts, c='g', linewidth=.1, marker='^', alpha=0.8)

				plt.xlabel('Time since first bout [s]')
				plt.ylabel(score_type + ' for every bout across all trials [dF/F]')
				if s_scores is not None:
					ax.set_ylim([min(0, max(np.min(n_all_bouts), np.min(s_all_bouts))), np.max(n_all_bouts)*(2)])
					ax.set_xlim([-.1*np.max(n_all_bout_times), max(np.max(n_all_bout_times), np.max(s_all_bout_times))*(1.1)])
				else:
					ax.set_ylim([min(0, np.min(n_all_bouts)), np.max(n_all_bouts)])
					ax.set_xlim([-.1*np.max(n_all_bout_times), np.max(n_all_bout_times)*1.1])

				if titlen is not None:
					plt.title("")

		#	if s_scores is not None:
		#		text_x = max(np.max(n_all_bout_times), np.max(s_all_bout_times)) + 100
		#		text_y = max(np.min(n_all_bouts), np.min(s_all_bouts)) - 0.3
		#	else:
				text_x = np.max(n_all_bout_times) + 100
				text_y = np.min(n_all_bouts) - 0.3

				n_xp, n_pxp, n_x0, n_y0, n_c, n_k, n_r2 = self.fit_exponential(n_all_bout_times, np.array(n_all_bouts) + 1)
				if s_scores is not None:
					s_xp, s_pxp, s_x0, s_y0, s_c, s_k, s_r2 = self.fit_exponential(s_all_bout_times, np.array(s_all_bouts) + 1)


				nplot, = ax.plot(n_xp, n_pxp-1, linewidth=2.0, color='b')
				if s_scores is not None:
					splot, = ax.plot(s_xp, s_pxp-1, linewidth=2.0, color='g')
					plt.legend([nplot, splot], ["Novel object: decay rate = " + "{0:.2f}".format(n_k) + r", $r^2 = $" + "{0:.2f}".format(n_r2), 
																	"Social interaction: decay rate = " + "{0:.2f}".format(s_k) + r", $r^2 = $" + "{0:.2f}".format(s_r2)])
				else:
					plt.legend([nplot], ["Sucrose response: decay rate = " + "{0:.4f}".format(n_k) + r", $r^2 = $" + "{0:.2f}".format(n_r2)])

				# ax.text(min(200, text_x), text_y + 0.25, "y = c*exp(-k*(x-x0)) + y0")
				# ax.text(min(200, text_x), text_y + 0.20, "k = " + "{0:.2f}".format(k) + ", c = " + "{0:.2f}".format(c) + 
				# 																", x0 = " + "{0:.2f}".format(x0) + ", y0 = " + "{0:.2f}".format(y0) )
				# ax.text(min(200, text_x), text_y + 0.15, "r^2 = " + str(r2))

				if out_path is not None:
					plt.savefig(out_path + titlen + "_" + titles + "_combined_" + score_type + "_vs_time.pdf")
				else:
					plt.show()



def test_PairAnalyze(options):
	"""
	Test the PairAnalyze class.
	"""

	PA = PairAnalyze( options )
	PA.Load()
#	PA.CompareAllPeaks()

#	PA.CompareEvent(0)
#	PA.CompareEvent(0, "wilcoxon" )
#	print "Novel interaction time:"
	#PA.EventLength_vs_Score(PA.n_scores, PA.n_start_times, PA.n_end_times, PA.n_length_times, just_first=False, title="Novel", out_path=PA.output_path)
#	print "Social interaction time:"
	#PA.EventLength_vs_Score(PA.s_scores, PA.s_start_times, PA.s_end_times, PA.s_length_times, just_first=False, title="Social", out_path=PA.output_path)

	#	PA.CombinedAverage(PA.n_scores, PA.n_start_times, PA.s_scores, PA.s_start_times, titlen='novel', titles='social', score_type='area under curve', out_path=PA.output_path + 'area_')
	#PA.CombinedAverage(PA.n_scores, PA.n_start_times, PA.s_scores, PA.s_start_times, titlen='novel', titles='social', score_type='peak height', out_path=PA.output_path + 'peak_')
##	PA.CombinedAverage(PA.n_scores, PA.n_start_times, None, None, titlen='novel', titles='social', score_type='peak height', out_path=PA.output_path + 'peak_')

	PA.CombinedAverage(PA.n_scores, None, PA.s_scores, None, titlen='Before', titles='After', score_type='value at 1s from event', out_path=PA.output_path + 'peak_')



#	PA.AverageScores(PA.n_scores, PA.n_start_times, title='Novel', score_type='area', out_path=PA.output_path)
#	PA.AverageScores(PA.s_scores, PA.s_start_times, title='Social', score_type='area', out_path=PA.output_path)
	#PA.ScoreHistogram(PA.n_scores, title='Novel', score_type='Peak', out_path=PA.output_path)
	#PA.ScoreHistogram(PA.s_scores, title='Social', score_type='Peak', out_path=PA.output_path)



if __name__ == "__main__":

		# Parse command line options
		from optparse import OptionParser

		parser = OptionParser()
		parser.add_option("-n", "--novel-path", dest="novel_path", default=None,
                      help="Specify the path to the folder containing the cohort of novel object trials (or test condition #1).")
		parser.add_option("-s", "--social-path", dest="social_path", default=None,
                      help="Specify the path to the folder containing the cohort of social interaction trials (or test condition #2).")
		parser.add_option("-p", "--pair-path", dest="p_path", default=None,
                      help="Specify the path to the folder containing the paired scores (such as 'before' and 'after' an event).")
		parser.add_option("-o", "--output-path", dest="output_path", default=None,
                      help="Specify the ouput path.")

		(options, args) = parser.parse_args()

		test_PairAnalyze(options)