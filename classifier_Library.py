#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	email_process			as	ep
import	pandas				as	pd
from	pandas				import	DataFrame
from	sklearn.ensemble		import	RandomForestClassifier
from    sklearn.feature_selection       import  SelectKBest
from    sklearn.feature_selection       import  chi2
from	matplotlib 			import	rcParams

dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
		(0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
		(0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
		(0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
		(0.4, 0.6509803921568628, 0.11764705882352941),
		(0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
		(0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
		(0.4, 0.4, 0.4)]

rcParams['figure.figsize'] = (10, 6)
rcParams['axes.color_cycle'] = dark2_colors
rcParams['figure.dpi'] = 150
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = False
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'none'

"""
Read dataset
"""

def	getTraining(nEmail_tr):
	bow_tr          = ep.read_bagofwords_dat('../HW1/Train/train_emails_bag_of_words_200.dat',nEmail_tr)
	emailClass_tr   = file('../HW1/Train/train_emails_classes_200.txt').readlines()
	emailClass_tr   = np.array([str.split(emailClass_tr[i]) for i in range(nEmail_tr)]).squeeze()
	emailClass_tr   = np.where(emailClass_tr=='NotSpam',0,1)
	return		bow_tr, emailClass_tr

def	getTesting(nEmail_te):
	bow_te          = ep.read_bagofwords_dat('../HW1/Test/test_emails_bag_of_words_0.dat',nEmail_te)
	emailClass_te   = file('../HW1/Test/test_emails_classes_0.txt').readlines()
	emailClass_te   = np.array([str.split(emailClass_te[i]) for i in range(nEmail_te)]).squeeze()
	emailClass_te   = np.where(emailClass_te=='NotSpam',0,1)
	return		bow_te, emailClass_te

def	getVocabulary(nWord):
	vocal           = file('../HW1/Train/train_emails_vocab_200.txt').readlines()
	vocal           = np.array([str.split(vocal[i]) for i in range(nWord)]).squeeze()
	return		vocal

"""
Feature selection
"""

def	selectFea(X, y, nFeatures):
	select_feature_ind	= SelectKBest(chi2, k=nFeatures).fit(X, y).get_support(True)
	return select_feature_ind

"""
Get the new bow (bag of words) matrix according to the selected features
"""
def	getSelectBOW(data, ind):
	return	data[:,ind]

"""
Function
--------
calibration_plot

Builds a plot like the one above, from a classifier and review data

Inputs
-------
clf :	Classifier object
A MultinomialNB classifier
X :	(Nexample, Nfeature) array
	The bag-of-words data
Y :	(Nexample) integer array
1 if a review is Fresh
"""

def calibration_plot(clf, xtest, ytest):
	prob = clf.predict_proba(xtest)[:, 1]
	outcome = ytest
	data = pd.DataFrame(dict(prob=prob, outcome=outcome))
	
	#group outcomes into bins of similar probability
	bins = np.linspace(0, 1, 20)
	cuts = pd.cut(prob, bins)
	binwidth = bins[1] - bins[0]
	#freshness ratio and number of examples in each bin
	cal = data.groupby(cuts).outcome.agg(['mean', 'count'])
	cal['pmid'] = (bins[:-1] + bins[1:]) / 2
	cal['sig'] = np.sqrt(cal.pmid * (1 - cal.pmid) / cal['count'])
	#the calibration plot
	ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
	p = plt.errorbar(cal.pmid, cal['mean'], cal['sig'])
	plt.plot(cal.pmid, cal.pmid, linestyle='--', lw=1, color='k')
	plt.ylabel("Empirical P (Spam)")
	remove_border(ax)
	#the distribution of P(fresh)
	ax = plt.subplot2grid((3, 1), (2, 0), sharex=ax)
	plt.bar(left=cal.pmid - binwidth / 2, height=cal['count'], width=.95 * (bins[1] - bins[0]),
			fc=p[0].get_color())
	plt.xlabel("Predicted P (Spam)")
	remove_border()
	plt.ylabel("Number")
	plt.savefig('../HW1/Figures/calibration_plot.pdf', format='PDF')


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
	
	"""
	Minimize chartjunk by stripping out unnecesary plot borders and axis ticks
	The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
	"""
	
	ax = axes or plt.gca()
	ax.spines['top'].set_visible(top)
	ax.spines['right'].set_visible(right)
	ax.spines['left'].set_visible(left)
	ax.spines['bottom'].set_visible(bottom)
	
	#turn off all ticks
	ax.yaxis.set_ticks_position('none')
	ax.xaxis.set_ticks_position('none')
	#now re-enable visibles
	if top:
		ax.xaxis.tick_top()
	if bottom:
		ax.xaxis.tick_bottom()
	if left:
		ax.yaxis.tick_left()
	if right:
		ax.yaxis.tick_right()
