#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	email_process			as	ep
from	pandas				import	DataFrame
from	sklearn.ensemble		import	RandomForestClassifier
from    sklearn.feature_selection       import  SelectKBest
from    sklearn.feature_selection       import  chi2

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

