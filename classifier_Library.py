#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	email_process			as	ep
from	pandas				import	DataFrame
from	sklearn.ensemble		import	RandomForestClassifier
from    sklearn.feature_selection       import  SelectKBest
from    sklearn.feature_selection       import  chi2

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

