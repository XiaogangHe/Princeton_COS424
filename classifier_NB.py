#!/usr/bin/env python

import	numpy			as	np
import	matplotlib.pyplot	as	plt
from	sklearn.naive_bayes	import	MultinomialNB
import	email_process		as	ep
from	sklearn.metrics		import	accuracy_score
from	sklearn.feature_selection import SelectKBest
from	sklearn.feature_selection import chi2

### For training datasets
nEmail_tr	= 45000
emailClass_tr	= file('../HW1/Train/train_emails_classes_200.txt').readlines()
emailClass_tr	= np.array([str.split(emailClass_tr[i]) for i in range(nEmail_tr)]).squeeze()
emailClass_tr	= np.where(emailClass_tr=='NotSpam',0,1)
bow_tr		= ep.read_bagofwords_dat('../HW1/Train/train_emails_bag_of_words_200.dat',nEmail_tr)

### Use the Naive Bayes classifier for multinomial models
nEmail_te	= 5000
emailClass_te	= file('../HW1/Test/test_emails_classes_0.txt').readlines()
emailClass_te	= np.array([str.split(emailClass_te[i]) for i in range(nEmail_te)]).squeeze()
emailClass_te	= np.where(emailClass_te=='NotSpam',0,1)
clf		= MultinomialNB()
clf.fit(bow_tr,emailClass_tr)
bow_te		= ep.read_bagofwords_dat('../HW1/Test/test_emails_bag_of_words_0.dat',nEmail_te)
pre_te		= clf.predict(bow_te)

NB_acu		= accuracy_score(pre_te, emailClass_te)
print		NB_acu		

###
X_new		= SelectKBest(chi2, k=200).fit(bow_tr, emailClass_tr).get_support(True)
print		X_new
