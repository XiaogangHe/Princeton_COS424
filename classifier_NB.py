#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	email_process			as	ep
import	classifier_Library		as	clfLib
from	pandas				import	DataFrame
from	sklearn.naive_bayes		import	MultinomialNB
from	sklearn.naive_bayes		import	BernoulliNB
from	sklearn.feature_selection	import	SelectKBest
from	sklearn.feature_selection	import	chi2

### For training datasets
nEmail_tr	= 45000
nWord		= 9579
emailClass_tr	= file('../HW1/Train/train_emails_classes_200.txt').readlines()
emailClass_tr	= np.array([str.split(emailClass_tr[i]) for i in range(nEmail_tr)]).squeeze()
emailClass_tr	= np.where(emailClass_tr=='NotSpam',0,1)
bow_tr		= ep.read_bagofwords_dat('../HW1/Train/train_emails_bag_of_words_200.dat',nEmail_tr)

vocal   	= file('../HW1/Train/train_emails_vocab_200.txt').readlines()
vocal		= np.array([str.split(vocal[i]) for i in range(nWord)]).squeeze()

### For test datasets
nEmail_te	= 5000
emailClass_te	= file('../HW1/Test/test_emails_classes_0.txt').readlines()
emailClass_te	= np.array([str.split(emailClass_te[i]) for i in range(nEmail_te)]).squeeze()
emailClass_te	= np.where(emailClass_te=='NotSpam',0,1)
bow_te		= ep.read_bagofwords_dat('../HW1/Test/test_emails_bag_of_words_0.dat',nEmail_te)

### Option for feature selection
nFeatures	= 200
feature_select	= False

if feature_select:
	kbest	= SelectKBest(chi2, k=nFeatures)
	bow_tr	= kbest.fit_transform(bow_tr, emailClass_tr)
	bow_te	= kbest.transform(bow_te)

print	bow_te.shape
print	bow_tr.shape

### Naive Bayes classifier for multinomial models
clf_Mul		= MultinomialNB()
clf_Mul.fit(bow_tr,emailClass_tr)
pre_te_Mul	= clf_Mul.predict(bow_te)
acu_MulNB	= clf_Mul.score(bow_te, emailClass_te)
print		acu_MulNB		

### Naive Bayes classifier for Bernoulli models
bow_tr_occ	= np.sign(bow_tr)	# Change word counts to word occurence
bow_te_occ	= np.sign(bow_te)	# Change word counts to word occurence
clf_Ber		= BernoulliNB()
clf_Ber.fit(bow_tr_occ,emailClass_tr)
pre_te_Ber	= clf_Ber.predict(bow_te_occ)
acu_BerNB	= clf_Ber.score(bow_te_occ, emailClass_te)
print		acu_BerNB		



