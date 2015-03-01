#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	email_process			as	ep
from	pandas				import	DataFrame
from	sklearn.naive_bayes		import	MultinomialNB
from	sklearn.naive_bayes		import	BernoulliNB
from	sklearn.metrics			import	accuracy_score
from	sklearn.feature_selection	import	SelectKBest
from	sklearn.feature_selection	import	chi2

### For training datasets
nEmail_tr	= 45000
nWord		= 9579
emailClass_tr	= file('../HW1/Train/train_emails_classes_200.txt').readlines()
emailClass_tr	= np.array([str.split(emailClass_tr[i]) for i in range(nEmail_tr)]).squeeze()
emailClass_tr	= np.where(emailClass_tr=='NotSpam',0,1)
bow_tr		= ep.read_bagofwords_dat('../HW1/Train/train_emails_bag_of_words_200.dat',nEmail_tr)
bow_tr_occ	= np.sign(bow_tr)	# Change word counts to word occurence

vocal   	= file('../HW1/Train/train_emails_vocab_200.txt').readlines()
vocal		= np.array([str.split(vocal[i]) for i in range(nWord)]).squeeze()

### For test datasets
nEmail_te	= 5000
emailClass_te	= file('../HW1/Test/test_emails_classes_0.txt').readlines()
emailClass_te	= np.array([str.split(emailClass_te[i]) for i in range(nEmail_te)]).squeeze()
emailClass_te	= np.where(emailClass_te=='NotSpam',0,1)
bow_te		= ep.read_bagofwords_dat('../HW1/Test/test_emails_bag_of_words_0.dat',nEmail_te)
bow_te_occ	= np.sign(bow_te)	# Change word counts to word occurence

### Naive Bayes classifier for multinomial models
clf_Mul		= MultinomialNB()
clf_Mul.fit(bow_tr,emailClass_tr)
pre_te_Mul	= clf_Mul.predict(bow_te)
acu_MulNB	= clf_Mul.score(bow_te, emailClass_te)
print		acu_MulNB		

### Naive Bayes classifier for Bernoulli models
clf_Ber		= BernoulliNB()
clf_Ber.fit(bow_tr_occ,emailClass_tr)
pre_te_Ber	= clf_Ber.predict(bow_te_occ)
acu_BerNB	= clf_Ber.score(bow_te_occ, emailClass_te)
print		acu_BerNB		

### Get the feature index
X_new		= SelectKBest(chi2, k=200).fit(bow_tr, emailClass_tr).get_support(True)
print		X_new
