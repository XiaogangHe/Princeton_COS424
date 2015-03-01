#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	email_process			as	ep
from	pandas				import	DataFrame
from	sklearn.ensemble		import	RandomForestClassifier

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

### Random Forests classifier 
# Select the top K most important features
nFea		= 10
forest		= RandomForestClassifier(n_estimators=20,random_state=0)
forest.fit(bow_tr, emailClass_tr)
importances	= forest.feature_importances_
std 		= np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices		= np.argsort(importances)[::-1]
print		vocal[indices[:nFea]]

bow_tr_impFea	= bow_tr[:,indices[:nFea]]
bow_te_impFea	= bow_te[:,indices[:nFea]]
forest.fit(bow_tr_impFea, emailClass_tr)
forest.predict(bow_te_impFea)
print		forest.score(bow_te_impFea, emailClass_te)
