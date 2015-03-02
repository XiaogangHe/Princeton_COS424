#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	email_process			as	ep
import	classifier_Library		as	clfLib
import	time
from	pandas				import	DataFrame
from	sklearn.naive_bayes		import	MultinomialNB
from	sklearn.naive_bayes		import	BernoulliNB
from	sklearn.feature_selection	import	SelectKBest
from	sklearn.feature_selection	import	chi2

### Read datasets
nEmail_tr	= 45000
nEmail_te	= 5000
nWord		= 9579
nFeatures	= 200

bow_tr, emailClass_tr = clfLib.getTraining(nEmail_tr)
bow_te, emailClass_te = clfLib.getTesting(nEmail_te)

### Option for feature selection
feature_select	= True

print "Do feature selection: %s \n" % (feature_select)
if feature_select:
	kbest	= SelectKBest(chi2, k=nFeatures)
	bow_tr	= kbest.fit_transform(bow_tr, emailClass_tr)
	bow_te	= kbest.transform(bow_te)

### Naive Bayes classifier for multinomial models
print "Classifier: MultimomialNB" 
time_start	= time.time()
clf_Mul		= MultinomialNB()
clf_Mul.fit(bow_tr,emailClass_tr)
time_tr 	= time.time()
print "Training time: %f" % (time_tr - time_start)
pre_te_Mul	= clf_Mul.predict(bow_te)
time_te 	= time.time()
print "Testing time: %f" % (time_te - time_tr)

acu_MulNB_tr	= clf_Mul.score(bow_tr, emailClass_tr)
acu_MulNB_te	= clf_Mul.score(bow_te, emailClass_te)
print "Accuracy on training data: %0.3f" % (acu_MulNB_tr)
print "Accuracy on test data: %0.3f \n" % (acu_MulNB_te)

### Naive Bayes classifier for Bernoulli models
print "Classifier: BernoulliNB" 
bow_tr_occ	= np.sign(bow_tr)	# Change word counts to word occurence
bow_te_occ	= np.sign(bow_te)	# Change word counts to word occurence
time_start	= time.time()
clf_Ber		= BernoulliNB()
clf_Ber.fit(bow_tr_occ,emailClass_tr)
time_tr 	= time.time()
print "Training time: %f" % (time_tr - time_start)
pre_te_Ber	= clf_Ber.predict(bow_te_occ)
time_te 	= time.time()
print "Testing time: %f" % (time_te - time_tr)

acu_BerNB_tr	= clf_Ber.score(bow_tr_occ, emailClass_tr)
acu_BerNB_te	= clf_Ber.score(bow_te_occ, emailClass_te)
print "Accuracy on training data: %0.3f" % (acu_BerNB_tr)
print "Accuracy on test data: %0.3f \n" % (acu_BerNB_te)


