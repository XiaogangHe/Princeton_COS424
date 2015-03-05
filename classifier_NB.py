#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	email_process			as	ep
import	classifier_Library		as	clfLib
import	time
import	os
from	pandas				import	DataFrame
from	sklearn.naive_bayes		import	MultinomialNB
from	sklearn.naive_bayes		import	BernoulliNB
from	sklearn.feature_selection	import	SelectKBest
from	sklearn.feature_selection	import	chi2
from	sklearn.metrics			import	classification_report
from 	sklearn.metrics			import	recall_score
from	sklearn.metrics			import	roc_curve, auc

### Read datasets
nEmail_tr	= 45000
nEmail_te	= 5000
nWord		= 9579
nFeatures	= 1000

bow_tr, emailClass_tr = clfLib.getTraining(nEmail_tr)
bow_te, emailClass_te = clfLib.getTesting(nEmail_te)

### Option for feature selection
feature_select	= False

print "Do feature selection: %s (%s) \n" % (feature_select, nFeatures)
if feature_select:
	kbest	= SelectKBest(chi2, k=nFeatures)
	bow_tr	= kbest.fit_transform(bow_tr, emailClass_tr)
	bow_te	= kbest.transform(bow_te)

### Naive Bayes classifier for multinomial models
time_start	= time.time()
clf_Mul		= MultinomialNB()
clf_Mul.fit(bow_tr,emailClass_tr)
time_tr 	= time.time()
pre_te_Mul	= clf_Mul.predict(bow_te)
time_te 	= time.time()

acu_MulNB_tr	= clf_Mul.score(bow_tr, emailClass_tr)
acu_MulNB_te	= clf_Mul.score(bow_te, emailClass_te)
probas_		= clf_Mul.fit(bow_tr, emailClass_tr).predict_proba(bow_te)
fpr, tpr, thresholds		= roc_curve(emailClass_te, probas_[:,1])
roc_auc		= auc(fpr, tpr)
recall_nospam, recall_spam	= recall_score(emailClass_te, pre_te_Mul, average=None)
recall		= recall_spam
fall_out	= 1 - recall_nospam
label		= ['NotSpam', 'Spam']
report		= classification_report(emailClass_te, pre_te_Mul, target_names=label)
np.savez('../HW1/Results/Metric_MNB_%s_%s' % (feature_select, nFeatures), recall, fall_out, fpr, tpr, roc_auc, thresholds)
clfLib.calibration_plot(clf_Mul.fit(bow_tr,emailClass_tr), bow_te, emailClass_te)
os.system('mv ../HW1/Figures/calibration_plot.pdf ../HW1/Figures/calibration_plot_MNB_%s_%s.pdf' % (feature_select, nFeatures))

print "Classifier: MultimomialNB \n" 
print "Training time: %f" % (time_tr - time_start)
print "Testing time: %f" % (time_te - time_tr)
print "Accuracy on training data: %0.3f" % (acu_MulNB_tr)
print "Accuracy on test data: %0.3f \n" % (acu_MulNB_te)
print report

### Naive Bayes classifier for Bernoulli models
bow_tr_occ	= np.sign(bow_tr)	# Change word counts to word occurence
bow_te_occ	= np.sign(bow_te)	# Change word counts to word occurence
time_start	= time.time()
clf_Ber		= BernoulliNB()
clf_Ber.fit(bow_tr_occ,emailClass_tr)
time_tr 	= time.time()
pre_te_Ber	= clf_Ber.predict(bow_te_occ)
time_te 	= time.time()

acu_BerNB_tr	= clf_Ber.score(bow_tr_occ, emailClass_tr)
acu_BerNB_te	= clf_Ber.score(bow_te_occ, emailClass_te)
probas_		= clf_Ber.fit(bow_tr_occ, emailClass_tr).predict_proba(bow_te)
fpr, tpr, thresholds		= roc_curve(emailClass_te, probas_[:,1])
roc_auc		= auc(fpr, tpr)
recall_nospam, recall_spam	= recall_score(emailClass_te, pre_te_Ber, average=None)
recall		= recall_spam
fall_out	= 1 - recall_nospam
report		= classification_report(emailClass_te, pre_te_Ber, target_names=label)
np.savez('../HW1/Results/Metric_BNB_%s_%s' % (feature_select, nFeatures), recall, fall_out, fpr, tpr, roc_auc, thresholds)
clfLib.calibration_plot(clf_Ber.fit(bow_tr,emailClass_tr), bow_te, emailClass_te)
os.system('mv ../HW1/Figures/calibration_plot.pdf ../HW1/Figures/calibration_plot_BNB_%s_%s.pdf' % (feature_select, nFeatures))

print "Classifier: BernoulliNB \n" 
print "Training time: %f" % (time_tr - time_start)
print "Testing time: %f" % (time_te - time_tr)
print "Accuracy on training data: %0.3f" % (acu_BerNB_tr)
print "Accuracy on test data: %0.3f \n" % (acu_BerNB_te)
print report


