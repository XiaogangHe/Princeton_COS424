#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	email_process			as	ep
import	classifier_Library		as	clfLib
import	time
import	os
from	pandas				import	DataFrame
from	sklearn				import	svm
from	sklearn.feature_selection	import	SelectKBest
from	sklearn.feature_selection	import	chi2
from	sklearn.metrics			import	classification_report
from 	sklearn.metrics			import	recall_score
from	sklearn.metrics			import	roc_curve, auc

### Read datasets
nEmail_tr	= 45000
nEmail_te	= 5000
nWord		= 9579
nFeatures	= 20

bow_tr, emailClass_tr = clfLib.getTraining(nEmail_tr)
bow_te, emailClass_te = clfLib.getTesting(nEmail_te)

### Option for feature selection
feature_select	= True

print "Do feature selection: %s (%s) \n" % (feature_select, nFeatures)
if feature_select:
	kbest	= SelectKBest(chi2, k=nFeatures)
	bow_tr	= kbest.fit_transform(bow_tr, emailClass_tr)
	bow_te	= kbest.transform(bow_te)

### Linear SVM
clf_LSVM	= svm.LinearSVC()
time_start	= time.time()
clfLSVMFit	= clf_LSVM.fit(bow_tr, emailClass_tr)
time_tr 	= time.time()
pre_te_LSVM	= clf_LSVM.predict(bow_te)
time_te 	= time.time()

acu_LSVM_tr	= clf_LSVM.score(bow_tr, emailClass_tr)
acu_LSVM_te	= clf_LSVM.score(bow_te, emailClass_te)
probas_		= clf_LSVM.decision_function(bow_te)
fpr, tpr, thresholds		= roc_curve(emailClass_te, probas_[:])
roc_auc		= auc(fpr, tpr)
recall_nospam, recall_spam	= recall_score(emailClass_te, pre_te_LSVM, average=None)
recall		= recall_spam
fall_out	= 1 - recall_nospam
label		= ['NotSpam', 'Spam']
report		= classification_report(emailClass_te, pre_te_LSVM, target_names=label)
np.savez('../HW1/Results/Metric_LSVM_%s_%s' % (feature_select, nFeatures), recall, fall_out, fpr, tpr, roc_auc, thresholds)
#clfLib.calibration_plot(clfLSVMFit, bow_te, emailClass_te)
#os.system('mv ../HW1/Figures/calibration_plot.pdf ../HW1/Figures/calibration_plot_LSVM_%s_%s.pdf' % (feature_select, nFeatures))

print "Classifier: Linear SVM \n" 
print "Training time: %f" % (time_tr - time_start)
print "Testing time: %f" % (time_te - time_tr)
print "Accuracy on training data: %0.3f" % (acu_LSVM_tr)
print "Accuracy on test data: %0.3f \n" % (acu_LSVM_te)
print report

