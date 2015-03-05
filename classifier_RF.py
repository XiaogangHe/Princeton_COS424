#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	email_process			as	ep
import	classifier_Library		as	clfLib
import	time
import	os
from	pandas				import	DataFrame
from	sklearn.ensemble		import	RandomForestClassifier
from	sklearn.metrics			import	classification_report
from 	sklearn.metrics			import	recall_score
from	sklearn.metrics			import	roc_curve, auc

### Read datasets
nEmail_tr	= 45000
nEmail_te	= 5000
nWord		= 9579
nTree		= 10
nFeatures	= 20

bow_tr, emailClass_tr	= clfLib.getTraining(nEmail_tr)
bow_te, emailClass_te	= clfLib.getTesting(nEmail_te)
vocal			= clfLib.getVocabulary(nWord)

### Random forests
### Option for feature selection
feature_select	= True

print "Do feature selection: %s (%s) \n" % (feature_select, nFeatures)
clf_RF		= RandomForestClassifier(n_estimators=nTree,random_state=0)
clfRFFit	= clf_RF.fit(bow_tr, emailClass_tr)

if feature_select:
	importances	= clf_RF.feature_importances_
	indices		= np.argsort(importances)[::-1]
	bow_tr		= bow_tr[:,indices[:nFeatures]]
	bow_te		= bow_te[:,indices[:nFeatures]]
	print		vocal[indices[:nFeatures]]

time_start      = time.time()
clfRFFit	= clf_RF.fit(bow_tr, emailClass_tr)
time_tr         = time.time()
pre_te_RF	= clf_RF.predict(bow_te)
time_te         = time.time()

acu_RF_tr	= clf_RF.score(bow_tr, emailClass_tr)
acu_RF_te	= clf_RF.score(bow_te, emailClass_te)
probas_		= clfRFFit.predict_proba(bow_te)
fpr, tpr, thresholds		= roc_curve(emailClass_te, probas_[:,1])
roc_auc		= auc(fpr, tpr)
recall_nospam, recall_spam	= recall_score(emailClass_te, pre_te_RF, average=None)
recall		= recall_spam
fall_out	= 1 - recall_nospam
label		= ['NotSpam', 'Spam']
report		= classification_report(emailClass_te, pre_te_RF, target_names=label)
np.savez('../HW1/Results/Metric_RF_%s_%s' % (feature_select, nFeatures), recall, fall_out, fpr, tpr, roc_auc, thresholds)
clfLib.calibration_plot(clfRFFit, bow_te, emailClass_te)
os.system('mv ../HW1/Figures/calibration_plot.pdf ../HW1/Figures/calibration_plot_RF_%s_%s.pdf' % (feature_select, nFeatures))

print "Classifier: RF \n" 
print "Training time: %f" % (time_tr - time_start)
print "Testing time: %f" % (time_te - time_tr)
print "Accuracy on training data: %0.3f" % (acu_RF_tr)
print "Accuracy on test data: %0.3f \n" % (acu_RF_te)
print report

