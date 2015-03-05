#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	email_process			as	ep
import	classifier_Library		as	clfLib
import	time
import	os
from	pandas				import	DataFrame
from	sklearn.feature_selection	import	SelectKBest
from	sklearn.feature_selection	import	chi2
from	sklearn.neighbors		import	KNeighborsClassifier
from	sklearn.metrics			import	classification_report
from 	sklearn.metrics			import	recall_score
from	sklearn.metrics			import	roc_curve, auc

### Read datasets
nEmail_tr	= 45000
nEmail_te	= 5000
nWord		= 9579
nFeatures	= 200
nNeigh		= 20

bow_tr, emailClass_tr = clfLib.getTraining(nEmail_tr)
bow_te, emailClass_te = clfLib.getTesting(nEmail_te)

### Option for feature selection
feature_select	= False

print "Do feature selection: %s (%s, %s) \n" % (feature_select, nFeatures, nNeigh)
if feature_select:
	kbest	= SelectKBest(chi2, k=nFeatures)
	bow_tr	= kbest.fit_transform(bow_tr, emailClass_tr)
	bow_te	= kbest.transform(bow_te)

### KNN
time_start	= time.time()
clf_KNN		= KNeighborsClassifier(n_neighbors=nNeigh)
clfKNNFit	= clf_KNN.fit(bow_tr,emailClass_tr)
time_tr 	= time.time()
pre_te_KNN	= clf_KNN.predict(bow_te)
time_te 	= time.time()

acu_KNN_tr	= clf_KNN.score(bow_tr, emailClass_tr)
acu_KNN_te	= clf_KNN.score(bow_te, emailClass_te)
probas_		= clfKNNFit.predict_proba(bow_te)
fpr, tpr, thresholds		= roc_curve(emailClass_te, probas_[:,1])
roc_auc		= auc(fpr, tpr)
recall_nospam, recall_spam	= recall_score(emailClass_te, pre_te_KNN, average=None)
recall		= recall_spam
fall_out	= 1 - recall_nospam
label		= ['NotSpam', 'Spam']
report		= classification_report(emailClass_te, pre_te_KNN, target_names=label)
np.savez('../HW1/Results/Metric_KNN_%s_%s_%s' % (feature_select, nFeatures, nNeigh), recall, fall_out, fpr, tpr, roc_auc, thresholds)
clfLib.calibration_plot(clfKNNFit, bow_te, emailClass_te)
os.system('mv ../HW1/Figures/calibration_plot.pdf ../HW1/Figures/calibration_plot_KNN_%s_%s_%s.pdf' % (feature_select, nFeatures, nNeigh))

print "Classifier: KNN \n" 
print "Training time: %f" % (time_tr - time_start)
print "Testing time: %f" % (time_te - time_tr)
print "Accuracy on training data: %0.3f" % (acu_KNN_tr)
print "Accuracy on test data: %0.3f \n" % (acu_KNN_te)
print report

