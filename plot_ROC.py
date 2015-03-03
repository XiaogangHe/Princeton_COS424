#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt

"""
ROC plot
"""
clf_set		= ['MNB', 'BNB', 'KNN', 'LSVM', 'RF']
clf_label	= ['MNB', 'BNB', 'KNN', 'Linear SVM', 'RF']
colorsT		= ['-b', '-r', '-g', '-k', '-y', '-p']
colorsF		= ['--b', '--r', '--g', '--k', '--y', '--p']
feature_set	= [20, 50, 100, 200, 500, 1000]

plt.figure(figsize=(6,6))
for i, iclf in enumerate(clf_set):
	#metric_T	= np.load('../HW1/Results/Metric_%s_True_%s.npz' % (iclf, feature_set[i]))
	metric_T1	= np.load('../HW1/Results/Metric_%s_True_500.npz' % (iclf))
	metric_T2	= np.load('../HW1/Results/Metric_%s_True_200.npz' % (iclf))
	metric_F	= np.load('../HW1/Results/Metric_%s_False_200.npz' % (iclf))
	plt.plot(metric_T1['arr_2'], metric_T1['arr_3'], colorsT[i], linewidth=2.5, label=clf_label[i])
	plt.plot(metric_T2['arr_2'], metric_T2['arr_3'], colorsF[i], linewidth=2.5)
	plt.plot(metric_F['arr_2'], metric_F['arr_3'], colorsF[i], linewidth=1)
			    
#plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')
plt.rc('font', family='Palatino') 
plt.xlim([-0.005, 0.025])
plt.ylim([0.7, 1.05])
plt.legend(loc=4, prop={'size':10})
plt.xlabel('False Positive Ratio', size='large')
plt.ylabel('True Positive Ratio', size='large')
plt.title('ROC Curve', size='x-large')
plt.savefig('../HW1/Figures/ROC_curve.pdf', format='PDF')
plt.show()             
