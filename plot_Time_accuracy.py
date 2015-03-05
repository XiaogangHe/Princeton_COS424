#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	xlrd

"""
Plot computational time
"""
colorsTr	= ['-bp', '-rp', '-gp', '-kp', '-yp', '-pp']
colorsTe	= ['-.bp', '-.rp', '-.gp', '-.kp', '-.yp', '-.pp']
colorsAccTr	= ['-bo', '-ro', '-go', '-ko', '-yo', '-po']
colorsAccTe	= ['-.bo', '-.ro', '-.go', '-.ko', '-.yo', '-.po']
clf_set		= ['MNB', 'BNB', 'KNN', 'LSVM', 'RF']
clf_label	= ['MNB', 'BNB', 'KNN', 'Linear SVM', 'RF']
feature_set	= ['20', '50', '100', '200', '500', '1000', 'All']

data		= xlrd.open_workbook('../HW1/Results/ComputationTime.xlsx')
table_time	= data.sheet_by_name(u'Time')
table_accu	= data.sheet_by_name(u'Accuracy')

plt.figure(figsize=(6,6))
for i, iclf in enumerate(clf_set):
	ax1		= plt.subplot(2,1,1)
	time_clf	= table_time.row_values(i+2)[1:]
	plt.plot(time_clf[:7], colorsTr[i], linewidth=2.5, markersize=8, label=clf_label[i])
	plt.plot(time_clf[7:], colorsTe[i], linewidth=2.5, markersize=8)
	plt.gca().set_yscale('log')
	plt.ylabel('Time (s)', size='large')
	plt.xlim([-0.5, 6.5])
	plt.setp( ax1.get_xticklabels(), visible=False)

	plt.subplot(2,1,2)
	accu_clf	= table_accu.row_values(i+2)[1:]
	plt.plot(accu_clf[:7], colorsAccTr[i], linewidth=2.5, markersize=8, label=clf_label[i])
	plt.plot(accu_clf[7:], colorsAccTe[i], linewidth=2.5, markersize=8)
	plt.xlim([-0.5, 6.5])
	plt.ylim([0.9, 1.01])
	plt.ylabel('Accuracy', size='large')
	plt.legend(loc=0, prop={'size':8})

#plt.gca().set_xscale('log')
plt.subplots_adjust(hspace=0.05, bottom=0.125)
plt.rc('font', family='Palatino') 
plt.xticks(range(7),feature_set)
plt.xlabel('Number of selected features', size='large')
plt.savefig('../HW1/Figures/Time_Accuracy.pdf', format='PDF')
plt.show()             
