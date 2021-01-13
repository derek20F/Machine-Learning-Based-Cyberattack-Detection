# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 04:37:31 2020
Security Analytics - task1 - Isolation Tree
@author: b0350
"""

import pandas as pd
import numpy as np
from numpy import quantile, where, random
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
#%% read the encoded data from csv
stime = time.time()

# Feature 1
'''
test_data = pd.read_csv('CSV_T1/test_data_encoded_new.csv')
training_data = pd.read_csv('CSV_T1/training_data_encoded_new.csv')
valid_data_with_labels = pd.read_csv('CSV_T1/valid_data_with_labels_encoded_new.csv')
'''
# Feature 3
'''
test_data = pd.read_csv('CSV_T1/test_data_pca_IPcount_port.csv')
training_data = pd.read_csv('CSV_T1/training_data_pca_IPcount_port.csv')
valid_data_with_labels = pd.read_csv('CSV_T1/valid_data_with_labels_pca_IPcount_port.csv')
'''
# Feature 2
test_data = pd.read_csv('CSV_T1/test_data_encoded_new_addFeature.csv')
training_data = pd.read_csv('CSV_T1/training_data_encoded_new_addFeature.csv')
valid_data_with_labels = pd.read_csv('CSV_T1/valid_data_with_labels_encoded_new_addFeature.csv')


print("Time spent on loading data is: " + str(time.time()-stime))

#%% drop state
'''
dropList=['srcPort','dstPort']
training_data = training_data.drop(dropList, axis=1)
test_data = test_data.drop(dropList, axis=1)
valid_data_with_labels = valid_data_with_labels.drop(dropList, axis=1)
'''
valid_data = valid_data_with_labels.drop('label', axis=1)

# sample the training_data
##training_data = training_data.sample(n=100000, axis=0, replace=False, random_state = 1)

# %%Normalize
from sklearn.preprocessing import normalize

training_data = normalize(training_data)
test_data = normalize(test_data)
valid_data = normalize(valid_data)



# %% Building model
###from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest


from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pickle

stime = time.time()

# Setting the OCSVM parameters
clf = IsolationForest(random_state=0, n_jobs = -1)
# Fit the model
clf.fit(training_data)
print("Time spent on fit ocsvm is: "+str(time.time()-stime))
#%% save the model to disk
filename = 'iTree_1013.sav'
pickle.dump(clf, open(filename, 'wb'))

# %% predit on valid_set
#training_pred = clf.predict(training_data)

stime = time.time()
valid_score = clf.score_samples(valid_data)
valid_pred = clf.predict(valid_data)
print("Time spent on predit valid_data is: "+str(time.time()-stime))

# %% Save the predicted valid result
df_valid_pred = pd.DataFrame(columns=['valid_pred', 'valid_score'])
df_valid_pred['valid_pred'] = valid_pred
df_valid_pred['valid_score'] = valid_score
#df_valid_pred.to_csv('CSV_T1/valid_pred_iTree.csv',header=True,index=False)
#df_valid_pred.to_csv('Useful_Model/Feature1-encoded/iTree/valid_pred_score_iTree.csv',header=True,index=False)
#df_valid_pred.to_csv('Useful_Model/Feature3-PCA/iTree/valid_predicted_Score.csv',header=True,index=False)
df_valid_pred.to_csv('Useful_Model/Featrue2-normalize/iTree/valid_predicted_Score.csv',header=True,index=False)


# %% Encode the label of valid set
valid_label = valid_data_with_labels['label']
valid_label=valid_label.str.contains("Botnet") #True means anomaly
valid_label = valid_label.apply(lambda x: -1 if x==True else 1) #-1 means anomaly

#%% Next, we'll obtain the threshold value from the scores by using the quantile function. Here, we'll get the lowest 3 percent of score values as the anomalies.
'''
Reference:
    https://www.datatechnotes.com/2020/04/anomaly-detection-with-one-class-svm.html

'''
'''
thresh = quantile(scores, 0.03)
print(thresh)

#Next, we'll extract the anomalies by comparing the threshold value and identify the values of elements.

index = where(scores<=thresh)
values = x[index]
'''


#%% Evaluate the model by confusion matrix
from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(valid_pred,valid_label))

#Confusion Matrix
import seaborn as sns
cm = metrics.confusion_matrix(valid_pred,valid_label,labels=[1,-1])
sns.heatmap(cm, annot = True, fmt='g')
tn,fp,fn,tp = cm.ravel()
#cm.set(xlabel="predicted label",ylabel="true label")


precision = tp/(tp+fp) #predicted postive中有多少是真的positive
recall = tp/(tp+fn) #true positive rate (true positive有多少被正確找出來)
fp_rate = fp/(fp+tn)
print("Precision = "+str(precision))
print("Recall = "+str(recall))
print("False Positive rate = "+str(fp_rate))


#%% Classification Report
from sklearn.metrics import classification_report

#target_names = ["-1","1"]
target_names = ['anomaly','normal']
print(classification_report(valid_label, valid_pred, target_names=target_names))

# %% ROC Curve & AUC
fpr,tpr, thresholds = metrics.roc_curve(valid_pred, valid_label, pos_label = -1)
auc_sc = metrics.auc(fpr, tpr)
#print("True positive rate is: "+str(tpr))
#print("Flase positive rate is "+str(fpr))

plt.plot(fpr, tpr, color='navy',label='ROC curve (area = %0.2f)' % auc_sc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic with test data')
plt.legend()
plt.show()





#%%
'''
#%% load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
'''

'''
#%% Read the result
predicted_valid = pd.read_csv('CSV_T1/predicted_valid.csv')
predicted_label = predicted_valid['predicted_label']
valid_pred = predicted_label
'''

'''
recall
Out[2]: 0.10590053585772825

precision
Out[3]: 0.35347021890378416

fp_rate
Out[4]: 0.05125321509669504

Acc = 77.24%
AUC = 0.53
'''
'''
# export the final attack report
result = pd.read_csv('CSV_T1/valid_data_with_labels.csv')
# get the index of botnet
anomaly_index_list = df_valid_pred.index[df_valid_pred['valid_pred']==-1].tolist()
#result = pd.DataFrame()

result = result.loc[anomaly_index_list]

result.to_csv('CSV_T1/attack_log_iTree_Port_no-normalize.csv')
'''

# %% predict on test_set
stime = time.time()
test_score = clf.score_samples(test_data)
test_pred = clf.predict(test_data)
print("Time spent on predict test_data is: "+str(time.time()-stime))
df_test_pred = pd.DataFrame(columns=['test_pred', 'test_score'])
df_test_pred['test_pred'] = test_pred
df_test_pred['test_score'] = test_score
#df_test_pred.to_csv('CSV_T1/test_pred_iTree_addFeature_normalize.csv',header=True,index=False)
#df_test_pred.to_csv('Useful_Model/Feature1-encoded/iTree/test_pred_score.csv',header=True,index=False)
#df_test_pred.to_csv('Useful_Model/Feature3-PCA/iTree/test_pred_score.csv',header=True,index=False)
df_test_pred.to_csv('Useful_Model/Featrue2-normalize/iTree/test_pred_score.csv',header=True,index=False)


# export the final attack report (for the test_data)
result = pd.read_csv('CSV_T1/test_data.csv')
# get the index of botnet
anomaly_index_list = df_test_pred.index[df_test_pred['test_pred']==-1].tolist()
#result = pd.DataFrame()

result = result.loc[anomaly_index_list]

#result.to_csv('CSV_T1/attack_log_iTree_Port_no-normalize.csv',header=True,index=False)
#result.to_csv('Useful_Model/Feature1-encoded/iTree/test_attack_log.csv',header=True,index=False)
#result.to_csv('Useful_Model/Feature3-PCA/iTree/test_attack_log.csv',header=True,index=False)
result.to_csv('Useful_Model/Featrue2-normalize/iTree/test_attack_log.csv',header=True,index=False)
