# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 00:44:25 2020
Task 2 - Generate the adversarial example
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
print("Start to load the data...")
training_data_with_labels = pd.read_csv('CSV_T2/training_data_with_labels.csv')
test_data_with_labels = pd.read_csv('CSV_T2/test_data_with_labels.csv')
valid_data_with_labels = pd.read_csv('CSV_T2/valid_data_with_labels.csv')
print("Time spent on loading data is: " + str(time.time()-stime))
 
# %% preprocessing the data / Feature extraction and selection
# Clean the data
# Remove the leading space and trailing space
print("Removing the leading and trailing spaces")
strFields = ['timestamp','protocol','srcIP','srcPort','direction','dstIP','dstPort','state']
for i in strFields:
    test_data_with_labels[i] = test_data_with_labels[i].str.strip()
    training_data_with_labels[i] = training_data_with_labels[i].str.strip()
    valid_data_with_labels[i] = valid_data_with_labels[i].str.strip()
    
# %%    
# Replace the missing value with -1
'''
Only srcPort, dstPort, srcService, dstService, state have nan missing values. All of them do not have -1 value originally
'''
#問題出在service!。service有missing value
print("replace the missing value with -1")
floatFields=['srcService','dstService']
strFields=['srcPort','dstPort','state']
for i in floatFields:
    training_data_with_labels[i]=training_data_with_labels[i].fillna(value=-1)
    test_data_with_labels[i]=test_data_with_labels[i].fillna(value=-1)
    valid_data_with_labels[i]=valid_data_with_labels[i].fillna(value=-1)

for i in strFields:
    training_data_with_labels[i]=training_data_with_labels[i].fillna(value='-1')
    test_data_with_labels[i]=test_data_with_labels[i].fillna(value='-1')
    valid_data_with_labels[i]=valid_data_with_labels[i].fillna(value='-1')

#%% Convert port with Hexidecimal into decimal
print("Convert Hex port into decimal port")
training_data_with_labels['srcPort']=training_data_with_labels["srcPort"].astype(str)
test_data_with_labels['srcPort']=test_data_with_labels["srcPort"].astype(str)
valid_data_with_labels['srcPort']=valid_data_with_labels["srcPort"].astype(str)

training_data_with_labels['dstPort']=training_data_with_labels["dstPort"].astype(str)
test_data_with_labels['dstPort']=test_data_with_labels["dstPort"].astype(str)
valid_data_with_labels['dstPort']=valid_data_with_labels["dstPort"].astype(str)



portFields = ['srcPort','dstPort']
for i in portFields:
    training_data_with_labels[i]=training_data_with_labels[i].apply(lambda x: int(x,0))
    test_data_with_labels[i]=test_data_with_labels[i].apply(lambda x: int(x,0))
    valid_data_with_labels[i]=valid_data_with_labels[i].apply(lambda x: int(x,0))
    
# Modify the data
#%% Drop useless field
uselessFields = ['timestamp','srcIP','dstIP']
m_training_data_with_labels = training_data_with_labels.drop(uselessFields, axis = 1)
m_test_data_with_labels = test_data_with_labels.drop(uselessFields, axis = 1)
m_valid_data_with_labels = valid_data_with_labels.drop(uselessFields, axis = 1)

#%% Encoding - Fit categorical features to numerical
from sklearn.preprocessing import LabelEncoder

print("Start encodering.")
stime = time.time()
##### m_valid_data = m_valid_data_with_labels.drop('label', axis=1)


datamap = pd.concat([m_training_data_with_labels, m_test_data_with_labels, m_valid_data_with_labels])

datamap['direction']=datamap['direction'].astype(str)

datamap = datamap.convert_dtypes()


protocol_encoder = LabelEncoder()
direction_encoder = LabelEncoder()
state_encoder = LabelEncoder()
srcService_encoder = LabelEncoder()
dstService_encoder = LabelEncoder()


protocol_encoder.fit(datamap['protocol'])
direction_encoder.fit(datamap['direction'])
srcService_encoder.fit(datamap['srcService'])
dstService_encoder.fit(datamap['dstService'])
state_encoder.fit(datamap['state'])

#del datamap

m_training_data_with_labels['protocol']=protocol_encoder.transform(m_training_data_with_labels['protocol'])
m_test_data_with_labels['protocol']=protocol_encoder.transform(m_test_data_with_labels['protocol'])
m_valid_data_with_labels['protocol']=protocol_encoder.transform(m_valid_data_with_labels['protocol'])

m_training_data_with_labels['direction']=direction_encoder.transform(m_training_data_with_labels['direction'])
m_test_data_with_labels['direction']=direction_encoder.transform(m_test_data_with_labels['direction'])
m_valid_data_with_labels['direction']=direction_encoder.transform(m_valid_data_with_labels['direction'])

m_training_data_with_labels['srcService']=srcService_encoder.transform(m_training_data_with_labels['srcService'])
m_test_data_with_labels['srcService']=srcService_encoder.transform(m_test_data_with_labels['srcService'])
m_valid_data_with_labels['srcService']=srcService_encoder.transform(m_valid_data_with_labels['srcService'])

m_training_data_with_labels['dstService']=dstService_encoder.transform(m_training_data_with_labels['dstService'])
m_test_data_with_labels['dstService']=dstService_encoder.transform(m_test_data_with_labels['dstService'])
m_valid_data_with_labels['dstService']=dstService_encoder.transform(m_valid_data_with_labels['dstService'])


m_training_data_with_labels['state']=state_encoder.transform(m_training_data_with_labels['state'])
m_test_data_with_labels['state']=state_encoder.transform(m_test_data_with_labels['state'])
m_valid_data_with_labels['state']=state_encoder.transform(m_valid_data_with_labels['state'])

print("Time spent on encodering is: "+str(time.time()-stime))





#%% MLP
from sklearn.neural_network import MLPClassifier
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc

m_training_data = m_training_data_with_labels.drop('label', axis=1)
training_data_labels = m_training_data_with_labels.label

m_test_data = m_test_data_with_labels.drop('label', axis=1)
test_data_labels = m_test_data_with_labels.label

m_valid_data = m_valid_data_with_labels.drop('label', axis=1)
valid_data_labels = m_valid_data_with_labels.label

# Normalize
from sklearn.preprocessing import normalize
m_training_data = normalize(m_training_data)
m_test_data = normalize(m_test_data)
m_valid_data = normalize(m_valid_data)



# Encode the label to -1 and 1 (-1 means anomaly)
training_data_labels = training_data_labels.str.contains("Botnet") #True means anomaly
training_data_labels = training_data_labels.apply(lambda x: -1 if x==True else 1) #-1 means anomaly

test_data_labels = test_data_labels.str.contains("Botnet") #True means anomaly
test_data_labels = test_data_labels.apply(lambda x: -1 if x==True else 1) #-1 means anomaly


valid_data_labels = valid_data_labels.str.contains("Botnet") #True means anomaly
valid_data_labels = valid_data_labels.apply(lambda x: -1 if x==True else 1) #-1 means anomaly

# total using 11 features
# Train MLP model
trainStartTime = time.time()
#mlp = MLPClassifier(hidden_layer_sizes=(20,10), activation = 'logistic' ,solver='adam', learning_rate_init=0.005, learning_rate = 'constant')
#mlp = MLPClassifier(hidden_layer_sizes=(100,50,25,10,5), activation = 'logistic' ,solver='adam', learning_rate_init=0.000005, learning_rate = 'constant', alpha=1)
#mlp = MLPClassifier(hidden_layer_sizes=(20,10,4,2), activation = 'tanh' ,solver='sgd', learning_rate_init=0.0005, learning_rate = 'constant', alpha=0.005)
#mlp = MLPClassifier(hidden_layer_sizes=(10,20,10), activation = 'identity' ,solver='sgd', learning_rate_init=0.000005, learning_rate = 'constant', alpha=0.0001)
mlp = MLPClassifier(hidden_layer_sizes=(8,6,4), activation = 'logistic',
                    solver='adam', learning_rate_init=0.001, warm_start=True,
                    learning_rate = 'constant', alpha=0.0001) 
                    #batch_size=100000) #early_stopping=True, n_iter_no_change=20, 
y_class = [-1,1] #the possible outcome

epochs = 10

train_score_curve = []
test_score_curve = []
test_predicted_probability = []

for i in range(epochs):
    mlp.partial_fit(m_training_data, training_data_labels, y_class)
    print("========== Step " + str(i+1) + " ==========")
    #train_predicted_probability = mlp.predict_proba(train_features)
    
    train_score = mlp.score(m_training_data, training_data_labels) 
    train_score_curve.append(train_score)
    print("Train score = " + str(train_score))
    # See the score on unseen valid dataset
    test_score = mlp.score(m_test_data, test_data_labels)
    test_score_curve.append(test_score)
    print("test score = " + str(test_score))
    test_predicted_probability = mlp.predict_proba(m_test_data)
    #fpr,tpr, thresholds = roc_curve(test_data_labels, test_predicted_probability[:,0])
    #auc_sc = auc(fpr, tpr)
    #print("Test AUC = " + str(auc_sc))
    
    #test_predicted_probability = mlp.predict_proba(test_features)
    #print("predict = " + str(test_predicted_probability))

trainFinishTime = time.time()
print("Time spent on training is: " + str(trainFinishTime - trainStartTime) + " sec")

# Predict on the valid set and get the probability
valid_predicted_probability = mlp.predict_proba(m_valid_data)
valid_predicted_labels = mlp.predict(m_valid_data)
print("predict = " + str(valid_predicted_probability))



# ROC Curve & AUC on valid set
fpr,tpr, thresholds = roc_curve(valid_data_labels, valid_predicted_probability[:,0])
auc_sc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='navy',label='Valid ROC curve (area = %0.2f)' % auc_sc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Valid Data ROC')
plt.legend()
plt.show()


# Plot training and valid curve

plt.close('all')
x_axis = []
for i in range(epochs):
    x_axis.append(i+1)

plt.figure()
plt.plot(x_axis, train_score_curve, "r", x_axis, test_score_curve, "b")
plt.legend(labels=['train','test'],loc='best')
plt.xlabel("Training Step")
plt.ylabel("Accuracy")
#plt.plot(maxValidAccStep, maxValidAcc, marker='x', markersize=5, color="Green")
#plt.annotate(str(maxValidAccP)+"%",xy=(maxValidAccStep, maxValidAcc))

#plt.savefig('train and valid curve.png', dpi=600)

#%% see the weight of mlp model
# print(mlp.coefs_)

#%%
#Confusion Matrix
import seaborn as sns
from sklearn import metrics
plt.close('all')
cm = metrics.confusion_matrix(valid_predicted_labels,valid_data_labels,labels=[1,-1]) #Note the label order: -1 is the positive
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
#from sklearn.metrics import classification_report

#target_names = ["-1","1"]
target_names = ['anomaly','normal']
print(classification_report(valid_data_labels, valid_predicted_labels, target_names=target_names))
#%% save the model to disk
import pickle
filename = 'task2_mlp.sav'
pickle.dump(mlp, open(filename, 'wb'))
