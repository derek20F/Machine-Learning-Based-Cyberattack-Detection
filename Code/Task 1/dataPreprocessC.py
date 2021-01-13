# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 02:47:59 2020
Security Analytics Project 2
Data Preparation 3.
Generate new features: IP_count, and using PCA
@author: Chen-An Fan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.decomposition import PCA
'''
data_fields = ['timestamp','duration','protocol','srcIP','srcPort','direction',
               'dstIP','dstPort','state','srcService','dstService','NoPackets',
               'NoBytesBoth','NoBytesSrcToDst']



Fields Descripition of our data
(1) timestamp, (2) duration, (3) protocol, (4) source IP address, 
(5) source port, (6) direction, (7) destination IP address, 
(8) destination port, (9) state, 
(10) source type of service, (11) destination type of service, 
(12) the number of total packets, 
(13) the number of bytes transferred in both directions, 
(14) the number of bytes transferred from the source to the destination.
'''

begin = time.time()

# %% load data - take around 30 sec

stime = time.time()
test_data = pd.read_csv('CSV_T1/test_data.csv')
training_data = pd.read_csv('CSV_T1/training_data.csv')
valid_data_with_labels = pd.read_csv('CSV_T1/valid_data_with_labels.csv')
print("Time spent on loading data is: " + str(time.time()-stime))


# %% preprocessing the data / Feature extraction and selection
# Clean the data
# Remove the leading space and trailing space
print("Removing the leading and trailing spaces")
strFields = ['timestamp','protocol','srcIP','srcPort','direction','dstIP','dstPort','state']
for i in strFields:
    test_data[i] = test_data[i].str.strip()
    training_data[i] = training_data[i].str.strip()
    valid_data_with_labels[i] = valid_data_with_labels[i].str.strip()
    
    
# %%    
# Replace the missing value with -1
'''
Only srcPort, dstPort, srcService, dstService, state have nan missing values. All of them do not have -1 value originally
'''
print("replace the missing value with -1")
floatFields=['srcService','dstService']
strFields=['srcPort','dstPort','state']
for i in floatFields:
    training_data[i]=training_data[i].fillna(value=-1)
    test_data[i]=test_data[i].fillna(value=-1)
    valid_data_with_labels[i]=valid_data_with_labels[i].fillna(value=-1)

for i in strFields:
    training_data[i]=training_data[i].fillna(value='-1')
    test_data[i]=test_data[i].fillna(value='-1')
    valid_data_with_labels[i]=valid_data_with_labels[i].fillna(value='-1')


#%% Convert port with Hexidecimal into decimal
print("Convert Hex port into decimal port")
training_data['srcPort']=training_data["srcPort"].astype(str)
test_data['srcPort']=test_data["srcPort"].astype(str)
valid_data_with_labels['srcPort']=valid_data_with_labels["srcPort"].astype(str)

training_data['dstPort']=training_data["dstPort"].astype(str)
test_data['dstPort']=test_data["dstPort"].astype(str)
valid_data_with_labels['dstPort']=valid_data_with_labels["dstPort"].astype(str)



portFields = ['srcPort','dstPort']
for i in portFields:
    training_data[i]=training_data[i].apply(lambda x: int(x,0))
    test_data[i]=test_data[i].apply(lambda x: int(x,0))
    valid_data_with_labels[i]=valid_data_with_labels[i].apply(lambda x: int(x,0))

#%% Generate new feature: IP_count

# Create the dictionary to count the number of each IP in the dataset
training_srcIP_counts = training_data['srcIP'].value_counts(normalize=True).to_dict()
training_dstIP_counts = training_data['dstIP'].value_counts(normalize=True).to_dict()

test_srcIP_counts = test_data['srcIP'].value_counts(normalize=True).to_dict()
test_dstIP_counts = test_data['dstIP'].value_counts(normalize=True).to_dict()

valid_srcIP_counts = valid_data_with_labels['srcIP'].value_counts(normalize=True).to_dict()
valid_dstIP_counts = valid_data_with_labels['dstIP'].value_counts(normalize=True).to_dict()

# Generate new fields: srcIP_count, dstIP_count
test_data['srcIP_count'] = test_data['srcIP'].apply(lambda x: test_srcIP_counts.get(x))
test_data['dstIP_count'] = test_data['dstIP'].apply(lambda x: test_dstIP_counts.get(x))


training_data['srcIP_count'] = training_data['srcIP'].apply(lambda x: training_srcIP_counts.get(x))
training_data['dstIP_count'] = training_data['dstIP'].apply(lambda x: training_dstIP_counts.get(x))

valid_data_with_labels['srcIP_count'] = valid_data_with_labels['srcIP'].apply(lambda x: valid_srcIP_counts.get(x))
valid_data_with_labels['dstIP_count'] = valid_data_with_labels['dstIP'].apply(lambda x: valid_dstIP_counts.get(x))


#training_srcIP_factor


#%% Drop useless field
uselessFields = ['timestamp','srcIP','dstIP']
training_data = training_data.drop(uselessFields, axis = 1)
test_data = test_data.drop(uselessFields, axis = 1)
valid_data_with_labels = valid_data_with_labels.drop(uselessFields, axis = 1)



#%% Convert direction data into string
'''
training_data['direction'].astype(str)
test_data['direction'].astype(str)
valid_data_with_labels['direction'].astype(str)

training_data['srcService'].astype(str)
test_data['srcService'].astype(str)
valid_data_with_labels['srcService'].astype(str)

training_data['dstService'].astype(str)
test_data['dstService'].astype(str)
valid_data_with_labels['dstService'].astype(str)
'''
'''
#%% save the clean data to csv (line 1-88) (before encoding)
stime = time.time()
training_data.to_csv('CSV_T1/training_data_clean_unencoded.csv',header=True,index=False)
test_data.to_csv('CSV_T1/test_data_clean_unencoded.csv',header=True,index=False)
valid_data_with_labels.to_csv('CSV_T1/valid_data_with_labels_clean_unencoded.csv',header=True,index=False)
print("Time spent on export clean csv is: "+str(time.time()-stime))
# see the unique of direction column
# training_data['direction'].unique()
'''

#%% Encoding - Fit categorical features to numerical
from sklearn.preprocessing import LabelEncoder

### onehot = OneHotEncoder(handle_unknown='ignore')


print("Start encodering.")
stime = time.time()
valid_data = valid_data_with_labels.drop('label', axis=1)


datamap = pd.concat([training_data, test_data, valid_data])

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

training_data['protocol']=protocol_encoder.transform(training_data['protocol'])
test_data['protocol']=protocol_encoder.transform(test_data['protocol'])
valid_data_with_labels['protocol']=protocol_encoder.transform(valid_data_with_labels['protocol'])

training_data['direction']=direction_encoder.transform(training_data['direction'])
test_data['direction']=direction_encoder.transform(test_data['direction'])
valid_data_with_labels['direction']=direction_encoder.transform(valid_data_with_labels['direction'])

training_data['srcService']=srcService_encoder.transform(training_data['srcService'])
test_data['srcService']=srcService_encoder.transform(test_data['srcService'])
valid_data_with_labels['srcService']=srcService_encoder.transform(valid_data_with_labels['srcService'])

training_data['dstService']=dstService_encoder.transform(training_data['dstService'])
test_data['dstService']=dstService_encoder.transform(test_data['dstService'])
valid_data_with_labels['dstService']=dstService_encoder.transform(valid_data_with_labels['dstService'])


training_data['state']=state_encoder.transform(training_data['state'])
test_data['state']=state_encoder.transform(test_data['state'])
valid_data_with_labels['state']=state_encoder.transform(valid_data_with_labels['state'])

print("Time spent on encodering is: "+str(time.time()-stime))

#test_data['protocol'] = lb_make.fit_transform(test_data['protocol'])

# %% PCA - fit the PCA by training set
pca = PCA()
pca.fit(training_data)
#%% PCA - transform
training_data_pca = pca.transform(training_data)
test_data_pca = pca.transform(test_data)
valid_label = valid_data_with_labels['label'] #take out the label, do not trasform the label
valid_data = valid_data_with_labels.drop('label', axis=1)
valid_data_pca = pca.transform(valid_data)
#%% put them back to dataframe
training_data_pca = pd.DataFrame(data=training_data_pca, index=None, columns=training_data.columns)
test_data_pca = pd.DataFrame(data=test_data_pca, index=None, columns=test_data.columns)
valid_data_pca = pd.DataFrame(data=valid_data_pca, index=None, columns=valid_data.columns)
valid_data_pca_with_labels = valid_data_pca.assign(label = valid_label)

# %% Save the encoded and clean data
stime = time.time()
training_data_pca.to_csv('CSV_T1/training_data_pca_IPcount_port.csv',header=True,index=False)
test_data_pca.to_csv('CSV_T1/test_data_pca_IPcount_port.csv',header=True,index=False)
valid_data_pca_with_labels.to_csv('CSV_T1/valid_data_with_labels_pca_IPcount_port.csv',header=True,index=False)
print("Time spent on export encoded csv is: "+str(time.time()-stime))


end = time.time()

print("Total time spent on the preprocess is: "+str(end-begin))