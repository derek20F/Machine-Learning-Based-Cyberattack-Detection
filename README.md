# Machine learning based cyberattack detection
## Summary
* Extracted features (such as IP, port, protocol, and duration) from network traffic data
* Implemented two unsupervised clustering algorithms (One-class SVM & Isolation Forest) to distinguish the botnet traffic from the normal traffic



## Description
There are two tasks in this project:

##### Task1:

Applying unsupervised machine learning techniques for anomaly detection

In this task, I use one-class SVM and isolation forest as my unsupervised model.

##### Task2:

Using gradient descent-based method to generate adversarial samples against supervised learning models beyond the computer vision domain. 

In this task, I use MLP as my machine learning model.

##### Data

Two network traffic (NetFlow) datasets are provided, one for each task. Both datasets contain botnet traffic and normal traffic. You need to identify botnet IP addresses from both two datasets. In addition, for Task II you also need to choose a botnet IP address, and explain how to manipulate its network traffic in order to bypass detection.









