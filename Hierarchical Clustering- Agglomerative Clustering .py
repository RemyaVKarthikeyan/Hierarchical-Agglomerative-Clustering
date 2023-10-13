#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#run before importing Kmeans
import os
os.environ["OMP_NUM_THREADS"]='1'

#importing the dataset
dataset=pd.read_csv("Mall_Customers.csv")
dataset

#importing StandardScaler from scikit-learn library
from sklearn.preprocessing import StandardScaler

# select all rows (:) and only the columns at index 3 and 4
X=dataset.iloc[:,[3,4]].values

# creating an instance of the StandardScaler class and storing it in the variable sc_X
sc_X=StandardScaler()

#standardizing the values stored in X (mean =0, sd =1)
X=sc_X.fit_transform(X)
X

#Using dendrogram for finding the optimal number of clusters
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(15,6))
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

#Fitting the hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5,affinity='euclidean', linkage='ward')
y_hc=hc.fit_predict(X)
y_hc

#Visualizing the clusters
plt.figure(figsize=(8,8))
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='Cluster 4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='Cluster 5')
plt.title('Cluster of Customers')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Income(Scaled)')
plt.legend()
plt.show()




# In[ ]:




