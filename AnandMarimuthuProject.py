# E-Commerce Customer Segmentation

# Abstract:
'''
A key challenge for e-commerce businesses is to analyze the trend in the market to increase their sales. 
The trend can be easily observed if the companies can group the customers; 
based on their activity on the ecommerce site. 
This grouping can be done by applying different criteria like previous orders, 
mostly searched brands and so on.
'''

# Problem Statement:
'''
Given the e-commerce data, 
use k-means clustering algorithm to cluster customers with similar interest.
'''

# Dataset Information:
'''
The data was collected from a well known e-commerce website over a period of time based on the customer’s search profile.
'''

# Importing all the Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Reading the dataset

dataset = pd.read_excel('cust_data.xlsx')
print(dataset)

# Exploratory Data Analysis

print(dataset.shape)
print(dataset.head())
print(dataset.describe())



# Data Cleaning

print(dataset.isnull().sum())
del dataset['Gender']
del dataset['Cust_ID']
del dataset['Orders']
print(dataset.isnull().sum())

dataset = dataset.drop_duplicates()
print(dataset.shape)
print(dataset.dtypes)

# Model

from sklearn.preprocessing import StandardScaler
df = dataset
scaler = StandardScaler()
X_std = scaler.fit_transform(df)


from sklearn.cluster import KMeans
km = KMeans(n_clusters=2) 
km.fit(X_std)


# Finding the Value of K

inertias = [] 
list_k = list(range(1, 20))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(X_std)
    inertias.append(km.inertia_)

plt.figure(figsize=(6, 6))
plt.plot(list_k, inertias, '-o')
plt.xlabel(r"Number of clusters 'k'")
plt.ylabel('Inertia')

# k = 11 is most fitting

km = KMeans(n_clusters=11) 
km.fit(X_std)

# Learning Outcome:
'''
This helps me get a better understanding of how the variables are linked to each other and I was able to apply cluster analysis to business problem such as customer segmentation
'''