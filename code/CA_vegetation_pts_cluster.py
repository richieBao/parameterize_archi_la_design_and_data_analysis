# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 23:16:29 2021

@author: richie
"""

import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

CA_vegetation_fp=r'../data/CA_vegetation_pts.csv'
CA_vegetation=pd.read_csv(CA_vegetation_fp,sep=",",header=None) 
# print(CA_vegetation)
X=CA_vegetation.to_numpy()
print(X.shape)
db=DBSCAN(eps=0.5, min_samples=10).fit(X)
labels=db.labels_
print(labels)
print(np.unique(labels))

fig, ax=plt.subplots()
ax.scatter(X[:,0],X[:,1],s=5,marker='o',c=labels)
plt.show()

label_df=pd.DataFrame(labels,columns=["cluster_label"])
print(label_df)
label_df.to_csv('../data/CA_vegetation_pts_cluster.csv', index=False)


if __name__ == "__main__":
    pass
    

    