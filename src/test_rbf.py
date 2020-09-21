# This script aims to clarify the RBF kernel implementation
# We will make sure the RBF kernel implementation
# using sklearn library is similar to our manual
# implementation

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

def kernel_rbf(x, y, sigma):
    d = 0
    for i in range(len(x)):
        d_temp = (x[i]-y[i])**2
        d = d + d_temp
    weight = np.exp(-0.5*(d/sigma**2))
    return weight


outlier = [5,6,5,6]
df_outlier = pd.DataFrame(columns=['feat_1','feat_2','feat_3','feat_4'])
df_outlier.loc[0] = outlier

X_train = pd.DataFrame(columns=['feat_1','feat_2','feat_3','feat_4'])
X_train.loc[0] = [7,7,7,8]
X_train.loc[1] = [7,9,7,8]
X_train.loc[2] = [10,7,7,8]
X_train.loc[3] = [7,1,7,8]
X_train.loc[4] = [9,1,7,8]

outlier_list = np.reshape(np.array(outlier),(1,-1)) * np.ones((X_train.shape[0],X_train.shape[1]))

a = []
for idx, elt in enumerate(np.array(X_train)):
    a.append(kernel_rbf(elt, outlier_list[idx], 5))

b = rbf_kernel(X=np.array(outlier).reshape(1, -1), Y=X_train, gamma=1/(2*5**2)).flatten()

print((a==b).all()) # the two implementation are equal