import numpy as np
import pandas as pd

def fakeDataset2Dimensions(): # generated artificial gaussian data (2 dimensions)
    fake_data_p = 2
    fake_data_n = 1000

    X_fake = np.random.randn(fake_data_n, fake_data_p)
    Y_fake = np.zeros((fake_data_n,1))

    # outliers
    idx_outliers = np.random.randint(0, 1000, 10) # 1% outliers
    sigma = 0.8
    for i in idx_outliers:
        res = np.random.randn(1, fake_data_p)
        r = np.random.randint(0,3)
        if(r==0):
            mu = [20,0]
        elif(r==1):
            mu = [0,20]
        else:
            mu = [20,20]
        X_fake[i] = np.array(mu + res * sigma)
    Y_fake[idx_outliers] = 1

    # convert to DF
    X_fake_df = pd.DataFrame(columns=['feat1','feat2'], data=X_fake)
    Y_fake_df = pd.DataFrame(columns=['to_remove'], data=Y_fake)

    # add custom column
    Y_fake_df['PD_Credit_Card_outlier_gael_is_outlier'] = 'Inlier'
    Y_fake_df.iloc[idx_outliers] = 'Outlier'
    Y_fake_df = Y_fake_df.drop(columns=['to_remove'])

    return X_fake_df, Y_fake_df, idx_outliers

def fakeDataset5Dimensions(): # generated artificial gaussian data (5 dimensions)
    fake_data_p = 5
    fake_data_n = 1000

    X_fake = np.random.randn(fake_data_n, fake_data_p)
    Y_fake = np.zeros((fake_data_n,1))

    # outliers
    idx_outliers = np.random.randint(0, 1000, 10) # 1% outliers
    sigma = 0.8
    for i in idx_outliers:
        res = np.random.randn(1, fake_data_p)
        r = np.random.randint(0,2)
        if(r==0):
            mu = [10,100,0,0,0]
        elif(r==1):
            mu = [0,100,1000,10000,0]
        X_fake[i] = np.array(mu + res * sigma)
    Y_fake[idx_outliers] = 1

    # convert to DF
    X_fake_df = pd.DataFrame(columns=['feat1','feat2', 'feat3', 'feat4', 'feat5'], data=X_fake)
    Y_fake_df = pd.DataFrame(columns=['to_remove'], data=Y_fake)

    # add custom column
    Y_fake_df['PD_Credit_Card_outlier_gael_is_outlier'] = 'Inlier'
    Y_fake_df.iloc[idx_outliers] = 'Outlier'
    Y_fake_df = Y_fake_df.drop(columns=['to_remove'])

    return X_fake_df, Y_fake_df, idx_outliers