import numpy as np
import pandas as pd

import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from imblearn.over_sampling import SMOTE

import DataGeneration as dg

EPS=1.e-6

# we check whether SMOTE is needed. If outlier proportion is already > 5%, then sampling is not needed.
def isSamplingNeeded(labels):
    nb_outliers = labels[labels[labels.columns[0]]=='Outlier'].shape[0]
    nb_inliers = labels[labels[labels.columns[0]]=='Inlier'].shape[0]
    outlier_proportion = nb_outliers/nb_inliers
    return False if outlier_proportion > 0.05 else True

# perform SMOTE to start from a more balanced dataset
def performSMOTE(data, labels):
    oversample = SMOTE(sampling_strategy=0.05) # we want at least 5% of minority class
    data_aug, label_aug = oversample.fit_resample(data, labels)
    nb_generated_outliers = data_aug.shape[0] - data.shape[0]
    print("number of generated outliers: {}".format(nb_generated_outliers))
    return data_aug, label_aug

# select neighborhood according to class proportions
def selectNeighborhood(data, labels, outlier):
    ratio_outliers = 1
    n_neighbors = 20
    while (ratio_outliers < 0.1 or ratio_outliers > 0.9): # we want at least 10% of the neighbor minor class
        n_neighbors = n_neighbors + 1
        if (n_neighbors % 100) == 0:
            print("trying with {} neighbors".format(n_neighbors))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(data)
        distances, indices = nbrs.kneighbors(np.reshape(np.array(outlier),(1,-1)))
        distances = distances.flatten()
        indices = indices.flatten()
        X = data.iloc[indices,:]
        Y = labels.iloc[indices,:]
        nb_outliers_neighborhood = Y[Y['Outlier_Inlier']=='Outlier'].shape[0]
        nb_inliers_neighborhood = Y[Y['Outlier_Inlier']=='Inlier'].shape[0]
        try:
            ratio_outliers = nb_outliers_neighborhood/nb_inliers_neighborhood
        except:
            continue # in case of a division by zero
    print("{}% outliers in neighborhood using {} neighbors".format(int(ratio_outliers*100), n_neighbors))
    return X, Y, indices, distances

# compute bandwidth as 0.05*SQRT(distance to the closest inlier)
def kernel_rbf(x, y, sigma):
    d = 0
    for i in range(len(x)):
        d_temp = (x[i]-y[i])**2
        d = d + d_temp
    return np.exp(-0.5*(d/sigma**2))

def computeBandwidth(labels, indices, distances):
    for idx, indice_neighbor in enumerate(indices):
        row = labels[labels['Outlier_Inlier'].index==indice_neighbor]
        if(row['Outlier_Inlier'].iloc[0]=='Inlier'):
            bandwidth = 0.05*np.sqrt(distances[idx])
            print("Bandwidth={}".format(bandwidth))
            break
    return bandwidth

# perform StratifiedKFold validation to select the best regularization strength
def findBestRegularization(X, Y, outlier, bandwidth):
    n_splits = 3
    ave_pr_dict = {}
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    for lambda_reg in np.power(10,np.arange(1,10))/(10e4):

        ave_pr_mean = 0

        for train_index, test_index in kf.split(X, Y):

            X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
            Y_train, Y_test = np.array(Y)[train_index], np.array(Y)[test_index]

            # a) Compute weights with kernel. Inspired by LIME method https://arxiv.org/abs/1602.04938.
            outlier_list = np.reshape(np.array(outlier),(1,-1)) * np.ones((X_train.shape[0],X_train.shape[1]))
            sample_weight = []
            for idx, elt in enumerate(np.array(X_train)):
                sample_weight.append(kernel_rbf(elt, outlier_list[idx], bandwidth))

            # b) Perform logistic regression
            clf = LogisticRegression(random_state=42, solver='liblinear', penalty='l1', C=1/lambda_reg)
            clf.fit(X_train, np.array(Y_train).ravel(), sample_weight=sample_weight)

            proba = clf.predict_proba(X_test)

            ave_pr_mean = average_precision_score(Y_test, proba[:,1], pos_label='Outlier') + ave_pr_mean

        ave_pr_dict[lambda_reg] = ave_pr_mean / n_splits

    lambda_opt = max(ave_pr_dict, key=ave_pr_dict. get)
    print("best regularization strength: {}".format(lambda_opt))
    print("Average PR: {}".format(ave_pr_dict[lambda_opt]))
    return lambda_opt

# compute logistic regression using above parameters
def computeRegression(X, Y, lambda_opt, bandwidth, outlier):
    outlier_list = np.reshape(np.array(outlier),(1,-1)) * np.ones((X.shape[0],X.shape[1]))
    sample_weight = []
    for idx, elt in enumerate(np.array(X)):
        sample_weight.append(kernel_rbf(elt, outlier_list[idx], bandwidth))
    clf = LogisticRegression(random_state=42, solver='liblinear', penalty='l1', C=1/lambda_opt)
    clf.fit(X, np.array(Y).ravel(), sample_weight=sample_weight)
    return clf

# display coefficients to see most important features
def showCoef(X, clf):
    coefs = np.array(clf.coef_).flatten()
    result = pd.DataFrame(coefs,columns=['weight'])
    result['name'] = X.columns
    result['strength'] = result['weight'].abs()
    result = result.sort_values('strength',ascending=False)
    result = result[result.strength>EPS].drop('strength',axis=1)
    fig = px.bar(result, x='name', y='weight')
    plot(fig)

def explain(outlier_idx, data, labels):
    if(isSamplingNeeded(labels)):                                                 # SMOTE is needed only if outlier proportion is < 5%
        data_aug, label_aug = performSMOTE(data, labels)                          # perform SMOTE to start from a more balanced dataset
    else:
        print('No sampling needed')
        data_aug = data
        label_aug = labels
    label_aug.columns = ['Outlier_Inlier']
    outlier = data[data.index==outlier_idx]
    X, Y, indices, distances = selectNeighborhood(data_aug, label_aug, outlier)   # select neighborhood according to class proportions
    bandwidth = computeBandwidth(label_aug, indices, distances)                   # compute bandwidth as 0.05*SQRT(distance to the closest inlier)
    lambda_opt = findBestRegularization(X, Y, outlier, bandwidth)                 # perform StratifiedKFold validation to select the best regularization strength
    clf = computeRegression(X, Y, lambda_opt, bandwidth, outlier)                 # compute logistic regression using above parameters
    showCoef(X, clf)                                                              # display coefficients to see most important features

X_fake, Y_fake, idx_outliers = dg.fakeDataset5Dimensions()
X_fake[X_fake.index==idx_outliers[0]] # we will test on the first outlier

explain(idx_outliers[0], X_fake, Y_fake)