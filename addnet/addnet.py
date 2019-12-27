import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from addnet.utils import *

class SubNet(object):
    def __init__(self, binarization, reverses, **kawgs):
        self.coef_ = None
        self.binarization = binarization
        # list of list of thresholds
        self.reverses = reverses
        # default: <, if reverse is True, >
        ## error check
        if len(self.binarization) != len(self.reverses):
            raise ValueError(f"nanka okashii")

        # seikika kou. 
        self.regularization_param = 1.
        self.bin_dim = sum(map(len, self.binarization)) + 1

        # make bounds of coefficients
        self.bounds = [None] * self.bin_dim
        idx = 0
        for i, (thresholds, reverse) in enumerate(zip(self.binarization, self.reverses)):
            for j, t in enumerate(thresholds):
                if reverse and j==0:
                    self.bounds[idx] = (None, None)
                elif not reverse and j==len(thresholds)-1:
                    self.bounds[idx] = (None, None)
                else:
                    self.bounds[idx] = (0, None)
                idx += 1
        self.bounds[self.bin_dim-1] = (None, None)
        # print(self.bounds)

        
    def binarize(self, X):
        if len(X.shape) != 2:
            raise ValueError(f"X must be 2d-array")
        
        binarized_x = np.zeros((X.shape[0], self.bin_dim))
        # b_{p, i} in paper
        idx = 0
        for i, (thresholds, reverse) in enumerate(zip(self.binarization, self.reverses)):
            for t in thresholds:
                if not reverse: # default comparison: <
                    binarized_x[X[:,i]<t, idx] = 1.
                    binarized_x[X[:,i]>=t, idx] = 0.
                else: # >
                    binarized_x[X[:,i]>t, idx] = 1.
                    binarized_x[X[:,i]<=t, idx] = 0.
                idx += 1
        # dammy variable. it behaves as bias term.
        binarized_x[:, self.bin_dim-1] = 1.
        return binarized_x
        

    def fit(self, X, y):
        """it calculate weight vector with X and y.

        Args:
            X: 2d-array.
            y: labels. 1d-array
        """
        # binarize X
        binarized_x = self.binarize(X)

        #
        ranged_sigmoid = get_ranged_sigmoid()
        
        # objective function
        def obj_f_l2(w):
            px = ranged_sigmoid(binarized_x.dot(w))
            return - ((1-y) * np.log(1-px) \
                      + y * np.log(px)).sum() \
                      + self.regularization_param * np.dot(w, w)
        # derivative function 
        def jac_f_l2(w):
            px = ranged_sigmoid(binarized_x.dot(w))
            return ((px - y).reshape(-1, 1) * binarized_x).sum(axis=0) + 2*self.regularization_param*w
        
        x0 = np.random.random(binarized_x.shape[1])
        result = minimize(obj_f_l2, x0=x0,
                          bounds=self.bounds, jac=jac_f_l2, method="SLSQP")
        
        if not result.success:
            raise ValueError(result.message)
        self.coef_ = result.x

        # toriaezu sklearn de yattemiru
        # koredato umaku ikanai hugou wo seigyo dekinai 
        # clf = LogisticRegression(fit_intercept=False)
        # clf.fit(binarized_x, y)
        # self.coef_ = clf.coef_.copy()[0]
        # self.coef_.shape = (bin_dim,)

        
    def decision_function(self, X):
        return sigmoid(self.binarize(X) @ self.coef_)
        
        
    def predict(self, X):
        f = self.decision_function(X)
        # f[f < 0.5] = 0; f[f >= 0.5] = 1
        return np.where(f < 0.5, 0, 1)



class TwoLayerAddNet(object):
    def __init__(self, bin_thresholds, reverses=None):
        self.sub_feature_sets = []
        # list of feature set of children subnetwork
        self.first_networks = []
        # list of 1st subnetworks. len(self.first_networks) == len(sub_feature_set)
        self.second_networks = None
        self.bin_thresholds = bin_thresholds
        self.reverses = reverses
        
        
    def fit(self, X, y, sub_feature_sets):
        # make subnetwork and training data of 2nd layer network
        X2 = np.empty((X.shape[0], len(sub_feature_set)))
        for i, feature_set in enumerate(sub_feature_set):
            subnet = SubNet()
            subnet.fit(X, y)
            self.first_networks.append(subnet)
            X2[:, i] = subnet.predict(X)
            
        # train 2nd layer weight
        self.second_networks = SubNet()
        self.second_networks.fit(X2, y)
        
        
    def predict(self, X, y):
        p = None
        return p    



