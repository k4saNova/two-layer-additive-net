import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import check_increasing
from addnet.utils import *

class SubNet(object):
    def __init__(self, max_seg=16, binarization=None, **kawgs):
        self.coef_ = None
        self.max_seg = max_seg
        # a coefficient vector of the model
        if binarization is not None:
            self.bin_thresholds = binarization
            self.bin_dim = sum(map(len, self.bin_thresholds)) + 1
        else:
            self.bin_thresholds = None
            self.bin_dim = None
        self.max_iter = 1000
        # list of list of thresholds
        # seikika kou.
        self.regularization_param = 1.
        self.uncertainry_metrics = "gini"

            
    def binarize(self, X):
        if len(X.shape) != 2:
            raise ValueError(f"X must be 2d-array")

        binarized_x = np.zeros((X.shape[0], self.bin_dim))
        # b_{p, i} in paper
        idx = 0
        for i, (thresholds, increasing) in enumerate(zip(self.bin_thresholds, self.is_increasing)):
            for t in thresholds:
                if increasing: # x > t
                    binarized_x[X[:,i]>t, idx] = 1.
                    binarized_x[X[:,i]<=t, idx] = 0.
                else: # t > x
                    binarized_x[X[:,i]<t, idx] = 1.
                    binarized_x[X[:,i]>=t, idx] = 0.
                idx += 1
        # dammy variable. it behaves as bias term.
        binarized_x[:, self.bin_dim-1] = 1.
        return binarized_x


    def get_bound_of_coef(self):
        """it returns the bounds of the coefficients of the model.

        Return:
            bounds List[Tuple(min, max)]
        """
        # make bounds of coefficients
        bounds = [None] * self.bin_dim
        idx = 0
        for i, (thresholds, increasing) in enumerate(zip(self.bin_thresholds, self.is_increasing)):
            for j, t in enumerate(thresholds):
                if (increasing and j==0) or (not increasing and j==len(thresholds)-1):
                    bounds[idx] = (None, None)
                else:
                    bounds[idx] = (0, None)
                idx += 1
        bounds[self.bin_dim-1] = (None, None)
        return bounds


    def set_binarization_params(self, X, y):
        if self.bin_thresholds is None:
            tree = Tree(X.shape[1])# segment_space(X, y, K, metrics=self.uncertainry_metrics)
            tree.fit(X, y, self.max_seg)
            self.bin_thresholds = tree.bin_thresholds
            self.bin_dim =  sum(map(len, self.bin_thresholds)) + 1

            
    def check_increasing(self, X, y):
        """it sets self.is_increasing
        Args:
            X: 2d-array. training instances.
            y: 1d-array. labels
        """
        dim = X.shape[1]
        self.is_increasing = [None] * dim
        for d in range(dim):
            # here, we use spearman's rank correlation
            self.is_increasing[d] = check_increasing(X[:, d], y)


    def fit(self, X, y):
        """it calculate weight (a.k.a. importance) vector with X and y.

        Args:
            X: 2d-array.
            y: labels. 1d-array
        """

        # error check
        if len(X.shape) != 2:
            raise ValueError(f"X must be 2d-array")
        
        if np.unique(y).shape[0] != 2:
            raise ValueError(f"y must just contain 2 class labels.")
        
        # check increasity (or decreasity) of each attribute
        self.check_increasing(X, y)
        
        # it sets self.bin_thresholds, self.bin_dim, and, self.is_increasing
        self.set_binarization_params(X, y)
        
        # binarize X
        binarized_x = self.binarize(X)

        # we use range_sigmoid function to avoid error about floating number
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

        # set bounds of the coefficient to constrain the model to be monotonicity
        bounds = self.get_bound_of_coef()
        
        # minimize the objective function (cross entropy with l2 regularization)
        x0 = np.random.random(binarized_x.shape[1])
        options = {"maxiter": self.max_iter} 
        result = minimize(obj_f_l2, x0=x0, bounds=bounds, jac=jac_f_l2, options=options, method="SLSQP")

        if not result.success:
            # TODO: set an appropriate exception
            print("Subnet training:", end="")
            print(result.message)
        self.coef_ = result.x


    def decision_function(self, X):
        return sigmoid(self.binarize(X) @ self.coef_)


    def predict(self, X):
        return np.where(self.decision_function(X) < 0.5, 0, 1)



class TwoLayerAddNet(object):
    def __init__(self, bin_thresholds):
        self.sub_feature_sets = []
        # list of feature set of children subnetwork
        self.first_networks = []
        # list of 1st subnetworks. len(self.first_networks) == len(sub_feature_set)
        self.second_networks = None
        self.bin_thresholds = bin_thresholds


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
