import numpy as np
from warnings import warn
from numpy.linalg import norm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import check_increasing
from sklearn.preprocessing import LabelEncoder
from addnet.utils import *

class SubNet(object):    
    def __init__(self, max_seg=16, bin_thresholds=None, check_increasing_type="sp", **kawgs):
        """SubNet initializer
        Args:
            max_seg:
            bin_thresholds:
            check_increasing_type (optional): "lr": check with LogisticRegression (default)
                                              "sp": check with Spearman correlation coefficient
        """
        # a coefficient vector of the model
        self.coef_ = None

        # parameters of segmentation
        self.max_seg = max_seg
        if bin_thresholds is not None: # 
            self.bin_thresholds = binarization
            self.bin_dim = sum(map(len, self.bin_thresholds)) + 1
        else:
            self.bin_thresholds = None
            self.bin_dim = None
        self.check_increasing_type = check_increasing_type 
            
        # parameters of optimization 
        self.max_iter = 1000
        self.method = "SLSQP"
        
        # regularization parameter of logistic regression {0, 1}^d -> {0, 1}
        self.regularization_param = 1.
        self.uncertainry_metrics = "gini"
        
        # parameters of utilities
        self.label_encoder = None

        
    def binarize(self, X):
        """it returns binarized X.
        Args:
            X: array-like of shape (n_samples, n_features)
        """
        if len(X.shape) != 2:
            raise ValueError(f"X.shape must be (n_samples, n_features)")
        
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
        """it sets self.bin_thresholds and self.bin_dim with utils.Tree
        """
        if self.bin_thresholds is None:
            tree = Tree(X.shape[1], K=self.max_seg)
            tree.fit(X, y)
            self.bin_thresholds = tree.bin_thresholds
            self.bin_dim =  sum(map(len, self.bin_thresholds)) + 1
            
            
    def check_increasing(self, X, y):
        """it sets self.is_increasing
        Args:
            X: 2d-array. training instances.
            y: 1d-array. labels
        """
        if self.check_increasing_type == "sp":
            dim = X.shape[1]
            self.is_increasing = [None] * dim
            for d in range(dim):
                # here, we use spearman's rank correlation
                self.is_increasing[d] = check_increasing(X[:, d], y)

        elif self.check_increasing_type == "lr":
            clf = LogisticRegression()
            clf.fit(X, y)
            w = clf.coef_.flatten()
            self.is_increasing = (w >= 0).tolist()

        else:
            self.is_increasing = [True] * dim

        
    def fit(self, X, y):
        """it calculate weight (a.k.a. importance) vector with X and y.

        Args:
            X: 2d-array.
            y: labels. 1d-array
        """

        # error check
        if len(X.shape) != 2:
            raise ValueError(f"X must be 2d-array")

        labels = np.unique(y)
        if labels.shape[0] != 2:
            raise ValueError(f"y must just contain 2 class labels.")

        # if not ((0 in labels) and (1 in labels)):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            
            
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
        result = minimize(obj_f_l2, x0=x0, bounds=bounds, jac=jac_f_l2,
                          options={"maxiter": self.max_iter}, method=self.method)

        if not result.success:
            warn(f"Warning in SubNet.fit(): {result.message}")
       
        # set result
        self.coef_ = result.x
        

    def decision_function(self, X):
        return sigmoid(self.binarize(X).dot(self.coef_))
    
    
    def predict(self, X):
        """it returns predicted class labels for samples in X.

        Args: 
            X: array-like of shape (n_samples, n_features)
        
        Returns:
            y: array-like of shape (n_samples, )
        """
        y_pred = np.where(self.decision_function(X) < 0.5, 0, 1)
        return self.label_encoder.inverse_transform(y_pred)



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
