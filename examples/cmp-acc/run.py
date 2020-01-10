import sys
sys.path.append(".")

import pickle
import numpy as np 
from enum import Enum
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from addnet import SubNet
from addnet.utils import Tree


class ClfType(Enum):
    SubNet = "SN   "
    Tree   = "SNTR "
    DT     = "DT   "
    SVC    = "SVC  "
    RF     = "RF   "
    LR     = "LR   "

    
class DataSet(Enum):
    Iris = 0
    BreastCancer = 1

    
# Constants    
SEED = 314
RESULT_PATH = "examples/cmp-acc/result.pkl"
FEATURE_RANGE = (0, 1) # regularizatioin


def load_dataset(data_idx):
    if data_idx == DataSet.Iris.value:
        iris = load_iris()
        x = iris.data
        y = iris.target
        
        # drop a class label because iris has 3 classes
        dropped_label = 0 # 0, 1, 2
        x = x[y!=dropped_label, :]
        y = y[y!=dropped_label]
        return x, y
    
    elif data_idx == DataSet.BreastCancer.value:
        bc = load_breast_cancer()
        return bc.data, bc.target
        
        
def main(data_idx=0):
    # load dataset
    x, y = load_dataset(data_idx)
    
    # data split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
    
    # scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # following is result
    result = {}
    for ctype in ClfType:
        result[ctype.name] = {}
    
    def measure_accuracy(clf, name):
        clf.fit(x_train, y_train)
        y_train_pred = clf.predict(x_train)
        y_test_pred = clf.predict(x_test)
        # store the results
        result[name]["clf"] = clf
        result[name]["train:accuracy"] = accuracy_score(y_train, y_train_pred)
        result[name]["train:confusion_matrix"] = confusion_matrix(y_train, y_train_pred)
        result[name]["test:accuracy"] = accuracy_score(y_test, y_test_pred)
        result[name]["test:cm"] = confusion_matrix(y_test, y_test_pred)

        
    ### SUBNET
    bin_thresholds = [[0.25, 0.5, 0.75] for _ in range(x_train.shape[1])]
    max_grids = [4]*x_train.shape[1]
    clf = SubNet(binarize_type="auto", check_increasing_type="lr")
                 # dot_path="examples/cmp-acc/tree.dot")#, bin_thresholds=bin_thresholds)
    measure_accuracy(clf, ClfType.SubNet.name)
    
    ### SNTR (tree for segmentation)
    clf = Tree(x_train.shape[1], K=10)
    measure_accuracy(clf, ClfType.Tree.name)
    
    ### DT
    clf = DecisionTreeClassifier()
    measure_accuracy(clf, ClfType.DT.name)
    
    ### SVC
    clf = SVC()
    measure_accuracy(clf, ClfType.SVC.name)
    
    
    ### RF
    clf = RandomForestClassifier()
    measure_accuracy(clf, ClfType.RF.name)
    
    ### LR
    clf = LogisticRegression()
    measure_accuracy(clf, ClfType.LR.name)

    with open(RESULT_PATH, "wb") as f:
        pickle.dump(result, f)

    print(" --- DATASET --- ")
    print(f"{x_train.shape=}, {y_train.shape=}")
    print(f"{x_test.shape=}, {y_test.shape=}")
    print("")
        
    print(" --- SUBNET --- ")
    clf = result[ClfType.SubNet.name]["clf"]
    print(f"{clf.bin_thresholds=}")
    print(f"{clf.bin_dim=}")
    print(f"{clf.is_increasing=}")
    print(f"{clf.get_bound_of_coef()=}")
    print(f"{clf.coef_=}")
    print("")
    
    print(" --- ACCURACY --- ")
    # write a table
    l1 = " | ".join([f"{ctype.value}" for ctype in ClfType])
    l2 = " | ".join([f"{result[ctype.name]['train:accuracy']:.3f}"
                     for ctype in ClfType])
    l3 = " | ".join([f"{result[ctype.name]['test:accuracy']:.3f}"
                     for ctype in ClfType])
    print("|       | " + l1 + " | ")
    print("| train | " + l2 + " | ")
    print("| test  | " + l3 + " | ")

    

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else: # len(sys.argv) > 1
        arg = int(sys.argv[1])
        if arg in ["--help", "-H"]:
            print("# TODO: WRITE HELP MESSAGE")
        elif arg in [data.value for data in DataSet]:
            main(arg)
        else:
            print(f"arg must be one of {[data.value for data in DataSet]}")
        
