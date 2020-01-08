import sys
sys.path.append(".")
sys.path.append("..")

import pickle
import numpy as np 
from enum import Enum
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from addnet import SubNet
from addnet.utils import Tree

class ClfType(Enum):
    SubNet = "SN   "
    Tree   = "Tree "
    DT     = "DT   "
    SVC    = "SVC  "
    RF     = "RF   "
    LR     = "LR   "

    
SEED = 314
def main():
    iris = load_iris()
    x = iris.data
    y = iris.target

    # drop a class label (2 is dropped)
    dropped_label = 0
    x = x[y!=dropped_label, :]
    y = y[y!=dropped_label]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)

    # scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # following is result
    result = {}
    for ctype in ClfType:
        result[ctype.name] = {}

    def set_result(k, clf, train_acc, train_cm, test_acc, test_cm):
        result[k]["clf"] = clf
        result[k]["train:accuracy"] = train_acc
        result[k]["train:confusion_matrix"] = train_cm
        result[k]["test:accuracy"] = test_acc
        result[k]["test:cm"] = test_cm

        
    ### SUBNET
    # bin_thresholds = [[0.5] for _ in range(4)]
    clf = SubNet(max_seg=30)# , binarization=bin_thresholds)
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    set_result(ClfType.SubNet.name, clf,
               accuracy_score(y_train, y_train_pred), confusion_matrix(y_train, y_train_pred),
               accuracy_score(y_test, y_test_pred), confusion_matrix(y_test, y_test_pred))

    
    ### Tree (tree for segmentation)
    clf = Tree(x_train.shape[1])
    clf.fit(x_train, y_train, K=10)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    set_result(ClfType.Tree.name, clf,
               accuracy_score(y_train, y_train_pred), confusion_matrix(y_train, y_train_pred),
               accuracy_score(y_test, y_test_pred), confusion_matrix(y_test, y_test_pred))
    
    
    ### DT
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    set_result(ClfType.DT.name, clf,
               accuracy_score(y_train, y_train_pred), confusion_matrix(y_train, y_train_pred),
               accuracy_score(y_test, y_test_pred), confusion_matrix(y_test, y_test_pred))
    
    ### SVC
    clf = SVC()
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    set_result(ClfType.SVC.name, clf,
               accuracy_score(y_train, y_train_pred), confusion_matrix(y_train, y_train_pred),
               accuracy_score(y_test, y_test_pred), confusion_matrix(y_test, y_test_pred))
    
    
    ### RF
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    set_result(ClfType.RF.name, clf,
               accuracy_score(y_train, y_train_pred), confusion_matrix(y_train, y_train_pred),
               accuracy_score(y_test, y_test_pred), confusion_matrix(y_test, y_test_pred))
     
    
    ### LR
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    set_result(ClfType.LR.name, clf,
               accuracy_score(y_train, y_train_pred), confusion_matrix(y_train, y_train_pred),
               accuracy_score(y_test, y_test_pred), confusion_matrix(y_test, y_test_pred))

    
    with open("result.pkl", "wb") as f:
        pickle.dump(result, f)

    print(" --- SUBNET --- ")
    clf = result[ClfType.SubNet.name]["clf"]
    
    print(f"{clf.bin_thresholds=}")
    print(f"{clf.bin_dim=}")
    print("\n")
    
    print(" --- ACCURACY --- ")
    # write a markdown table
    l1 = " | ".join([f"{ctype.value}" for ctype in ClfType])
    l2 = " | ".join([f"{result[ctype.name]['train:accuracy']:.3f}"
                     for ctype in ClfType])
    l3 = " | ".join([f"{result[ctype.name]['test:accuracy']:.3f}"
                     for ctype in ClfType])
    print("|       | " + l1 + " | ")
    print("| train | " + l2 + " | ")
    print("| test  | " + l3 + " | ")


if __name__ == '__main__':
    main()
