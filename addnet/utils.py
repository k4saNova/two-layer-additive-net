import numpy as np

def sigmoid(x):
    """it returns value of sigmoid function

    Arg:
        x: float or array

    Return:
        sig(x): sig(x) = 1/(1+exp(-x))
    """
    return 1. / (1. + np.exp(-x))


def get_ranged_sigmoid(eps=1e-15):
    """it returns ranged_sigmoid function.
    it is to avoid errors about floating numbers

    Arg:
        eps: minimum number (default: 1e-15)

    Return:
        ranged_sigmoid: it is sigmoid function.
                        the range is x \in [-M, M], M=log((1-eps)/eps)
    """
    abs_max = np.log((1-eps)/eps)
    def ranged_sigmoid(x):
        return sigmoid(np.clip(x, -abs_max, abs_max))
    return ranged_sigmoid


def get_uncertainry_metrics(labels, metrics):
    """it returns a function that computes the criteria like gini impurity.

    """
    class_labels = np.unique(labels)

    def gini_impurity(y):
        return 1. - sum(y[y==cls].shape[0]**2 for cls in class_labels)/y.shape[0]**2

    return gini_impurity


class Tree(dict):
    # id of root node
    root_id = "T"

    def __init__(self, dim):
        super().__init__()
        self.unode_id = set() # updatable node
        self.dim = dim
        self.bin_thresholds = [[] for _ in range(self.dim)]
        self.num_region = 1 # number of segmentation. it is NOT # of leafs.


    def __setitem__(self, k, v):
        self.unode_id.add(k) # atarashii node ha updatable
        super().__setitem__(k, v)


    def split_node(self):
        """ it splits a node if possible
        """
        # step1 choose a node
        parent_id =  max(self.unode_id, key=lambda nk: self[nk].criteria)
        self.unode_id.remove(parent_id)

        # step2 find good condition
        d, threshold, gain = self[parent_id].split_self()

        if gain > 0:
            self[parent_id].cond = (d, threshold)
            # update bin_thresholds and num_region
            self.bin_thresholds[d].append(threshold)
            self.bin_thresholds[d].sort()
            ld = len(self.bin_thresholds[d]) + 1
            self.num_region = self.num_region*ld//(ld-1)

            # make left and right child node information
            left_id, right_id = self[parent_id].generate_children_id()
            left_indexes  = self[parent_id].indexes & (Node.X[:, d] <= threshold)
            right_indexes = self[parent_id].indexes & np.logical_not(left_indexes)

            # split node
            self[left_id]  = Node(left_id, left_indexes, self[parent_id])
            self[right_id] = Node(right_id, right_indexes, self[parent_id])


    def predict(self, X):
        y = np.empty(X.shape[0])
        for idx, x in enumerate(X):
            node = self[Tree.root_id]
            while node.has_children():
                d, t = node.cond
                if x[d] <= t:
                    # go to left node
                    node = self[node.children_id[0]]
                else:
                    # go to right node
                    node = self[node.children_id[1]]
            y[idx] = node.majority
        return y


class Node(object):
    f_criteria = None
    X = None
    y = None
    def __init__(self, ID, indexes, parent_node):
        self.ID = ID
        self.indexes = indexes

        # set node_id and depth
        if parent_node is None:
            self.parent_id = None
            self.depth = 0
        else:
            self.parent_id = parent_node.ID
            self.depth = parent_node.depth + 1

        # uncertainry criteria of this node
        self.criteria = Node.f_criteria(Node.y[indexes])
        # # of instances in this node
        self.num_support = Node.y[indexes].shape[0]
        # dominant class label
        self.majority = max(((cls, Node.y[(indexes) & (Node.y==cls)].shape[0])
                             for cls in np.unique(Node.y)),
                            key=lambda x: x[1])[0]
        self.children_id = None # (left_id, right_id)
        self.cond = None # (d, threshold)


    def __repr__(self):
        repr_id = f"ID: {self.ID}\n"
        if self.cond is not None:
            d, t = self.cond
            repr_cond = "\n".join([
                f"if x[{d}] <= {t:.3f}: {self.children_id[0]}",
                f"else: {self.children_id[1]}",
                f"\n"])
        else:
            repr_cond = ""

        repr_class = "\n".join([
            f"support ratio = {self.num_support}/{Node.X.shape[0]}",
            f"majority class = {self.majority}"])
        return repr_id + repr_cond + repr_class


    def split_self(self, max_iter=100):
        num_instance, dim = Node.X.shape

        def minimize_criteria(x, y):
            vmin, vmax = x.min(), x.max()
            original_criteria = Node.f_criteria(y)
            current_criteria = original_criteria
            threshold = (vmin + vmax) / 2.
            for it in range(max_iter):
                new_threshold = (vmin + vmax) / 2.

                # left side from new_threshold
                # vmin <---here---> t ------ vmax
                left_cond = x <= new_threshold
                num_left = x[left_cond].shape[0]
                left_criteria = Node.f_criteria(y[left_cond])

                # right side from new_threshold
                # vmin ------ t <---here---> vmax
                right_cond =  new_threshold < x
                num_right = x[right_cond].shape[0]
                right_criteria = Node.f_criteria(y[right_cond])

                new_criteria = (num_left/num_instance)*left_criteria \
                               + (num_right/num_instance)*right_criteria

                if current_criteria <= new_criteria:
                    break

                # update threshold and current_criteria
                threshold, current_criteria = new_threshold, new_criteria
                if left_criteria < right_criteria:
                    vmax = new_threshold
                elif left_criteria > right_criteria:
                    vmin = new_threshold
                else: # left_criteria == right_criteria
                    if num_left < num_right:
                        vmin = new_threshold
                    else:
                        vmax = new_threshold
            else: # this is for-else statement
                print(f"it reached {max_iter} iterations!!")

            # return threshold and gain
            return threshold, original_criteria - current_criteria

        # kokode zen zigen de split wo tamesite sairyouno mono wo kaesu
        best_d, best_threshold, best_gain = 0, 0, 0
        for d in range(dim):
            t, g = minimize_criteria(Node.X[self.indexes, d], Node.y[self.indexes])
            if g > best_gain:
                best_d, best_threshold, best_gain = d, t, g
        return best_d, best_threshold, best_gain


    def generate_children_id(self):
        self.children_id = (f"{self.ID}:L", f"{self.ID}:R")
        return self.children_id


    def has_children(self):
        return self.children_id is not None


def segment_space(X, y, K, metrics="gini"):
    """ it returns a tree that segments the feature space.

    Args:
        X: training instances.
        y: labels.
        metrics (optional): uncertainry criteria (defualt: gini)
        K (optinal): # of segmentation

    Returns:
        tree: segmentation tree
    """
    # initialize Node class variables
    Node.X = X
    Node.y = y
    Node.f_criteria = get_uncertainry_metrics(y, metrics)

    # make tree
    tree = Tree(dim=X.shape[1]) # node_id-node pairs
    tree[Tree.root_id] = Node(Tree.root_id, np.ones(X.shape[0], dtype=bool), None)

    while tree.num_region < K and len(tree.unode_id) > 0:
        tree.split_node()

    return tree
