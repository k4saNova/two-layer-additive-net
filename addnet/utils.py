import numpy as np

def sigmoid(x):
    """it returns value of sigmoid function
    
    Arg:
        x: float or array
    
    Return:
        sig(x): sig(x) = 1/(1+exp(-x))
    """
    return 1. / (1. + np.exp(-x))


def get_ranged_sigmoid(x, eps=1e-15):
    abs_max = np.log((1-eps)/eps)
    def ranged_sigmoid(x):
        return sigmoid(np.clip(x, -abs_max, abs_max))

    return ranged_sigmoid
