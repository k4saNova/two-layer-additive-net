import sys
sys.path.append(".")
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from addnet import SubNet

# constants
N_SAMPLES = 200
SEED = 314

def test_subnet():
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.2)# , random_state=SEED)
    test_X, test_y = make_moons(n_samples=N_SAMPLES, noise=0.2)
    # X, y = make_blobs(n_samples=N_SAMPLES, centers=2, random_state=SEED)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # binarization
    # bin_width = [[0.25, 0.5, 0.75], [0.25, 0.5, 0.75]]
    subnet = SubNet()
    subnet.fit(X, y)
    # print(f"{subnet.bin_thresholds=}")
    # print(f"{subnet.is_increasing=}")

    print("training:")
    cm = confusion_matrix(y, subnet.predict(X))
    print(f"accuracy: {(cm[0,0]+cm[1,1])/cm.sum()}")
    print(cm)

    print("test:")
    cm = confusion_matrix(test_y, subnet.predict(test_X))
    print(f"accuracy: {(cm[0,0]+cm[1,1])/cm.sum()}")
    print(cm)
    
    
if __name__ == '__main__':
    test_subnet()
    
