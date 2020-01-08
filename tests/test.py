import sys
sys.path.append(".")
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from addnet import SubNet

# constants
N_SAMPLES = 200
SEED = 314

def main():
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.2)# , random_state=SEED)
    # X, y = make_blobs(n_samples=N_SAMPLES, centers=2, random_state=SEED)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # print(X)
    
    # binarization
    # bin_width = [[0.25, 0.5, 0.75], [0.25, 0.5, 0.75]]
    subnet = SubNet()
    subnet.fit(X, y)
    print(f"{subnet.bin_thresholds=}")
    print(f"{subnet.is_increasing=}")
    y_pred = subnet.predict(X)
    
    print(confusion_matrix(y, y_pred))

    
if __name__ == '__main__':
    main()
