import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import dft

from ksvd import ksvd
from sparseRep import random_sensor, sense, reconstruct

def generate_sequence(N, signal_length):
    # generate random dictionary
    D = dft(signal_length).real

    # generate random outputs
    y = np.zeros((N,signal_length))
    for i in range(y.shape[0]):
        x = np.random.choice(D.shape[1], int(D.shape[1]/10))
        y[i,:] = np.sum(D[x,:], axis=0)
    return y


if __name__ == "__main__":
    # Generate train data: N samples of white noise signals
    N = 500
    signal_length = 50
    data = generate_sequence(N, signal_length)

    # K-SVD algorithm to learn sparse representation of data
    D_size = 20
    max_sparsity = int(D_size/5)
    max_iter = 50
    D,_,_ = ksvd(data, D_size, max_sparsity, 
        maxiter=max_iter, debug=True)

    # generate test data
    test_data = generate_sequence(1, signal_length)

    # compressive sensing
    sensor = random_sensor((signal_length, D_size))
    representation = sense(test_data, sensor)

    # sparse reconstruction
    reconstruction = reconstruct(representation, sensor, D, max_sparsity)

    plt.plot(test_data[0,:], '--', label="true signal")
    plt.plot(reconstruction, '.-', label="reconstruction")
    plt.title("True vs. reconstructed signal")
    plt.legend()
    plt.show()