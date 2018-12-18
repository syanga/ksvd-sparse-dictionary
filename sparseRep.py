import numpy as np
from sklearn.linear_model import orthogonal_mp_gram
from tqdm import tqdm

"""
    # sparse representations and compressed sensing
    #
    # raw data -> sensing matrix -> representation
    #
    # representation -> dictionary -> reconstructed data
"""

# generate random sampling matrix. Default binary.
def random_sensor(size, low=0, high=2):
    return np.random.randint(low, high, size=size)

# acquisition using sensing matrix
def sense(data, sensor):
    return np.dot(data,sensor)

# reconstruct
def reconstruct(representation, sensor, dictionary, max_sparsity):
    if representation.shape[0] == 1:
        representation = representation.T
    elif representation.ndim == 1:
        representation = representation.reshape((representation.shape[0], -1))
        
    # reconstruction matrix, gram matrix
    SD = np.dot(sensor.T, dictionary)
    SD_norm = SD / np.linalg.norm(SD, axis=0)
    gram = np.dot(SD_norm.T, SD_norm)

    reconstruction = np.zeros([dictionary.shape[0], representation.shape[1]])
    for col in range(representation.shape[1]):
        w = orthogonal_mp_gram(gram, np.dot(SD_norm.T, representation[:,col]), 
            n_nonzero_coefs=max_sparsity)

        idx = np.nonzero(w)[0]
        w_hat = np.zeros(w.shape[0])
        w_hat[idx] = np.dot(np.linalg.pinv(SD[:,idx]), representation[:,col])
        reconstruction[:,col] = np.dot(dictionary, w_hat)

    return reconstruction
