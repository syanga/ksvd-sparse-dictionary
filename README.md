# ksvd-sparse-dictionary
Learn atoms of a sparse dictionary using the iterative K-SVD algorithm. Useful for compressive sensing applications.

# Example Result:
![Example Reconstruction](/output/true_vs_reconstruct.png)

Example generated using randomly generated sinusoids. See test.py.
* Dictionary size  : 60
* Max sparsity     : 15
* Signal length    : 50  
* (Random) Samples : 25
* Compression ratio: 2:1

# Example Usage:
```
# learn dictionary D from data
dictionary,_,_ = ksvd(data, dictionary_size, max_sparsity, maxiter=max_iter)

# sense test data using sensing matrix
representation = sense(test_data, sensing_matrix)

# sparse reconstruction using learned dictionary
reconstruction = reconstruct(representation, sensing_matrix, dictionary, max_sparsity)
```
# References:
* M. Aharon, M. Elad and A. Bruckstein, "K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation," in IEEE Transactions on Signal Processing, vol. 54, no. 11, pp. 4311-4322, Nov. 2006.

* Rubinstein, R., Zibulevsky, M. and Elad, M., "Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit Technical Report" - CS Technion, April 2008.
