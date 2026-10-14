# Suitesparse Experiments

## Getting Matrices from SuiteSparse
To download matrices from SuiteSparse using MATLAB, you first need to download the `ssget` MATLAB interface from [here](https://sparse.tamu.edu/interfaces). 

A script demonstrating how I downloaded and labeled my matrix files can be found [here](https://github.com/hajlasz/greedysparse/blob/main/suitesparse/get_matrices.m). Each matrix is given a number `i`, saved as `i.mat`, with matrices ordered by number of rows/columns (so larger `i` means larger matrix). These numbers are not the SuiteSparse ID numbers.

The test scripts below only consider the real symmetric matrices in the collection, and only those whose `i.mat` file was actually downloaded. 

## Getting Data for CDF Plot (Figure 1)
**Running the Methods**

To run the greedy, Lanczos, and power methods on the downloaded SuiteSparse matrices, use:
* `test_greedy(ind_start, ind_end)` runs the greedy method on `ind_start.mat`,.., `ind_end.mat`
* `test_lanczos(ind_start, ind_end)` runs the Lanczos method on `ind_start.mat`,.., `ind_end.mat`
* `test_power(ind_start, ind_end)` runs the power method on `ind_start.mat`,.., `ind_end.mat`

Each script saves a single struct `M` to `<method_name>_results_<ind_start>_to<ind_end>.mat`. The struct contains:
* `M.filenames`: the file numbers `i` covered by this run (i.e. donwloaded as `i.mat`)
* `M.inds`: corresponding indices in the `ssget` index
* `<method_name>_viewed_<init_cond>` array of size `10 x numel(filenames)`. Entry `(k,c)` is the fraction of nonzeros read, `nnz(A(S,S)) / nnz(A)`, at the moment the relative error of the dominant-eigenvalue estimate first dropped below `10^{(-k)}` for matrix `c`. If that tolerance was never reached, the entry is left as `NaN`. 
* `<method_name>_antistallA_<init_cond>` array of size `10 x numel(filenames)`. Entry `(k,c)` is the fraction of nonzeros read, `nnz(A(S,S)) / nnz(A)`, at the moment the relative error of the dominant-eigenvalue estimate first dropped below `10^{(-k)}` for matrix `c`. If the method stalled before reaching a relative error (ex: stuck on a block orthogonal to the dominant eigenvector), then the fraction of nonzeros read is set to `1`. 

A matrix is skipped (its column stays `NaN`) if the file fails to load, the matrix is diagonal or non-symmetric, or `eigs` fails to converge.

Batches of indices can be submitted in job arrays, such as [here](https://github.com/hajlasz/greedysparse/blob/main/suitesparse/scripts/example_jobarray.sbatch)

**Combining per-batch results**

After all batches finish, run `combine_results()` to merge every `results/<method>/*.mat` file into a single combined struct per method. 

**Plotting**

Plotting code can be found [here](https://github.com/hajlasz/greedysparse/blob/main/suitesparse/plotting/Plotting.ipynb) and [here](https://github.com/hajlasz/greedysparse/blob/main/suitesparse/plotting/frac_viewed.m). 
