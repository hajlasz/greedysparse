# greedysparse
Code for the paper "Dimension-independent eigenvalue approximation for sparse matrices" 

[arxiv preprint](https://arxiv.org/pdf/2610.14620)

In the code, "greedy" refers to the submatrix selection method.

### Suitesparse Experiments
[Suitesparse Experiments info](https://github.com/hajlasz/greedysparse/blob/main/suitesparse/README_SUITESPARSE.md)


### Chemistry Experiments
Neon:
* input files: [Neon Input Files](https://github.com/hajlasz/greedysparse/tree/main/chemistry_experiments/Neon)
* experiments run with [Neon script](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/Neon_sparsified_experiment.jl)

Ozone:
* input files: run [ozone input files script](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/ozone_new_inputs/input_data/get_ozone_inputs_gsno.py)
* ozone benchmark run with [run_asci_pt2_ooc_eps2](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/run_asci_pt2_ooc_eps2.jl)
* experiments run with [run_tpm_ooc](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/run_tpm_ooc.jl), [run_greedy_ooc](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/run_greedy_ooc.jl), and [run_asci](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/run_asci.jl)
