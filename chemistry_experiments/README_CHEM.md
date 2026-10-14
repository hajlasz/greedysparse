# Chemistry Experiments

## Getting Input Files

I used [PYSCF](https://pyscf.org/) to generate input files for the chemsitry experiments:
* `hcore.txt`: one-electron integrals ($r$ is position):
  
  $hcore(i,j) = \langle i\vert h\vert j\rangle= \int \phi_i^\ast (r)\hat{h}(r) \phi_j(r) d^3r$
  
* `eris.txt`: two-electron integrals in  physicists' notation, where the first two indices correspond to one of the states and the next two indices correspond to the other state:
  
  $eris(i,j,k,l) = \langle ij\vert kl\rangle = \int \int \phi^\ast_i(r_1)\phi_j^\ast(r_2) \frac{1}{\vert r_1-r_2\vert}\phi_k(r_1)\phi_l(r_2)\ d^3r_1\ d^3r_2$

* `symm.txt`: symmetry labels for each orbital (if orbital i and orbital j are occupied, the symmetry is given by symm(i) ^ symm(j) )

An example script generating these files can be found [here](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/Neon/input_data/get_Neon_inputs.py).

## Matrix from Input Files

### Neon 

The Neon experiments used the Slater Determinant basis, in which each index corresponds to one state of electrons distributed among orbitals. The matrix elements are given below ($\sigma$ denotes the spin, $\nu$ is parity of the excitation), and the code for computing them can be found [here](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/helper_functions/hamiltonian.jl).
* Diagonal entries:

$\langle I\vert \hat{H}\vert I\rangle= \sum_{i\in \vert I\rangle } \langle i\vert \hat{h}\vert i\rangle + \frac{1}{2}\sum_{i\in \vert I\rangle } \sum_{j\in \vert I\rangle } \biggr[\langle ij\vert ij\rangle - \delta_{\sigma_i \sigma_j} \langle ij\vert ji\rangle\biggr]$
  
* Single excitations ($i\to j$):

$\langle I\vert \hat{H}\vert J\rangle = (-1)^\nu \biggr[\delta_{\sigma_i\sigma_j} \langle i\vert \hat{h}\vert j\rangle  + \sum_{k\in \vert I\rangle \cap \vert J\rangle }\delta_{\sigma_i\sigma_j } \langle ik\vert jk\rangle -\delta_{\sigma_i \sigma_k }\delta_{\sigma_k\sigma_j} \langle ik\vert kj\rangle\biggr] $

* Double excitations ($ij\to k\ell$):

$\langle I\vert \hat{H}\vert J\rangle = (-1)^\nu \biggr[\delta_{\sigma_i \sigma_k} \delta_{\sigma_j\sigma_\ell }\langle ij\vert k\ell\rangle - \delta_{\sigma_i\sigma_\ell }\delta_{\sigma_j\sigma_k }\langle ij\vert \ell k\rangle\biggr]$
  
* Triple+ excitations:

$\langle I\vert \hat{H}\vert J\rangle = 0$

### Ozone
The ozone experiments use the **spin-flip symmetrized basis**, built on top of the Slater-determinant basis above. Let $T$ be the spin-flip operation on determinants: exchange the occupations of all up-spin and down-spin orbitals. In the code, a determinant is a $2n$-bit string with up spins in bits $1..n$ and down spins in bits $(n+1)..2n$.

The basis vectors are, for each pair $\{d,\ Td\}$,

$\vert \tilde{d}\rangle = \frac{1}{\sqrt{2}}\bigl(\vert d\rangle + \vert Td\rangle\bigr)$ if $Td\neq d$, and $\vert \tilde{d}\rangle = \vert d\rangle$ if $Td = d$ ("T-fixed").

The code stores one **representative** per pair — the binary minimum of $d$ and $Td$ (`rep_det`).

Matrix elements in the coupled basis are combinations of the Slater-determinant elements given above. Writing $n_I = 1$ if state $I$ is $T$-fixed and $n_I = 2$ otherwise, expanding the definitions and using $[H,T]=0$ gives

$$\langle \tilde{d}_I\vert \hat{H}\vert \tilde{d}_J\rangle = \sqrt{\tfrac{n_I}{n_J}} \sum_{d' \in \{d_J, Td_J\}} \langle d_I\vert \hat{H}\vert d'\rangle$$

All of this is in [spin_coupled_helpers.jl](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/spin_coupled/spin_coupled_helpers.jl).

## Code Setup for Eigenvalue Calculations

### Neon

The code works with the shifted operator $A = \sigma I - \varepsilon H$, finding the largest eval (:LR for ARPACK) if $\varepsilon  >0$ and the smallest eval (:SR for ARPACK) if $\varepsilon < 0$.

The iteration starts from a single determinant (the Hartree–Fock determinant). Each iteration has three steps:

1. **Expand** — apply $A$ to the current vector. $A$ is never stored as a matrix; applying it spreads weight from each determinant out to itself and to every determinant reachable by a single or double excitation.
2. **Sparsify** — keep only a fixed number $k$ of the resulting determinants and discard the rest
3. **Estimate** — find the leading eigenpair of $A$ restricted to the bag. As in the expand step this is matrix-free: $A_{S,S}$ is never assembled. The leading eigenvalue gives the current energy estimate and its eigenvector becomes the starting vector for the next iteration.

The shared driver for all the methods is [Neon_sparsified_experiment.jl](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/Neon_sparsified_experiment.jl). Different methods are obtained by swapping the second and third steps (how the bag is chosen, and how the eigenvalue is estimated). This driver works for any system small enough that the dictionary holding the support of $Ax$ fits in RAM.

* `mulA_expand!` (*expand*): given a vector $x$, computes the full, unrestricted product $Ax$ over every reachable determinant, returning the result together with its **support**. The product is accumulated in parallel across threads into *sharded dictionaries*. [expanding_matvec.jl](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/helper_functions/expanding_matvec.jl).
* `sparsify` (*sparsify*): takes the accumulated $Ax$ from `mulA_expand!` and returns the kept (sparsified) vector and its support, according to the method's rule. The greedy and TPM methods keep the $k$ largest-magnitude entries (`topk_from_shards`). These can be found in [expanding_matvec.jl](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/helper_functions/expanding_matvec.jl).
* `estimate_step` (*estimate*): on the support produced by `sparsify`, returns both the energy estimate and the iterate carried into the next expansion. As in the expand step, $A$ restricted to the bag is never assembled as a matrix — it is wrapped as a matrix-free `LinearMap` that applies $A_{S,S}$ on demand via the restricted matrix–vector product in [restricted_matvec.jl](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/helper_functions/restricted_matvec.jl). For the greedy method, ARPACK's `eigs` finds the leading eigenpair of this operator using only matrix–vector products; for TPM, the truncated $Ax$ is normalized and its Rayleigh quotient is taken.

### Ozone

Greedy and TPM use the same expand–sparsify–estimate structure as Neon, with two changes: the spin-flip-symmetrized basis and an **out-of-core expansion**, which stores intermediate contributions on disk because the full expanded vector is too large for RAM. The code works with the shifted operator $A = \sigma I - \varepsilon H$.

The shared driver is in [coupled_ooc_experiment.jl](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/coupled_ooc_experiment.jl). A fresh run starts from the Hartree–Fock determinant; saved checkpoints allow interrupted runs to resume.

Each iteration has three steps:

1. **Expand** — apply $A$ to batches of source states. Contributions are accumulated in memory, mapped to their spin-flip representatives, and written to sorted files on disk.
2. **Sparsify** — merge the files, sum all contributions to each state, and retain at most $m$ entries of largest magnitude. Together, these steps are implemented by `batched_expand_merge_select_ooc` in [batched_expand_and_select_ooc.jl](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/spin_coupled/batched_expand_and_select_ooc.jl). There is no pruning before selection, so using disk introduces no additional truncation beyond the final top-\(m\) step.
3. **Estimate** — the [greedy method](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/spin_coupled/run_greedy_coupled_ooc.jl) solves the restricted eigenproblem with ARPACK, using the normalized truncated vector as its initial guess. This implementation uses `which=:LR`, corresponding to the ground state when $\varepsilon>0$. The [TPM method](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/spin_coupled/run_tpm_coupled_ooc.jl) instead evaluates the Rayleigh quotient of the normalized truncated iterate.

Both methods apply $A_{S,S}$ through the matrix-free function `dual_indexed_mul_A_threaded!`.

### Ozone — ASCI

ASCI (Adaptive Sampling Configuration Interaction) uses the same symmetrized basis and disk-based accumulation. It has two size parameters: `N_cdets` (number of **core determinants**, i.e. $m_1$ in the paper), and `N_tdets` ($m_2$ in the paper). We truncate to `N_cdets` determinants before the mat-vec and keep `N_tdets` for the submatrix selection.

The iteration loop is in [run_asci_coupled_ooc.jl](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/asci/run_asci_coupled_ooc.jl). Each iteration has four steps:

1. **Select core determinatns** — choose up to `N_cdets` indices with largest magnitude in current iterate.
2. **Search** — generate allowed single and double excitations from the selected core determinants, accumulate contributions, and rank candidates by approximate first-order coefficients: $s_i=\frac{\sum_{j\in G}H_{ij}v_j}{E-H_{ii}}.$ Here $E$ is the current eigenvalue estimate. Optional `eps_thresh` screening skips small individual contributions.
4. **Merge** — form a space of at most `N_tdets` states. The default `:include` rule retains all of the core determinants and fills the remaining slots with the highest-scoring candidates. The alternative `:rank` rule ranks core determinant coefficients in previous iterate and the candidate scores together. Search and merging are implemented in [asci_search.jl](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/asci/asci_search.jl).
5. **Estimate** — assemble $A_{S,S}$ as an explicit sparse matrix using `build_sparse_A_csc` in [sparse_hamiltonian_csc.jl](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/asci/sparse_hamiltonian_csc.jl), then solve the restricted eigenproblem with ARPACK. The resulting eigenvector becomes the next iterate.

The assembled matrix is reused during the eigensolve and must fit in RAM.

### Ozone — PT2

Second-order Epstein–Nesbet perturbation theory estimates the energy correction from states outside the variational space \(S\):

$$
E_{\mathrm{PT2}}
=\sum_{i\notin S}\frac{|b_i|^2}{E_{\mathrm{var}}-H_{ii}},
\qquad
b_i=\sum_{j\in S}H_{ij}v_j.
$$

The PT2 drivers read a saved checkpoint and report $E_{\mathrm{var}}+E_{\mathrm{PT2}}$. They expand all states in $S$, sum the contributions to each external state, and then evaluate its correction. Implemented in [pt2_core.jl](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/asci/pt2_core.jl).

Three versions:

- [run_asci_pt2.jl](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/run_asci_pt2.jl) writes contributions to disk shards and merges each shard in memory.
- [run_asci_pt2_ooc.jl](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/run_asci_pt2_ooc.jl) uses sorted files, periodic compaction, and a streaming final sum.
- [run_asci_pt2_ooc_eps2.jl](https://github.com/hajlasz/greedysparse/blob/main/chemistry_experiments/scripts/run_asci_pt2_ooc_eps2.jl) adds screening to the compacted implementation.

The first and third variants support an $\varepsilon_2$ cutoff on individual numerator contributions.
