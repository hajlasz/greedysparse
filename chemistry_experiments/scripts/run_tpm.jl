# run_tpm.jl: TPM on Neon (Slater-determinant basis).
#
# Usage:
#   julia -t N run_tpm.jl NORB NELEC NFRZN EPS ERI.txt HCORE.txt SYMM.txt \
#         K_KEEP OUT.json MAX_ITERS TOL EVAL_TRUE
# Example:
#   julia -t 8 run_tpm.jl 23 10 1 1e-4 eris.txt hcore.txt symm.txt \
#         800000 out.json 500 1e-6 128.709475548738

using Arpack, Combinatorics, DelimitedFiles, LinearAlgebra, Random,
      SparseArrays, StatsBase, JSON, BenchmarkTools, LinearMaps, DataStructures
using DataStructures: RobinDict
using Base.Threads

# DO NOT REORDER THESE
include(joinpath(@__DIR__, "helper_functions", "bitops.jl"))                     # Det{W}, bit ops
include(joinpath(@__DIR__, "helper_functions", "bloom.jl"))                      # BloomFilter
include(joinpath(@__DIR__, "helper_functions", "state.jl"))                      # QCEnv, GreedyState, make_gs_from_S, excitation-target lookups
include(joinpath(@__DIR__, "helper_functions", "hamiltonian.jl"))               # matrix elements, build_hdiag!
include(joinpath(@__DIR__, "helper_functions", "restricted_matvec.jl"))         # mul_A!, mul_A_threaded!
include(joinpath(@__DIR__, "helper_functions", "expanding_matvec.jl"))          # mulA_expand!, topk_from_shards
include(joinpath(@__DIR__, "", "Neon_sparsified_experiment.jl"))
for (i, a) in enumerate(ARGS); println("ARGS[$i]: $a"); end
flush(stdout)
run_experiment(:tpm, parse_config(ARGS))
