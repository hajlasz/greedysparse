# run_greedy.jl: Sparsified Greedy on Neon (Slater-determinant basis).
#
# Usage:
#   julia -t N run_greedy.jl NORB NELEC NFRZN EPS ERI.txt HCORE.txt SYMM.txt \
#         K_KEEP OUT.json MAX_ITERS TOL EVAL_TRUE
# Example:
#   julia -t 8 run_greedy.jl 23 10 1 1e-4 eris.txt hcore.txt symm.txt \
#         800000 out.json 500 1e-6 128.709475548738

using Arpack, Combinatorics, DelimitedFiles, LinearAlgebra, Random,
      SparseArrays, StatsBase, JSON, BenchmarkTools, LinearMaps, DataStructures
using DataStructures: RobinDict
using Base.Threads

# DO NOT REORDER THESE
include(joinpath(@__DIR__, "helper_functions", "bitops.jl"))                     
include(joinpath(@__DIR__, "helper_functions", "bloom.jl"))                      
include(joinpath(@__DIR__, "helper_functions", "state.jl"))                      
include(joinpath(@__DIR__, "helper_functions", "hamiltonian.jl"))               
include(joinpath(@__DIR__, "helper_functions", "restricted_matvec.jl"))         
include(joinpath(@__DIR__, "helper_functions", "expanding_matvec.jl"))          
include(joinpath(@__DIR__, "", "Neon_sparsified_experiment.jl"))

for (i, a) in enumerate(ARGS); println("ARGS[$i]: $a"); end
flush(stdout)
run_experiment(:greedy, parse_config(ARGS))
