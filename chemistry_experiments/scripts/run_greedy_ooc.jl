# run_greedy_ooc.jl: Greedy method in the spin-coupled basis with the
# out-of-core (OOC) drain 
#
# Usage:
#   julia -t N run_greedy_ooc.jl \
#       NORB_TOT NELEC_TOT NORB_FRZN EPSILON \
#       ERI_FILE HCORE_FILE SYMM_FILE \
#       M BATCH_SIZE Q_MAX OUTPUT_JSON SPILL_DIR \
#       [RESUME=false]
#
# When RESUME is "true" (case-insensitive), the run will attempt to load a
# checkpoint at the canonical location:
#    <results_root>/checkpoints/<method>__<jsonbasename>.ckpt
# and continue from where the previous run stopped.
# (Submitted on the cluster via ozone/sbatch/greedy_ooc.sbatch.)

using Arpack
using Combinatorics
using DelimitedFiles
using LinearAlgebra
using Random
using SparseArrays
using StatsBase
using JSON
using LinearMaps
using DataStructures
using DataStructures: RobinDict
using Base.Threads
using Printf

# DO NOT REORDER THESE
# --- reused primitives + buffered-emission utilities (greedysparse Slater-determinant helpers) ---
include(joinpath(@__DIR__, "helper_functions", "bitops.jl"))            
include(joinpath(@__DIR__, "helper_functions", "bloom.jl"))             
include(joinpath(@__DIR__, "helper_functions", "state.jl"))             
include(joinpath(@__DIR__, "helper_functions", "hamiltonian.jl"))       
include(joinpath(@__DIR__, "helper_functions", "expanding_matvec.jl"))  
# --- spin-coupled, out-of-core machinery ---
include(joinpath(@__DIR__, "spin_coupled", "spin_coupled_helpers.jl"))          
include(joinpath(@__DIR__, "spin_coupled", "coupled_expanding_matvec.jl"))      
include(joinpath(@__DIR__, "spin_coupled", "batched_expand_and_select_ooc.jl")) 
include(joinpath(@__DIR__, "spin_coupled", "checkpoint.jl"))                    
include(joinpath(@__DIR__, "spin_coupled", "run_greedy_coupled_ooc.jl"))        
include(joinpath(@__DIR__, "helper_functions", "coupled_env.jl"))               
# --- shared config/env/dispatch ---
include(joinpath(@__DIR__, "coupled_ooc_experiment.jl"))

for (i, a) in enumerate(ARGS); println("ARGS[$i]: $a"); end
println("JULIA_NUM_THREADS = ", Threads.nthreads())
flush(stdout)

(12 <= length(ARGS) <= 13) || error("Expected 12–13 ARGS, got $(length(ARGS))")

# arguments are parsed by parse_coupled_ooc_config.
resume = length(ARGS) >= 13 && lowercase(ARGS[13]) == "true"

run_coupled_ooc_experiment(:greedy, parse_coupled_ooc_config(ARGS); resume = resume)