# run_asci.jl: ASCI in the spin-coupled basis with an out-of-core search and an
# explicit sparse-CSC eigensolve 
#
#
# Usage:
#   julia -t N run_asci.jl \
#       NORB_TOT NELEC_TOT NORB_FRZN EPSILON \
#       ERI_FILE HCORE_FILE SYMM_FILE \
#       N_CDETS N_TDETS BATCH_SIZE Q_MAX OUTPUT_JSON SPILL_DIR \
#       [MERGE_MODE=include] [EPS_THRESH=0] [ECONV=0] [RESUME=false] [MAX_DISK_GB=3000]
#
# Positional :
#   NORB_TOT NELEC_TOT NORB_FRZN EPSILON   system + shift parameter ε
#   ERI_FILE HCORE_FILE SYMM_FILE          integral / symmetry data files
#   N_CDETS                                number of core determinants
#   N_TDETS                                size of submatrix to diagonalize
#   BATCH_SIZE Q_MAX                       search batch size, max iterations
#   OUTPUT_JSON SPILL_DIR                  results file, scratch dir
#
# Optional (positional, with defaults):
#   MERGE_MODE   include | rank   (default include = MACIS rule)
#   EPS_THRESH   search screen    (default 0.0 = keep all candidates, exact)
#   ECONV        |ΔE| stop in Ha  (default 0.0 = run all Q_MAX iterations)
#   RESUME       true | false     (default false; resumes from canonical ckpt)
#   MAX_DISK_GB  abort if the spill dir exceeds this (default 3000)
#
# When RESUME is "true" (case-insensitive), the run loads the checkpoint at the
# canonical location
#    <results_root>/checkpoints/<method>__<jsonbasename>.ckpt
# and continues from where the previous run stopped.
# (Submitted on the cluster via ozone/sbatch/asci.sbatch.)

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
# --- reused primitives + buffered-emission utilities (Slater-determinant helpers) ---
include(joinpath(@__DIR__, "helper_functions", "bitops.jl"))            
include(joinpath(@__DIR__, "helper_functions", "bloom.jl"))           
include(joinpath(@__DIR__, "helper_functions", "state.jl"))         
include(joinpath(@__DIR__, "helper_functions", "hamiltonian.jl"))    
include(joinpath(@__DIR__, "helper_functions", "expanding_matvec.jl")) 
include(joinpath(@__DIR__, "helper_functions", "coupled_env.jl"))       
# --- spin-coupled, out-of-core machinery ---
include(joinpath(@__DIR__, "spin_coupled", "spin_coupled_helpers.jl"))         
include(joinpath(@__DIR__, "spin_coupled", "coupled_expanding_matvec.jl"))     
include(joinpath(@__DIR__, "spin_coupled", "batched_expand_and_select_ooc.jl")) 
include(joinpath(@__DIR__, "spin_coupled", "checkpoint.jl"))                    
# --- ASCI-specific machinery ---
include(joinpath(@__DIR__, "asci", "pt2_core.jl"))                      
include(joinpath(@__DIR__, "asci", "asci_search.jl"))                   
include(joinpath(@__DIR__, "asci", "sparse_hamiltonian_csc.jl"))        
include(joinpath(@__DIR__, "asci", "run_asci_coupled_ooc.jl"))          
# --- shared config/env/dispatch ---
include(joinpath(@__DIR__, "asci_experiment.jl"))

for (i, a) in enumerate(ARGS); println("ARGS[$i]: $a"); end
println("JULIA_NUM_THREADS = ", Threads.nthreads())
flush(stdout)

(13 <= length(ARGS) <= 18) || error("Expected 13–18 ARGS, got $(length(ARGS))")

# arguments are parsed by parse_asci_config.
merge_mode  = length(ARGS) >= 14 ? Symbol(lowercase(ARGS[14])) : :include
eps_thresh  = length(ARGS) >= 15 ? parse(Float64, ARGS[15]) : 0.0
econv       = length(ARGS) >= 16 ? parse(Float64, ARGS[16]) : 0.0
resume      = length(ARGS) >= 17 && lowercase(ARGS[17]) == "true"
max_disk_gb = length(ARGS) >= 18 ? parse(Float64, ARGS[18]) : 3000.0

run_asci_experiment(:asci, parse_asci_config(ARGS);
                    merge_mode  = merge_mode,
                    eps_thresh  = eps_thresh,
                    econv       = econv,
                    max_disk_gb = max_disk_gb,
                    resume      = resume)
