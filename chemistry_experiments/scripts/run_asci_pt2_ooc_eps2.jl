# run_asci_pt2_ooc_eps2.jl : Epstein-Nesbet PT2, OOC version, eps_2-screened.
#
#
#
# Usage:
#   julia -t N run_asci_pt2_ooc_eps2.jl \
#       NORB_TOT NELEC_TOT NORB_FRZN EPSILON \
#       ERI_FILE HCORE_FILE SYMM_FILE CHECKPOINT_FILE \
#       [P_OUT=0] [BATCH_SIZE=500] [SPILL_DIR] [COMPACT_EVERY=200] \
#       [MAX_DISK_GB=4000] [EPS2=0.0]
#

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
include(joinpath(@__DIR__, "helper_functions", "bitops.jl"))
include(joinpath(@__DIR__, "helper_functions", "bloom.jl"))
include(joinpath(@__DIR__, "helper_functions", "state.jl"))
include(joinpath(@__DIR__, "helper_functions", "hamiltonian.jl"))
include(joinpath(@__DIR__, "helper_functions", "expanding_matvec.jl"))
include(joinpath(@__DIR__, "helper_functions", "coupled_env.jl"))
include(joinpath(@__DIR__, "spin_coupled", "spin_coupled_helpers.jl"))
include(joinpath(@__DIR__, "spin_coupled", "coupled_expanding_matvec.jl"))
include(joinpath(@__DIR__, "spin_coupled", "batched_expand_and_select_ooc.jl"))
include(joinpath(@__DIR__, "spin_coupled", "checkpoint.jl"))
# --- PT2 machinery ---
include(joinpath(@__DIR__, "asci", "pt2_core.jl"))   
include(joinpath(@__DIR__, "asci", "pt2_expand.jl")) 
include(joinpath(@__DIR__, "asci", "pt2_spill.jl"))  
                                                    
# --- shared config/env/dispatch (MUST precede asci_pt2_ooc_eps2.jl) ---
include(joinpath(@__DIR__, "pt2_experiment.jl"))
# --- the eps2 entry point, plus its :ooc_eps2 harness methods ---
include(joinpath(@__DIR__, "asci", "asci_pt2_ooc_eps2.jl")) 

for (i, a) in enumerate(ARGS); println("ARGS[$i]: $a"); end
println("JULIA_NUM_THREADS = ", Threads.nthreads())
flush(stdout)

(8 <= length(ARGS) <= 14) || error("Expected 8–14 ARGS, got $(length(ARGS))")

P_out         = length(ARGS) >= 9  ? parse(Int, ARGS[9])       : 0
batch_size    = length(ARGS) >= 10 ? parse(Int, ARGS[10])      : 500
spill_dir     = length(ARGS) >= 11 ? ARGS[11]                  : mktempdir()
compact_every = length(ARGS) >= 12 ? parse(Int, ARGS[12])      : 200
max_disk_gb   = length(ARGS) >= 13 ? parse(Float64, ARGS[13])  : 4000.0
eps2          = length(ARGS) >= 14 ? parse(Float64, ARGS[14])  : 0.0

eps2 >= 0.0 || error("EPS2 must be >= 0, got $eps2")

run_pt2_experiment(:ooc_eps2, parse_pt2_config(ARGS);
                   P_out         = P_out,
                   batch_size    = batch_size,
                   spill_dir     = spill_dir,
                   compact_every = compact_every,
                   max_disk_gb   = max_disk_gb,
                   eps2          = eps2)
