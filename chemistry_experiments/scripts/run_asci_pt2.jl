# run_asci_pt2.jl: Epstein-Nesbet PT2 correction, Not the OOC version.
#
# Reads a converged ASCI checkpoint and computes the second-order correction to
# its variational energy. Reports E_var, E_PT2, and E_var + E_PT2.
#
#
# Usage:
#   julia -t N run_asci_pt2.jl \
#       NORB_TOT NELEC_TOT NORB_FRZN EPSILON \
#       ERI_FILE HCORE_FILE SYMM_FILE CHECKPOINT_FILE \
#       [N_SHARDS=256] [BATCH_SIZE=200000] [SPILL_DIR] [EPS2=0.0]
#
#   * NORB_TOT/NELEC_TOT/NORB_FRZN include frozen orbitals
#     (ozone = 90 24 3, Neon = 23 10 1).
#   * N_SHARDS controls peak RAM
#   * SPILL_DIR should be on fast scratch; defaults to a temp dir.
#   * EPS2 = 0.0 is screening threshold
#
# (Submitted on the cluster via ozone/sbatch/pt2.sbatch.)

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
include(joinpath(@__DIR__, "asci", "pt2_ooc.jl"))      
                                                       
# --- shared config/env/dispatch ---
include(joinpath(@__DIR__, "pt2_experiment.jl"))

for (i, a) in enumerate(ARGS); println("ARGS[$i]: $a"); end
println("JULIA_NUM_THREADS = ", Threads.nthreads())
flush(stdout)

(8 <= length(ARGS) <= 12) || error("Expected 8–12 ARGS, got $(length(ARGS))")

n_shards   = length(ARGS) >= 9  ? parse(Int, ARGS[9])       : 256
batch_size = length(ARGS) >= 10 ? parse(Int, ARGS[10])      : 200_000
spill_dir  = length(ARGS) >= 11 ? ARGS[11]                  : mktempdir()
eps2       = length(ARGS) >= 12 ? parse(Float64, ARGS[12])  : 0.0

run_pt2_experiment(:spill, parse_pt2_config(ARGS);
                   n_shards   = n_shards,
                   batch_size = batch_size,
                   spill_dir  = spill_dir,
                   eps2       = eps2)
