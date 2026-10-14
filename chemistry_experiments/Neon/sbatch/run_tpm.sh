#!/bin/bash
# run_tpm.sh
# TPM on Neon with A = sigma*I - eps*H.
#
# DEFAULT IS THE OPTIMAL SHIFT FOR NEON:
#     sigma = 1.0,  eps = -0.0098475     =>     A = I + 0.0098475*H
#
# eps* = 2/(lambda_2 + lambda_n) is NEGATIVE for this system.  The frozen-core
# neon block has an entirely negative spectrum, so the midpoint 1/eps* =
# -101.5486 lies below zero and lambda_2 + lambda_n = -203.0972.  Consequently
# the ground state maps to lambda_1(A) = -0.267467, the ALGEBRAICALLY SMALLEST
# eigenvalue of A.  TPM is fine with that -- the power method tracks the largest
# MAGNITUDE eigenvalue and eps* is chosen precisely so that is the ground state
# -- but the iterate flips sign every step, which is expected.
#
# eps* maximises the relative eigengap, so TPM should converge FASTER here than
# at sigma = 0, not slower.  The 8.8x slowdown in the paper is the eps = +1e-3
# case, where the identity dominates and the gap collapses.
#
# EIG_SELECT (4th argument) is accepted for symmetry with run_greedy.sh and is
# exported so a paired comparison is unambiguous, but TPM performs NO eigensolve
# -- it takes a Rayleigh quotient -- so it has no effect on this run.  It is
# deliberately left out of the output tag, so two TPM runs differing only in
# EIG_SELECT resolve to the same file and the second is refused as a duplicate.
#
# Usage: ./run_tpm.sh [K_KEEP] [SHIFT] [EPSILON] [EIG_SELECT]
#   ./run_tpm.sh 262500                 -> optimal shift, A = I + 0.0098475*H
#   ./run_tpm.sh 262500 0.0 1.0         -> baseline,      A = -H
#   ./run_tpm.sh 262500 1.0 1e-3        -> paper's run,   A = I - 1e-3*H
#
# Target: crunchy1.cims.nyu.edu -- 4x AMD Opteron 6272 (64 cores), 256 GB, RHEL 9.
# ---------------------------------------------------------------------------
K_KEEP=${1:-262500}
SHIFT=${2:-1.0}
EPSILON=${3:--0.0098475}
EIG_SELECT=${4:-algebraic}

case "$EIG_SELECT" in
    algebraic) ;;
    magnitude)
        echo "NOTE: EIG_SELECT=magnitude has no effect on TPM -- it takes a Rayleigh"
        echo "      quotient and never calls eigs.  The :LR/:SR vs :LM comparison is"
        echo "      a greedy/ASCI question; use ./run_greedy.sh for it."
        ;;
    *) echo "ERROR: EIG_SELECT must be 'algebraic' or 'magnitude', got '$EIG_SELECT'"; exit 1 ;;
esac

# The Julia side reads both of these from the environment (parse_config for the
# shift, EIG_SELECT[] in state.jl for the target).  Do NOT append either as a
# positional argument: run_tpm.jl has no ARGS length check and parse_config
# indexes args[1:12], so extras are silently discarded.
export GS_SHIFT="$SHIFT"
export GS_EIG_SELECT="$EIG_SELECT"

# Tagged with (K_KEEP, sigma, eps) so runs at different shifts cannot overwrite
# each other.  EIG_SELECT is omitted on purpose -- see the note above.
TAG="k${K_KEEP}_s${SHIFT}_e${EPSILON}"

LOGDIR=/home/nh2429/greedysparse/chemistry_experiments/Neon/sbatch/logs/tpm
STAMP=$(date +%Y%m%d_%H%M%S)_$$
if ! mkdir -p "$LOGDIR" 2>/dev/null || ! : > "$LOGDIR/.wtest_$$" 2>/dev/null; then
    LOGDIR=/tmp
    FALLBACK_LOG=1
fi
rm -f "$LOGDIR/.wtest_$$"
LOGFILE="$LOGDIR/tpm_${TAG}_${STAMP}.log"

if [[ -t 1 ]]; then
    exec > >(tee -a "$LOGFILE") 2>&1   # terminal + file
    TEE_PID=$!
else
    exec >>"$LOGFILE" 2>&1             # file only: no tty to lose
fi

echo "=== run_tpm.sh  started $(date) ==="
echo "log: $LOGFILE"
[[ -n "${FALLBACK_LOG:-}" ]] && echo "WARNING: log directory unwritable, fell back to /tmp"

# Report how we exit, whatever happens -- error, Ctrl-C, or kill -- then close
# stdout so tee sees EOF and wait for it, so no trailing output is lost.
on_exit() {
    local rc=$?
    echo
    if [[ $rc -eq 0 ]]; then
        echo "=== EXIT 0 (success)  $(date) ==="
    else
        echo "=== EXIT $rc (FAILED at line ${BASH_LINENO[0]:-?})  $(date) ==="
    fi
    echo "log: $LOGFILE"
    if [[ -n "${TEE_PID:-}" ]]; then
        exec 1>&- 2>&-
        wait "$TEE_PID" 2>/dev/null || true
    fi
}
trap on_exit EXIT
trap 'echo "!!! caught SIGTERM/SIGINT"; exit 130' TERM INT

set -euo pipefail
set -o errtrace
trap 'echo "!!! error on line $LINENO: $BASH_COMMAND"' ERR

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
JULIA_MODULE=julia-1.11             # check with: module avail julia
PROJECT=/home/nh2429/greedysparse   # set to "" to use your default v1.11 env
NTHREADS=16                         # keep the total across concurrent jobs <= 64
NICENESS=10

export JULIA_NUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS=1       # else OpenBLAS grabs all 64 cores underneath
export OMP_NUM_THREADS=1            # the Julia threads and thrashes

# ---------------------------------------------------------------------------
# Paths -- absolute, since there is no guaranteed working directory
# ---------------------------------------------------------------------------
REPO=/home/nh2429/greedysparse
CHEM="$REPO/chemistry_experiments"
SCRIPTS="$CHEM/scripts"
SYSDIR="$CHEM/Neon"
INPUT="$SYSDIR/input_data"

ERI_FILE="$INPUT/eris.txt"
HCORE_FILE="$INPUT/hcore.txt"
SYMM_FILE="$INPUT/symm.txt"

OUTDIR="$SYSDIR/results/tpm"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/tpm_${TAG}.json"

# ---------------------------------------------------------------------------
# Problem setup (Neon)
# ---------------------------------------------------------------------------
NORB_TOT=23
NELEC_TOT=10
NORB_FRZN=1

# REL_TOL is the |dE| threshold for the convergence streak, in Hartree.  The
# Neon driver does NOT rescale it for sigma (unlike run_tpm_coupled_ooc.jl),
# so it is applied raw.  That errs SAFE at the optimal shift: eps* widens the
# gap, so |dE| is larger for a given true error and the streak fires later
# rather than earlier.  It would err UNSAFE at eps = +1e-3, where the gap
# collapses by 8.8x -- divide REL_TOL by that factor if you run that case.
MAX_ITERS=2000
REL_TOL=1e-11
EVAL_TRUE=128.709475548738

# ---------------------------------------------------------------------------
# Load the julia module
# ---------------------------------------------------------------------------
set +u
if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
                /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
        if [[ -r "$init" ]]; then . "$init"; break; fi
    done
fi
module purge >/dev/null 2>&1 || true
module load "$JULIA_MODULE"
set -u

JULIA="$(command -v julia || true)"
[[ -n "$JULIA" ]] || {
    echo "ERROR: 'module load $JULIA_MODULE' did not put julia on PATH."
    echo "       Check the exact module name with: module avail julia"
    exit 1
}

if [[ -n "$PROJECT" ]]; then PROJECT_ARG=(--project="$PROJECT"); else PROJECT_ARG=(); fi

# ---------------------------------------------------------------------------
# Preflight -- fail in 30 seconds instead of silently in 2 hours
# ---------------------------------------------------------------------------
[[ -e "$OUTFILE" ]] && { echo "ERROR: $OUTFILE already exists -- refusing to overwrite"; exit 1; }

for f in "$SCRIPTS/run_tpm.jl" "$ERI_FILE" "$HCORE_FILE" "$SYMM_FILE"; do
    [[ -r "$f" ]] || { echo "ERROR: missing or unreadable: $f"; exit 1; }
done

# The tree must know about sigma at all.
grep -q "shift" "$SCRIPTS/helper_functions/state.jl" || {
    echo "ERROR: QCEnv in state.jl has no 'shift' field -- this tree predates the"
    echo "       sigma refactor.  GS_SHIFT would be ignored."
    exit 1
}
grep -q "GS_SHIFT" "$SCRIPTS/Neon_sparsified_experiment.jl" || {
    echo "ERROR: Neon_sparsified_experiment.jl does not read GS_SHIFT."
    echo "       ExperimentConfig needs a 'shift' field and parse_config needs"
    echo "         parse(Float64, get(ENV, \"GS_SHIFT\", \"0.0\"))"
    echo "       inserted directly after epsilon, or this run is silently sigma=0."
    exit 1
}

# TPM never calls eigs, so :LR/:SR/:LM does not arise here.  But estimate_step
# for :greedy lives in the same file, and mixing a fixed and an unfixed tree
# across the two methods would make the greedy/TPM comparison meaningless.
if [[ "$EPSILON" == -* ]]; then
    grep -q "which_target" "$SCRIPTS/Neon_sparsified_experiment.jl" || {
        echo "ERROR: EPSILON = $EPSILON is negative, but Neon_sparsified_experiment.jl"
        echo "       still hardcodes which = :LR.  TPM would run correctly while the"
        echo "       paired greedy run would not, so refusing to produce half a"
        echo "       comparison.  Replace it with which = which_target(env)."
        exit 1
    }
fi

"$JULIA" "${PROJECT_ARG[@]}" --startup-file=no -e '
    using Arpack, Combinatorics, StatsBase, JSON, BenchmarkTools,
          LinearMaps, DataStructures
    println("julia ", VERSION, "   deps OK")' || {
    echo "ERROR: package dependencies are not installed in this environment."
    echo "       Your Greene depot is not on CIMS. See one-time setup below."
    exit 1
}

echo "-------------------------------------"
echo "host       = $(hostname)"
echo "load       = $(uptime)"
echo "julia      = $JULIA"
echo "project    = ${PROJECT:-<default env>}"
echo "threads    = $NTHREADS"
echo "shift      = $SHIFT   (sigma, passed via GS_SHIFT)"
echo "epsilon    = $EPSILON"
echo "operator   = A = ${SHIFT}*I - (${EPSILON})*H"
echo "eig_select = $EIG_SELECT   (exported, but unused: TPM has no eigensolve)"
echo "K_KEEP     = $K_KEEP"
echo "script     = $SCRIPTS/run_tpm.jl"
echo "outfile    = $OUTFILE"
echo "-------------------------------------"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
nice -n "$NICENESS" "$JULIA" "${PROJECT_ARG[@]}" \
    --threads "$NTHREADS" --startup-file=no \
    "$SCRIPTS/run_tpm.jl" \
    "$NORB_TOT" \
    "$NELEC_TOT" \
    "$NORB_FRZN" \
    "$EPSILON" \
    "$ERI_FILE" \
    "$HCORE_FILE" \
    "$SYMM_FILE" \
    "$K_KEEP" \
    "$OUTFILE" \
    "$MAX_ITERS" \
    "$REL_TOL" \
    "$EVAL_TRUE"

# ---------------------------------------------------------------------------
# ONE-TIME PACKAGE SETUP (your ~/.julia on CIMS is empty; Greene's is not here).
# The depot lives in your NFS home, so you can do this from any CIMS Linux host:
#
#   module load julia-1.11
#   cd /home/nh2429/greedysparse
#   julia --project=. -e 'using Pkg; Pkg.add(["Arpack","Combinatorics", \
#         "StatsBase","JSON","BenchmarkTools","LinearMaps","DataStructures"]); \
#         Pkg.precompile()'
#
# LAUNCH (from the NYU VPN, via access.cims.nyu.edu):
#
#   ssh crunchy1.cims.nyu.edu
#   uptime                                   # check the box is not loaded
#   chmod +x run_tpm.sh
#
#   tmux new -s tpm                          # so the run survives logout
#   ./run_tpm.sh 262500                      # optimal shift, screen AND log
#                                            # detach: Ctrl-b d
#
# BASELINE for the shift comparison (writes a separate JSON):
#   ./run_tpm.sh 262500 0.0 1.0
#
# Fully detached instead (log file only, terminal echo disabled automatically):
#   nohup setsid ./run_tpm.sh 262500 >/dev/null 2>&1 &
#   tail -f /home/nh2429/greedysparse/chemistry_experiments/Neon/sbatch/logs/tpm/*.log
# ---------------------------------------------------------------------------