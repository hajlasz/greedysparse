#!/bin/bash
#SBATCH --job-name=pt2_10M_fine
#SBATCH --output=pt2_10M_fine_%j.out
#SBATCH --error=pt2_10M_fine_%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48
#SBATCH --mem=600G --time=24:00:00
#SBATCH --account torch_pr_554_general
# ==============================================================================
# A5b -- FINE eps_2 LADDER at |S| = 1e7.  Job 14907626 was UNCONVERGED.
#
# WHAT JOB 14907626 SHOWED
#     eps2     E_PT2 (Ha)      y = E_var+E_PT2   peak      wall
#     6e-6    -0.0157569365    -293.1717776393   17.8 GiB  0.38 h
#     3e-6    -0.0186094191    -293.1746301219   54.2 GiB  0.47 h
#     2e-6    -0.0199813353    -293.1760020381   88.3 GiB  0.54 h
#
# Successive gaps: 2.852 mHa then 1.372 mHa. A three-point power-law fit gives
# beta = 0.353 -- i.e. the ladder is sitting in the COARSE, slowly-converging
# regime, not the asymptotic one. (At |S| = 2e5 the same drift appeared: beta
# ran 0.648 -> 0.996 -> 1.510 -> 2.644 as eps2 fell 1e-4 -> 1e-6. Only the last
# interval was asymptotic.) With beta = 0.353 the implied bias at 2e-6 is
# ~8.9 mHa, so E_PT2(2e-6) = -19.98 mHa is nowhere near the limit, and the
# extrapolation from those three points must NOT be trusted -- extrapolating a
# drifting exponent is exactly the failure mode already seen at 200k.
#
# WHY 2e-6 IS TOO COARSE HERE (and the projections were 20x pessimistic)
# Disk came in at 88 GiB where 1684 GiB was projected. The x50 scaling from
# |S| = 2e5 assumed the screened FRACTION is |S|-independent. It is not: the
# eigenvector is normalized, so typical |x_i| ~ |S|^(-1/2), and a fixed eps2
# screens far harder at larger |S|. The same argument fixes the target:
#
#     eps2_equivalent(1e7)  ~  eps2(2e5) / sqrt(50)  =  1e-6 / 7.07  ~  1.4e-7
#
# So the asymptotic regime at 1e7 begins near 1.5e-7, roughly a factor of 14
# below where the first ladder stopped.
#
# PROJECTIONS (D ~ eps2^-1.458 fit on the three measured points; COMPACT=50
# halves the first-interval accumulation, which is what actually sets the peak)
#     eps2       peak(GiB)   vs 3800   wall
#     1e-6           121      0.03x    ~0.7 h
#     5e-7           333      0.09x    ~0.8 h
#     2.5e-7         915      0.24x    ~1.0 h
#     2e-7          1267      0.33x    ~1.1 h
# All four fit with wide margin. Total ~4 h.
#
# WHY COMPACT=50 (was 100)
# In job 14907626 the peak was set entirely by the FIRST compaction interval:
# at 2e-6 disk spiked to 88.3 GiB over batches 1-100, then compaction dropped
# it to 19.8 GiB and the steady state only crept to 26.5 GiB. Sources are
# ordered by |x_i|, so the first batches generate by far the most. Compacting
# at 50 halves that spike. Steady-state compactions cost 1-8 s; only the first
# is expensive (111 s at 2e-6).
#
# CONVERGENCE CRITERION -- this is the point of the run
# Fit E(eps2) = E0 + C*eps2^beta on the finest three points. The ladder is
# converged only when beta has STABILISED at >= 2. Then, and only then,
#     residual bias  ~=  |gap| / (r^beta - 1)
# is meaningful. If beta is still climbing at 2e-7, report the raw finest value
# and its gap as a bound; do not extrapolate.
#
# WHAT TO WATCH
# The benchmark line (a = -293.1810584, b = 0.068845) predicts
# |E_PT2(1e7)| = 23.43 mHa. As eps2 falls, |E_PT2| should climb from 19.98 mHa
# toward ~23.4 mHa and PLATEAU. If it sails past ~25 mHa without flattening,
# either the benchmark intercept is too high or the series is non-linear --
# stop and refit rather than pushing eps2 further.
# ==============================================================================
set -o pipefail
ulimit -n 65536 || ulimit -n 32768 || true
module purge
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
JB=/home/nh2429/julia/julia

SD=/scratch/nh2429/greedysparse/chemistry_experiments/scripts
BASE=/scratch/nh2429/greedysparse/chemistry_experiments/ozone_new_inputs/input_data/asci_gsno_om
CK=/scratch/nh2429/greedysparse/chemistry_experiments/ozone_new_inputs/results/checkpoints/asci__O3_asci_c500k_t10M.ckpt

# ===== EDIT =====
POUT=0
PBATCH=5000
COMPACT=50         # was 100; the first interval sets the peak
MAXDISK=3800       # GiB, Julia-side abort
HARDKILL_GIB=3900  # GiB, script-side SIGKILL between compactions
EPS=1e-3
EPS2_LIST="1e-7"
# ================

HARDKILL_B=$(( HARDKILL_GIB * 1024 * 1024 * 1024 ))
[[ -f "$CK" ]] || { echo "no checkpoint: $CK"; exit 1; }
echo "checkpoint : $CK"
echo "eps2 ladder: $EPS2_LIST      hard kill at ${HARDKILL_GIB} GiB"
echo "EXPECT E_var = -293.1560207028 Ha in each banner; stop if it differs."
echo "WATCH |E_PT2| climb from 19.98 mHa toward ~23.4 mHa and plateau."

for E2 in $EPS2_LIST; do
    PSP=/scratch/nh2429/pt2_10Mf_${SLURM_JOB_ID}_${E2}
    PKF=$(mktemp)
    rm -rf "$PSP"; mkdir -p "$PSP"
    echo ""
    echo "=============================================================="
    echo "  eps2 = $E2    spill = $PSP"
    echo "=============================================================="
    df -h "$PSP"

    t0=$(date +%s)
    "$JB" -t "$JULIA_NUM_THREADS" "$SD/run_asci_pt2_ooc_eps2.jl" \
      90 24 3 "$EPS" "$BASE/eris.txt" "$BASE/hcore.txt" "$BASE/symm.txt" \
      "$CK" "$POUT" "$PBATCH" "$PSP" "$COMPACT" "$MAXDISK" "$E2" \
      2> "$HOME/pt2_10Mf_${SLURM_JOB_ID}_${E2}.err" &
    JPID=$!

    # 30 s sampling: at fine eps2 the first interval fills fast, and the Julia
    # MAXDISK check only fires at compaction points.
    (
      while kill -0 "$JPID" 2>/dev/null; do
          B=$(du -sb "$PSP" 2>/dev/null | cut -f1)
          if [[ -n "$B" ]]; then
              echo "$B" >> "$PKF"
              awk -v b="$B" -v e="$E2" \
                  'BEGIN{printf "SPILL[%s] %d B = %.2f GiB\n", e, b, b/1073741824}'
              if [[ "$B" -gt "$HARDKILL_B" ]]; then
                  echo "*** HARD KILL: ${B} B exceeds ${HARDKILL_GIB} GiB ***"
                  kill -9 "$JPID" 2>/dev/null
                  break
              fi
          fi
          sleep 30
      done
    ) & MON=$!

    wait "$JPID"; rc=$?
    kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null
    t1=$(date +%s)

    PEAK_B=$(sort -n "$PKF" 2>/dev/null | tail -1); [[ -z "$PEAK_B" ]] && PEAK_B=0
    awk -v p="$PEAK_B" -v e="$E2" -v w="$((t1-t0))" -v rc="$rc" \
        'BEGIN{printf "RESULT eps2=%s  exit=%d  wall=%ds (%.2f h)  MEASURED PEAK = %d B = %.1f GiB\n",
               e, rc, w, w/3600, p, p/1073741824}'
    rm -rf "$PSP" "$PKF"
    [[ $rc -ne 0 ]] && { echo "ABORTING LADDER at eps2=$E2"; exit $rc; }
done

echo ""
echo "=============================================================="
echo "FINE LADDER COMPLETE.  Combine with job 14907626 for 7 points:"
echo "  6e-6 3e-6 2e-6 1e-6 5e-7 2.5e-7 2e-7"
echo ""
echo "1. local beta on each consecutive pair -- has it stabilised at >= 2?"
echo "2. if yes: E_PT2(0) from the finest three; residual = gap/(r^beta - 1)"
echo "   if no:  report raw finest value + gap as an upper bound on the bias"
echo "3. y_10M = -293.1560207028 + E_PT2 ;  compare |a - y_10M| to 1.594 mHa"
echo "=============================================================="
