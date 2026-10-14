#!/usr/bin/env python3
"""
Generate the ground-state natural-orbital inputs used by the ozone benchmarks.

System: ozone at the built-in "om" geometry, cc-pVTZ basis, C2v symmetry.
There are 24 electrons and 90 spatial orbitals. The orbital optimization
uses 18 active electrons in 87 orbitals, with three doubly occupied core
orbitals. The output integrals retain all 90 orbitals.

Procedure:
    RHF -> MP2 starting orbitals -> two-phase ground-state vHCISCF
    -> natural orbitals of the ground-state density -> transformed integrals.

Requirements:
    NumPy, PySCF, pyscf-shciscf, and Dice.
    Configure the PySCF-Dice interface before running. This command prints
    the settings.py template to place in the installed pyscf/shciscf folder:

        python asci_get_ozone_inputs_gsno_v4.py om --print-settings

Reproduce the benchmark input-generation procedure:
    Run from this script's directory, using a fresh output directory.
    Replace /absolute/path/to/Dice with your Dice executable.

    export OMP_NUM_THREADS=16
    python -u asci_get_ozone_inputs_gsno_v4.py om \
        --dice-exe "/absolute/path/to/Dice" \
        --threads 16 \
        --phase1-eps 1e-3 \
        --eps1 2e-4 \
        --phase1-max-macro 40 \
        --max-macro 50 \
        --outdir "$PWD/asci_gsno_om_reproduced" \
        > ozone_inputs.log 2>&1

Outputs:
    hcore.txt          One-electron integrals.
    eris.txt           Two-electron integrals in physicists' notation.
    symm.txt           Spatial-orbital symmetry labels.
    asci_anchors.json  Settings and reference energies.
    asci_orbitals.npz  Orbital coefficients and occupations.

Checkpoints:
    Orbitals are saved during optimization. To resume phase 2, add
    --restart-mo /path/to/saved_orbitals.npy.

"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from pyscf import ao2mo, gto, mcscf, mp, scf, symm

# ----------------------------------------------------------------------------
# Fixed problem definition (identical to Stage A)
# ----------------------------------------------------------------------------
BASIS = "cc-pvtz"
POINT_GROUP = "c2v"
NORB_EXPECTED = 90
NELEC_EXPECTED = 24
NFROZEN = 3                     # -> (18e, 87o)
NCAS, NELECAS = 87, 18
NOCC = NELEC_EXPECTED // 2
# Single-state (ground 1A1) optimization; no state averaging in v3.

GEOMS = {
    "om": "O 0 0 0; O 0 1.0958447 0.676807; O 0 -1.0958447 0.676807",
    "rm": "O 0 0 0; O 0 0.7313925479 1.2693659602; O 0 -0.7313925479 1.2693659602",
    "ts": "O 0 0 0; O 0 0.9555185059 1.0612108108; O 0 -0.9555185059 1.0612108108",
}

OM_E_NUC = 68.0416694516
OM_E_HF_TOT = -224.3338824654

SETTINGS_TEMPLATE = '''import os
SHCIEXE = os.environ.get("SHCIEXE", "Dice")
SHCISCRATCHDIR = os.environ.get("SHCISCRATCHDIR", ".")
SHCIRUNTIMEDIR = os.environ.get("SHCIRUNTIMEDIR", ".")
MPIPREFIX = os.environ.get("SHCI_MPIPREFIX", "")
'''


# ----------------------------------------------------------------------------
# Helpers (verbatim copies of the Stage-A helpers; do not let them drift)
# ----------------------------------------------------------------------------
def orbsym_ids(mol, mo, s_ao):
    return np.asarray(
        symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo, s=s_ao, check=True),
        dtype=int,
    )


def natural_orbitals_by_irrep(dm_mo, mo_coeff, osym):
    osym = np.asarray(osym)
    n = mo_coeff.shape[1]
    off = 0.0
    for ir in np.unique(osym):
        idx = np.where(osym == ir)[0]
        rest = np.where(osym != ir)[0]
        if len(rest):
            off = max(off, np.max(np.abs(dm_mo[np.ix_(idx, rest)])))
    assert off < 1e-8, f"RDM couples different irreps (max |elem| = {off:.3e})"

    entries = []
    for ir in np.unique(osym):
        idx = np.where(osym == ir)[0]
        w, v = np.linalg.eigh(dm_mo[np.ix_(idx, idx)])
        cols = mo_coeff[:, idx] @ v
        for k in range(len(idx)):
            entries.append((w[k], int(ir), cols[:, k]))

    entries.sort(key=lambda t: (-t[0], t[1]))
    noons = np.array([t[0] for t in entries])
    natsym = np.array([t[1] for t in entries], dtype=int)
    natorbs = np.column_stack([t[2] for t in entries])
    assert noons.shape[0] == n
    return noons, natorbs, natsym


def transform_and_anchor(mol, mf, mo):
    norb = mo.shape[1]
    eri = ao2mo.full(mol.intor("int2e_sph", aosym="s4"), mo, compact=False)
    eri.shape = (norb, norb, norb, norb)
    eri = eri.transpose(0, 2, 1, 3).copy()   # chemists' (pq|rs) -> physicists' <pq|rs>
    h1 = mo.T.dot(mf.get_hcore()).dot(mo)
    occ = np.arange(NOCC)
    sub = eri[np.ix_(occ, occ, occ, occ)]
    e_det = 2.0 * np.trace(h1[np.ix_(occ, occ)])
    e_det += 2.0 * np.einsum("ijij->", sub) - np.einsum("ijji->", sub)
    return h1, eri, float(e_det)


def write_inputs(outdir, mol, mo, h1, eri4, s_ao, natsym=None):
    norb = mo.shape[1]
    os.makedirs(outdir, exist_ok=True)
    irreps = orbsym_ids(mol, mo, s_ao)
    if natsym is not None:
        assert np.array_equal(irreps, natsym), "irrep labels disagree with construction"
    np.savetxt(os.path.join(outdir, "symm.txt"), irreps, fmt="%d")
    np.savetxt(os.path.join(outdir, "hcore.txt"), h1, delimiter=",")
    np.savetxt(os.path.join(outdir, "eris.txt"),
               eri4.reshape(norb**2, norb**2), delimiter=",")
    return irreps


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("geom", choices=sorted(GEOMS))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--dice-exe", required=("--print-settings" not in sys.argv),
                    help='e.g. "/scratch/.../run-dice-20250525.bash Dice"')
    ap.add_argument("--mpiprefix", default="",
                    help="prefix prepended to the Dice command (usually empty; "
                         "MPI, if any, belongs inside --dice-exe)")
    ap.add_argument("--eps1", type=float, default=2e-4,
                    help="target vHCI epsilon_1 in Ha (paper value 2e-4 = 0.2 mEh)")
    ap.add_argument("--phase1-eps", type=float, default=1e-3,
                    help="cheap epsilon_1 for the phase-1 orbital convergence")
    ap.add_argument("--phase1-max-macro", type=int, default=40,
                    help="v3 run showed the eps=1e-3 noise floor is reached "
                         "by ~macro 35; more is wasted time")
    ap.add_argument("--restart-mo", default=None, metavar="FILE.npy",
                    help="orbital coefficients (90x90, AO basis) from a "
                         "previous run's checkpoint; skips phase 1 and "
                         "starts phase 2 from these orbitals")
    ap.add_argument("--threads", type=int, default=16, help="Dice OpenMP threads")
    ap.add_argument("--max-macro", type=int, default=50)
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing test: eps1=5e-3, 1 macro-iteration, no outputs")
    ap.add_argument("--print-settings", action="store_true")
    args = ap.parse_args()

    if args.print_settings:
        print(SETTINGS_TEMPLATE)
        return

    try:
        from pyscf.shciscf import shci
    except ImportError as e:
        sys.exit(f"pyscf-shciscf is not importable ({e}).\n"
                 "Install:  pip install --user git+https://github.com/pyscf/shciscf\n"
                 "and create pyscf/shciscf/settings.py "
                 "(run this script with --print-settings for its contents).")

    geom = args.geom
    outdir = args.outdir or f"asci_gsno_{geom}"   # new dir; never touches v1/v2 outputs
    os.makedirs(outdir, exist_ok=True)
    workdir = os.path.abspath(os.path.join(outdir, "dice_work"))
    os.makedirs(workdir, exist_ok=True)

    eps1 = 5e-3 if args.smoke else args.eps1
    max_macro = 1 if args.smoke else args.max_macro
    mode = "SMOKE TEST" if args.smoke else "PRODUCTION"
    print(f"=== Stage B v3 (ground-state vHCISCF natural orbitals), {geom.upper()}, {mode} ===")
    print(f"eps1 = {eps1:.1e} Ha, single-state (ground 1A1), Dice workdir = {workdir}")

    # ---------------- molecule + RHF (same asserts as Stage A) --------------
    mol = gto.M(verbose=4, atom=GEOMS[geom], unit="Angstrom",
                basis=BASIS, symmetry=POINT_GROUP)
    assert mol.nelectron == NELEC_EXPECTED and mol.nao == NORB_EXPECTED
    s_ao = mol.intor_symmetric("int1e_ovlp")
    e_nuc = mol.energy_nuc()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged, "RHF did not converge"
    e_hf_elec = mf.e_tot - e_nuc
    if geom == "om":
        assert abs(e_nuc - OM_E_NUC) < 1e-8
        assert abs(mf.e_tot - OM_E_HF_TOT) < 1e-6
        print("[PASS] OM anchors: E_NUC and E_HF match the existing HF_only run")

    # ---------------- MP2 natural orbitals (initial guess) ------------------
    pt = mp.MP2(mf)
    pt.frozen = NFROZEN
    pt.kernel()
    dm_mo = pt.make_rdm1()
    assert np.max(np.abs(dm_mo[:NFROZEN, :NFROZEN] - 2.0 * np.eye(NFROZEN))) < 1e-10
    assert np.max(np.abs(dm_mo[:NFROZEN, NFROZEN:])) < 1e-10
    osym_rhf = orbsym_ids(mol, mf.mo_coeff, s_ao)
    noons0, mo_init, _ = natural_orbitals_by_irrep(dm_mo, mf.mo_coeff, osym_rhf)
    assert np.max(np.abs(noons0[:NFROZEN] - 2.0)) < 1e-10
    print(f"MP2 NO initial guess ready (E_MP2 = {pt.e_tot:.10f})")

    # ---------------- vHCISCF via Dice (two phases, checkpointed) -----------
    def make_ckpt(tag):
        """Callback run by pyscf after every macro-iteration: saves the
        current orbitals and appends the energy to a history file. Wrapped
        so that checkpointing can never kill the optimization."""
        mo_path = os.path.join(outdir, f"mo_ckpt_{tag}.npy")
        hist_path = os.path.join(outdir, f"opt_history_{tag}.txt")

        def _cb(envs):
            try:
                if isinstance(envs, dict) and envs.get("mo") is not None:
                    np.save(mo_path, np.asarray(envs["mo"]))
                    e = envs.get("e_tot", "")
                    with open(hist_path, "a") as f:
                        f.write(f"{e}\n")
            except Exception as exc:   # never abort the run over a checkpoint
                print(f"[ckpt warning, {tag}] {exc}")
        return _cb

    def build_mc(sweep_iter, sweep_epsilon, max_macro, max_micro,
                 conv_tol, conv_tol_grad, tag):
        """Fresh ground-state CASSCF(87,18) with a Dice solver on the given
        schedule. All settings other than schedule/criteria are shared."""
        mc = mcscf.CASSCF(mf, NCAS, NELECAS)      # ncore = 3 automatically
        assert mc.ncore == NFROZEN
        fs = shci.SHCI(mol)
        fs.executable = args.dice_exe
        fs.mpiprefix = args.mpiprefix
        fs.runtimeDir = workdir
        fs.scratchDirectory = workdir
        fs.integralFile = os.path.join(workdir, "FCIDUMP")
        fs.configFile = os.path.join(workdir, "input.dat")
        fs.outputFile = os.path.join(workdir, "output.dat")
        fs.nroots = 1             # ground state only (v3)
        fs.wfnsym = "A1"          # pins Dice to the A1 sector
        fs.spin = 0
        fs.sweep_iter = list(sweep_iter)
        fs.sweep_epsilon = list(sweep_epsilon)
        fs.nPTiter = 0            # variational only inside the SCF loop
        fs.DoRDM = True
        fs.dE = 1e-9
        fs.num_thrds = args.threads
        mc.fcisolver = fs
        mc.internal_rotation = True   # active-active rotations (vHCI not invariant)
        mc.conv_tol = conv_tol
        mc.conv_tol_grad = conv_tol_grad
        mc.max_cycle_macro = max_macro
        mc.max_cycle_micro = max_micro
        mc.natorb = False             # NOs constructed explicitly below
        mc.callback = make_ckpt(tag)
        return mc

    def run_kernel(mc, mo_guess, label):
        try:
            mc.kernel(mo_guess)
        except subprocess.CalledProcessError:
            out = os.path.join(workdir, "output.dat")
            print(f"\n*** Dice execution failed during {label}. "
                  "Last lines of its output: ***")
            if os.path.exists(out):
                print("".join(open(out).readlines()[-30:]))
            print(f"*** Inspect {workdir}/input.dat and output.dat. ***")
            raise

    if args.smoke:
        mc = build_mc([0], [eps1], max_macro, 4, 1e-5, 1e-3, "smoke")
        run_kernel(mc, mo_init, "smoke test")
        print(f"vHCISCF ground state: E = {float(mc.e_tot):.10f}")
        print("[SMOKE PASS] full PySCF <-> Dice round trip worked; "
              "no integrals written.")
        return

    if args.restart_mo:
        # Skip phase 1: start phase 2 from a previously checkpointed basis.
        mo_start = np.load(args.restart_mo)
        assert mo_start.shape == (NORB_EXPECTED, NORB_EXPECTED)
        assert np.max(np.abs(mo_start.T @ s_ao @ mo_start
                             - np.eye(NORB_EXPECTED))) < 1e-6, \
            "--restart-mo orbitals are not orthonormal in this AO basis"
        orbsym_ids(mol, mo_start, s_ao)   # raises if not symmetry-pure
        e_p1 = None
        print(f"--- Phase 1 skipped: restarting from {args.restart_mo} ---")
    else:
        # Phase 1: cheap epsilon, relaxed criteria. The vHCI energy at coarse
        # eps1 is only piecewise smooth in the orbitals, so we do NOT demand
        # convergence -- any orbital improvement is passed to phase 2.
        print(f"--- Phase 1: orbital relaxation at eps1 = {args.phase1_eps:.1e} ---")
        mc1 = build_mc([0], [args.phase1_eps], args.phase1_max_macro, 4,
                       conv_tol=1e-5, conv_tol_grad=3e-3, tag="phase1")
        run_kernel(mc1, mo_init, "phase 1")
        mo_start = mc1.mo_coeff
        np.save(os.path.join(outdir, "mo_phase1_final.npy"), mo_start)
        e_p1 = float(mc1.e_tot)
        print(f"phase-1 ground state: E = {e_p1:.10f} "
              f"(converged = {bool(mc1.converged)}; non-convergence here is OK)")

    # Phase 2: target epsilon, warm-started. conv_tol = 1e-5 sits above the
    # measured CI-reselection chatter (~6e-6 Ha/macro at eps1 = 2e-4);
    # stationarity is judged by the orbital gradient (< 1e-3).
    print(f"--- Phase 2: refinement at eps1 = {eps1:.1e} ---")
    mc = build_mc([0, 3], [max(1e-3, 5.0 * eps1), eps1], max_macro, 3,
                  conv_tol=1e-5, conv_tol_grad=1e-3, tag="phase2")
    run_kernel(mc, mo_start, "phase 2")

    # Persist the result BEFORE any acceptance test, so nothing is ever lost.
    e_final = float(mc.e_tot)
    np.save(os.path.join(outdir, "mo_phase2_final.npy"), mc.mo_coeff)
    print(f"vHCISCF ground state @ eps1: E = {e_final:.10f} "
          f"(saved mo_phase2_final.npy)")
    assert mc.converged, \
        "phase 2 did not converge; resubmit with --restart-mo " \
        f"{os.path.join(outdir, 'mo_phase2_final.npy')} to continue"
    # Variational cross-check: the eps1 space at the phase-2 minimizer must
    # beat the coarse-eps space at its own minimizer.
    if e_p1 is not None:
        assert e_final <= e_p1 + 1e-6, \
            f"E(phase 2) = {e_final:.8f} above E(phase 1) = {e_p1:.8f}: inconsistent"

    # ---------------- natural orbitals of the ground-state 1-RDM ------------
    d_ao = mc.make_rdm1()                       # ground-state AO density
    mo_c = mc.mo_coeff
    d_mo = mo_c.T @ s_ao @ d_ao @ s_ao @ mo_c
    assert abs(np.trace(d_mo) - NELEC_EXPECTED) < 1e-6
    osym_mc = orbsym_ids(mol, mo_c, s_ao)       # raises if symmetry drifted
    noons, natorbs, natsym = natural_orbitals_by_irrep(d_mo, mo_c, osym_mc)

    assert np.all(np.diff(noons) < 1e-12)
    assert np.max(np.abs(noons[:NFROZEN] - 2.0)) < 1e-8, "cores not first"
    assert noons.min() > -1e-8 and noons.max() < 2.0 + 1e-8
    assert np.max(np.abs(natorbs.T @ s_ao @ natorbs - np.eye(NORB_EXPECTED))) < 1e-8
    print("First 16 NOONs of the ground-state density:")
    print(np.array2string(noons[:16], precision=6, max_line_width=100))

    # ---------------- transform, anchor, write ------------------------------
    h1, eri4, e_det = transform_and_anchor(mol, mf, natorbs)
    # The RHF determinant is the variational minimum over all determinants:
    assert e_det > e_hf_elec - 1e-9, "determinant anchor below E_HF: convention bug"
    hdiag = np.diag(h1)
    assert np.all(hdiag[:NFROZEN] < -30.0) and np.all(hdiag[NFROZEN:] > -15.0), \
        "first three orbitals are not the O 1s cores"

    irreps = write_inputs(outdir, mol, natorbs, h1, eri4, s_ao, natsym)
    np.savez(os.path.join(outdir, "asci_orbitals.npz"),
             natorbs=natorbs, noons=noons, natsym=natsym,
             mo_casscf=mo_c, mo_init_mp2no=mo_init)

    anchors = {
        "stage": "B_gs_vhciscf_no_v4",
        "geometry": geom,
        "eps1": eps1,
        "phase1_eps": args.phase1_eps,
        "phase1_E_total": e_p1,
        "state_specific": True,
        "E_NUC": e_nuc,
        "E_RHF_total": mf.e_tot,
        "E_RHF_electronic": e_hf_elec,
        "E_MP2_total": pt.e_tot,
        "E_vhciscf_total": e_final,
        "E_det_reference_electronic": e_det,
        "E_det_reference_total": e_det + e_nuc,
        "noons_first16": noons[:16].tolist(),
        "irreps": irreps.tolist(),
    }
    with open(os.path.join(outdir, "asci_anchors.json"), "w") as f:
        json.dump(anchors, f, indent=2)

    print()
    print("================ ANCHORS (record these) ================")
    print(f"geometry                     : {geom.upper()}")
    print(f"E_NUC                        : {e_nuc:.10f}")
    print(f"E_RHF (total)                : {mf.e_tot:.10f}")
    print(f"E_vHCISCF (ground, @ eps1)   : {e_final:.10f}")
    if e_p1 is not None:
        print(f"E_vHCISCF (phase 1, 1e-3)    : {e_p1:.10f}")
    print(f"<D0|H|D0> (electronic)       : {e_det:.10f}")
    print(f"<D0|H|D0> + E_NUC (total)    : {e_det + e_nuc:.10f}")
    print(f"outputs written to           : {os.path.abspath(outdir)}")
    print("========================================================")


if __name__ == "__main__":
    main()