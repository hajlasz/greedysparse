from pyscf import gto, scf, mcscf 
from pyscf import ao2mo 
import numpy as np 


##### DEFINE THE MOLECULE AND BASIS SET #####
# atom=     : atomic composition and geometry (Ne at the origin).
# basis=    : the atomic-orbital (AO) basis set, i.e. the finite one-electron
#             subspace the whole calculation lives in.
# symmetry='d2h' : run the SCF in the D2h point group. The MOs come out
#             symmetry-adapted, so each can be labeled by a D2h irreducible
#             representation (used later to restrict allowed excitations).
#             Ne's true symmetry is the full rotation group, but D2h is the
#             largest Abelian point group PySCF supports; symmetry=True would
#             auto-detect the highest usable symmetry and also land on D2h here.
# verbose=4 : print SCF iteration details.
mol = gto.M(verbose=4, atom="Ne 0 0 0", basis="aug-cc-pvdz", symmetry='d2h')

##### RESTRICTED HARTREE-FOCK CALCULATION #####
# Produces the orthonormal MO basis (mf.mo_coeff) in which we build Slater determinants. 
# "Restricted" means we put spin-up and spin-down electrons in the same spatial orbital, 
# so each spatial MO corresponds to 2 spin-orbitals. mo_coeff[:, p] is MO p in the AO basis.
mf = scf.RHF(mol).run()
norb = mf.mo_coeff.shape[1] # total number of MOs (= number of AO basis functions here)


#### MO symmetry labels (one D2h irrep id per orbital) ####
irreps = scf.hf_symm.get_orbsym(mol, mf.mo_coeff)
np.savetxt('symm.txt', irreps, fmt='%d')

#### Two-electron integrals in the MO basis ####
# ao2mo.full transforms the AO 2e-integrals into the MO basis and returns them
# in CHEMISTS' notation:
#     (pq|rs) = ∫∫ φ_p(1)* φ_q(1) (1/r12) φ_r(2)* φ_s(2) d1 d2,
# with (p,q) on electron 1 and (r,s) on electron 2.
#   mol.intor('int2e_sph', aosym='s4'): AO 2e-integrals over spherical-harmonic
#       (not Cartesian) GTOs; 's4' exploits the 4-fold permutational symmetry of
#       the real AO integrals so equivalent ones aren't recomputed.
#   compact=False: return the full (norb^2, norb^2) matrix instead of the
#       symmetry-packed triangular form, so it reshapes straight to 4 indices.
eris = ao2mo.full(mol.intor('int2e_sph', aosym='s4'), mf.mo_coeff, compact=False)
 
# Reshape to 4 indices: now eris[p, q, r, s] = (pq|rs)  [chemists'].
eris.shape = (norb, norb, norb, norb)
 
# Convert chemists' (pq|rs) -> physicists' <pq|rs>.
# Physicists' <pq|rs> puts (p,r) on electron 1 and (q,s) on electron 2, and
# satisfies the identity  <pq|rs> = (pr|qs)
eris = eris.transpose(0, 2, 1, 3).copy()

# Flatten again and save to file
eris.shape = (norb**2, norb**2)
np.savetxt('eris.txt', eris, delimiter=',')

#### One-electron (core Hamiltonian) integrals in the MO basis ####
# get_hcore() is the one-electron core Hamiltonian in the AO basis: kinetic
# energy plus electron-nuclear attraction (T + V_ne). The similarity transform
# C^T h_AO C rotates it into the MO basis, giving h_core[p,q] = <p| T + V_ne |q>.
h_core = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
np.savetxt('hcore.txt', h_core, delimiter=',')
