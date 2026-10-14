from pyscf import gto, scf, mcscf 
from pyscf import ao2mo 
import numpy as np 


def geometry_sanity_check(mol):
    coords = mol.atom_coords()   # in Bohr
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    print("\n===== Geometry sanity check =====")
    for i in range(mol.natm):
        print(f"Atom {i}: {symbols[i]}  {coords[i]} (Bohr)")

    # compute all pairwise distances
    print("\nBond lengths:")
    for i in range(mol.natm):
        for j in range(i+1, mol.natm):
            r = np.linalg.norm(coords[i] - coords[j])
            print(f"  {symbols[i]}({i}) - {symbols[j]}({j}) = {r:.8f} Bohr")

    # if triatomic, print angle
    if mol.natm == 3:
        # angle at atom 0: 1-0-2
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

        print(f"\nAngle O(1)-O(0)-O(2) = {angle:.6f} degrees")

    print("================================\n")


# Define the molecule
# "atom=" defines the atomic composition and geometry
# basis="..." sets the atomic orbital basis set (i.e. what one-electron subspace to use)
# symmetry='d2h' ensures Hartree-Fock selects a subspace that is invariant w.r.t. D2h symmetry
# and then the resulting molecular orbitals can be labeled by irreducible representations of D2h
# If I set symmetry=True instead, PySCF will use the highest possible symmetry (which is D2h for Neon atom)
mol = gto.M(verbose=4, atom="O 0 0 0; O 0 1.0958447 0.676807; O 0 -1.0958447 0.676807", unit="Angstrom", basis="cc-pvtz", symmetry='c2v')
print("Detected symmetry group:", mol.groupname)
print(geometry_sanity_check(mol))

# Run Restricted Hartree-Fock calculation to get molecular orbital basis
# (i.e. orthonormal one-electron basis with which we define Slater Determinants)
mf = scf.RHF(mol).run()

# Total Number of Molecular Orbitals
norb = mf.mo_coeff.shape[1]

#### Output MO symmetry labels (in D2h for Neon) ####
irreps = scf.hf_symm.get_orbsym(mol, mf.mo_coeff)
np.savetxt('symm.txt', irreps, fmt='%d')

#### Output Two-Electron Integrals in MO basis ####
# This returns eris in flattened chemists' notation
# So when we reshape it to (norb, norb, norb, norb), (p,q,r,s) element is <pq|rs>
# int2e_sph means we used spherical-harmonic basis for AO basis
# aosym='s4' means we want to use the symmetry of the two-electron integrals, always true for 2-electron integrals
eris = ao2mo.full(mol.intor('int2e_sph', aosym='s4'),mf.mo_coeff, compact=False)

# Convert to physicists' notation by swapping middle two indices
eris.shape = (norb, norb, norb, norb)
eris = eris.transpose(0, 2, 1, 3).copy()

# Flatten again and save to file
eris.shape = (norb**2, norb**2)
np.savetxt('eris.txt', eris, delimiter=',')

#### Output One-Electron Integrals in MO basis ####
h_core = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
np.savetxt('hcore.txt', h_core, delimiter=',')

print("Total HF energy :", mf.e_tot)
print("Electronic part:", mf.e_tot - mol.energy_nuc())
