"""Physical and computational constants for Auto3D."""
from __future__ import annotations

# Energy conversion factors (CODATA 2018)
HARTREE_TO_EV = 27.211386245988  # 1 Hartree in eV
EV_TO_KCAL_PER_MOL = 23.060547830619026  # 1 eV in kcal/mol (original Auto3D value)
HARTREE_TO_KCAL_PER_MOL = 627.50947337481  # 1 Hartree in kcal/mol

# Geometry thresholds
BOND_STRETCH_TOLERANCE = 1.25  # Maximum bond stretch factor
COLLISION_THRESHOLD = 1.1  # Å, minimum distance for clash detection
MIN_ATOM_DISTANCE = 0.9  # Å, minimum allowed interatomic distance

# Conformer generation limits
MAX_CONFORMERS_CAP = 1000  # Maximum conformers per molecule

# Stereoisomer enumeration limit. RDKit's StereoEnumerationOptions defaults
# maxIsomers to 1024, which silently truncates molecules with >=11 unspecified
# stereocenters. Raise the cap high so realistic inputs are not silently lost,
# and warn when the enumerator returns exactly the cap (likely truncated).
MAX_STEREOISOMERS = 2 ** 16  # 65536

# Chunk sizing
DEFAULT_CAPACITY = 42  # molecules per GB of GPU/CPU memory for chunk sizing

# Conformer generation formula coefficients
# Based on: https://doi.org/10.1021/acs.jctc.0c01213
CONFORMER_ROTATABLE_COEFF = 8.481  # Coefficient for rotatable bond count
CONFORMER_ROTATABLE_EXP = 1.642    # Exponent for rotatable bond count
CONFORMER_MULTIPLIER = 2           # Multiplier for the formula
CONFORMER_RANDOM_SEED = 42         # Random seed for reproducible embedding

# Optimization sentinel values
INITIAL_FMAX_SENTINEL = 999.0  # Initial value for max force (unconverged)
INITIAL_ENERGY_SENTINEL = 999.0  # Initial value for energy (unconverged)

# Thermodynamics
STANDARD_PRESSURE = 101325  # Pa

# Model names
MODEL_AIMNET = "AIMNET"
MODEL_ANI2X = "ANI2x"
MODEL_ANI2XT = "ANI2xt"

# Supported model names (for validation)
SUPPORTED_MODELS = frozenset({MODEL_AIMNET, MODEL_ANI2X, MODEL_ANI2XT})

# Backward-compatible alias: "AIMNET" now maps to the aimnet registry default.
DEFAULT_AIMNET_MODEL = "aimnet2"
# Built-in (non-aimnet) engines kept for back-compat.
BUILTIN_ANI_MODELS = frozenset({MODEL_ANI2X.upper(), MODEL_ANI2XT.upper()})

# Default optimization parameters
DEFAULT_RMSD_THRESHOLD = 0.3  # Angstrom, for duplicate conformer removal
# eV, energy tolerance for the duplicate-conformer test. RMSD dedup compares
# heavy-atom skeletons only, so conformers differing solely in an O-H / N-H rotor
# orientation collapse to RMSD~=0 even though they are distinct minima with
# different energies. Two structures are treated as duplicates only when their
# heavy-atom RMSD is below threshold AND their energies agree within this
# tolerance, so genuine rotamers (which the H-rich conformer budget deliberately
# samples) survive. ~0.23 kcal/mol -- above post-optimization fp32 energy noise
# (~1e-3 eV) so truly identical conformers still dedup.
DEFAULT_DUPLICATE_ENERGY_TOL = 0.01
DEFAULT_CONVERGENCE_THRESHOLD = 0.01  # eV/Angstrom, force convergence
# eV/Angstrom, the deliberately tighter pre-optimization force tolerance used by
# the thermochemistry path (calc_thermo): a true minimum is needed before a
# Hessian/frequency calculation, unlike conformer generation where 0.01 suffices.
DEFAULT_THERMO_CONVERGENCE_THRESHOLD = 2e-4
DEFAULT_OPT_STEPS = 2000  # Maximum optimization steps
DEFAULT_PATIENCE = 250  # Steps before dropping oscillating conformer
DEFAULT_BATCHSIZE_ATOMS = 1024  # Atoms per batch for GPU optimization
DEFAULT_ENERGY_CLUSTER_WINDOW = 0.1  # eV, for RMSD clustering
# eV, energy convergence threshold (~0.02 kcal/mol). Must exceed the fp32 ULP at
# typical molecular total energies (~thousands of eV => ULP ~1e-3 eV); a smaller
# value sits below float32 noise and the energy criterion would never fire
# (review finding #23).
DEFAULT_ENERGY_TOL = 1e-3
DEFAULT_ENERGY_PATIENCE = 3  # Steps energy must be stable before converging
DEFAULT_RANDOM_SEED = 42  # Default random seed for reproducibility

# A linear molecule has one vanishing principal moment of inertia; a bent one
# has two comparable large moments and a much smaller third. This is the
# largest smallest-to-largest moment ratio still treated as linear. It is
# dimensionless, so unlike an absolute coordinate tolerance it behaves the same
# for a diatomic and for a long chain.
#
# Placed by measurement, not by fitting test cases: sweeping a symmetric
# triatomic's bend angle puts this boundary at ~22 degrees from linear (ratio
# vs. angle-off-linear for a CO2-like triatomic: 10 deg -> 2.1e-3, 20 deg ->
# 8.4e-3, 22 deg -> 1.0e-2, 30 deg -> 1.9e-2). That separates a linear
# molecule left imperfectly optimized -- CO2's real bending mode is thermally
# populated to several degrees at room temperature, e.g. 20 degrees off linear
# still measures only 8.4e-3 -- from a molecule that is genuinely bent, e.g.
# NO2, the most nearly-linear common bent species, at 134 degrees (46 degrees
# off linear), which measures 5.2e-2. 1e-2 sits about an order of magnitude
# above the thermal case and ~5x below the genuinely bent one.
LINEARITY_MOMENT_RATIO = 1e-2

# Imaginary modes below this magnitude (cm^-1) are numerical artifacts of an
# NNP Hessian at conformer-generation convergence; above it, the structure is a
# saddle point and its "free energy" is not a minimum's.
IMAGINARY_MODE_CUTOFF_CM = 50.0

# Optional Truhlar-style raising: real modes below this wavenumber are raised
# to it before the entropy sum. A 10 cm^-1 torsion contributes ~2.4 kcal/mol to
# -T*S at 298 K, which swamps most differences this module resolves. Zero
# disables raising, which is the default so existing numbers do not move
# without the caller asking.
LOW_FREQUENCY_CUTOFF_CM = 0.0

# eV per wavenumber, for reporting vibrational energies in cm^-1.
EV_PER_WAVENUMBER = 1.0 / 8065.54429
