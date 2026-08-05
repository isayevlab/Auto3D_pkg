# Auto3D examples

Runnable notebooks. Each one is self-contained and uses the files in
[`files/`](files/). They are also rendered in the
[documentation](https://auto3d.readthedocs.io/), which builds this directory
directly — so a notebook fixed here is fixed there.

Most of these call a neural network potential. The first run downloads the
AIMNet2 weights into `~/.cache/aimnet` (a few hundred MB, once per model), so
give the first cell a moment. Pass `use_gpu=False` if you have no CUDA device —
Auto3D treats a requested-but-missing GPU as a fatal error, not a fallback.

## Start here

| Notebook | What it covers |
|---|---|
| [quickstart](quickstart.ipynb) | Smallest useful example: SMILES in, ranked conformers out |
| [tutorial](tutorial.ipynb) | The same ground at a slower pace, with the output explained |
| [performance_tuning](performance_tuning.ipynb) | Batch size, memory, GPU selection, what actually moves the needle |
| [large_scale_processing](large_scale_processing.ipynb) | Chunking and running a library that will not fit in memory |

## Property calculators

Each wraps one Python API function, and each has a matching CLI command.

| Notebook | API | CLI |
|---|---|---|
| [single_point_energy](single_point_energy.ipynb) | `calc_spe` | `auto3d energy` |
| [geometry_optimization](geometry_optimization.ipynb) | `opt_geometry` | `auto3d optimize` |
| [thermodynamic_calculation](thermodynamic_calculation.ipynb) | `calc_thermo` | `auto3d thermo` |
| [tautomer](tautomer.ipynb) | `get_stable_tautomers` | `auto3d tautomers` |

`calc_thermo` needs the `ase` extra: `pip install "Auto3D[ase]"`.

## Drug discovery

| Notebook | What it covers |
|---|---|
| [virtual_screening](virtual_screening.ipynb) | Preparing a screening library |
| [tautomer_protomer_analysis](tautomer_protomer_analysis.ipynb) | Tautomer and protomer states for drug-like molecules |
| [stereochemistry](stereochemistry.ipynb) | Enumerating and keeping track of stereoisomers |
| [docking_integration](docking_integration.ipynb) | Handing conformers to a docking program |

## Computational chemistry

| Notebook | What it covers |
|---|---|
| [reaction_thermodynamics](reaction_thermodynamics.ipynb) | Reaction energies from optimized conformers |
| [boltzmann_populations](boltzmann_populations.ipynb) | Populations and conformational averaging |
| [strain_energy](strain_energy.ipynb) | Ligand strain |
| [molecular_descriptors](molecular_descriptors.ipynb) | 3D descriptors for ML/QSAR |

## Integration and custom models

| Notebook | What it covers |
|---|---|
| [md_preparation](md_preparation.ipynb) | Preparing structures for molecular dynamics |
| [qm_refinement](qm_refinement.ipynb) | Refining Auto3D output with a QM program |
| [using_custom_NNP](using_custom_NNP.ipynb) | Wrapping your own potential to the `CustomNNP` contract |
| [tautomer_with_userNNP](tautomer_with_userNNP.ipynb) | The same, driving tautomer ranking |

Writing a custom potential is the one place where getting the contract slightly
wrong fails in a confusing way. Read
[the custom NNP guide](https://auto3d.readthedocs.io/en/latest/howto/custom_nnp.html)
alongside those last two: your model implements
`forward(species, coords, charges) -> energies` (species **first**, energies
only), which is the mirror image of Auto3D's internal adapter interface.
