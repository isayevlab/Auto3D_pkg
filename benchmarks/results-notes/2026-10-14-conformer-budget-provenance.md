# Conformer budget constants: checked against the cited paper's deposited dataset

The conformer budget's rotatable-bond coefficient (8.481) and exponent (1.642) are cited to
Chan, Morris and Hutchison, *J. Chem. Theory Comput.* 2021, 17, 2099-2106
(`https://doi.org/10.1021/acs.jctc.0c01213`). The paper's own text could not be reached: the
publisher page is paywalled, the ACS supporting-information PDF returned HTTP 403, the
ChemRxiv preprint (landing page and both version PDFs) and its public API endpoint returned a
Cloudflare interstitial under every route tried (the fetch tool, `curl` with a browser user
agent, and a text-extraction proxy), PubMed's record has no PMC copy of the article itself,
Semantic Scholar's graph API returned no record for the DOI or the title, a full-text search on
Europe PMC for the literal coefficient alongside "rotatable bonds" returned six hits that were
all coincidental numbers in unrelated papers, and the figshare API returned HTTP 404 for the
preprint item. What follows instead checks the two constants against the authors' own deposited
data and analysis repository, `github.com/ghutchis/molecular-entropies` (last pushed
2021-03-29, its README pointing at the same preprint item as the paper's raw data), which
contains 117,229 molecules across a training set, a holdout set and a peptide set.

## What the fit describes

Every deposited molecule carries two rotatable-bond counts and six columns giving the number of
conformers retained within an energy window of 1 to 6 units in 1-unit steps (a companion file
repeats the same counts in 0.5-unit steps, which is why the window is almost certainly
kcal/mol, though the unit itself is nowhere stated in the deposit). Summing the two widest-window
columns over all 117,229 rows gives 10,019,299 retained conformers for the 5-unit window and
13,205,265 for the 6-unit window; the paper's abstract states "about 12 million conformers"
sampled, which brackets those two sums and nothing else in the deposit is of that order. So the
fitted quantity is the number of distinct conformers a molecule retains inside a roughly 5 to 6
kcal/mol window of its minimum, not a sampling prescription and not a count needed to recover the
lowest-energy conformer within a tolerance.

Fitting `conformers = a * rotatable_bonds ** b` by several recipes, against both of the
deposit's two rotor-count columns:

| Fit recipe | RDKit default (strict) count: a, b | RDKit non-strict count: a, b |
|---|---|---|
| Per-molecule least squares on raw counts, every molecule with at least one rotatable bond | 8.608, 1.618 | 7.853, 1.548 |
| Group means weighted by group size, 1 to 20 rotatable bonds | 8.607, 1.618 | 7.853, 1.548 |
| Group means weighted by group size, 2 to 20 rotatable bonds | 8.559, 1.621 | 7.810, 1.550 |
| Group means, unweighted least squares | 24.02, 1.158 | 26.83, 1.016 |
| Log-log line through the group means | 10.74, 1.465 | 10.77, 1.363 |
| Log-log line through every molecule | 3.05, 1.833 | 2.70, 1.745 |
| Log-log line through the 75th percentile of each group | 9.50, 1.621 | 9.40, 1.513 |

The first three recipes land within about 1.5 percent of both implemented constants (8.481 and
1.642) against RDKit's default (strict) rotatable-bond count; none of the last four come close in
both parameters at once, and none of the seven come as close against RDKit's non-strict count,
where the same three recipes return 7.853 and 1.548 instead. The implemented constants are
consistent with being the paper's fitted numbers, reproduced to 1.5% from the deposit; the
residual is far outside the fit's statistical uncertainty on 114,678 points, so the exact sample
or recipe was not recovered.

Re-descriptoring 1,500 randomly sampled deposited molecules with the installed RDKit identifies
which of RDKit's rotatable-bond options the two deposited columns correspond to: the deposit's
`NumRotorsStrict` column matches RDKit's default (strict) count, which does not count amide C-N
or ester C-O single bonds as rotatable (`NumRotors >= NumRotorsStrict` on every row; 2,551
molecules have zero rotatable bonds under the strict count, against 1,867 under the non-strict
one), and the deposit's `NumRotors` column matches RDKit's non-strict count, which does count
them. `calculate_conformer_count` uses RDKit's default (strict) count -- the same count the fit
above reproduces the implemented constants against.

## The rotor-free regime

The formula's rotatable-bond term is exactly zero for a molecule with no rotatable bonds, because
`0 ** 1.642 == 0`; ring puckering and other non-torsional degrees of freedom are invisible to it.
The deposited data shows that rotor-free molecules are not low-conformer molecules:

| Rotor count | Molecules | Mean retained conformers | Median | 75th percentile | 90th percentile |
|---|---|---|---|---|---|
| 0 (RDKit default, strict count) | 2,551 | 25.1 | 5.0 | 15.0 | 36.0 |
| 0 (RDKit non-strict count) | 1,867 | 31.0 | 7.0 | 19.0 | 42.0 |
| 1 (RDKit default, strict count) | 11,002 | 14.7 | 4.0 | 13.0 | 32.0 |

A molecule with no rotatable bonds retains, on average, more distinct low-energy conformers than
one with a single rotatable bond, because the fit's independent variable cannot see ring puckering
or other macrocyclic motion.

Requesting more conformers for such a molecule under this package's own embedding settings (RDKit's
ETKDGv3, random seed 42, RMSD pruning threshold 0.3 restricted to heavy atoms, symmetry-aware
pruning, RDKit 2025.09.6) in fact keeps more of them:

| Molecule | Heavy atoms | Rotatable bonds | Previous budget | Kept at previous budget | Kept at 10 requested | Kept at 16 requested | Kept at 52 requested |
|---|---|---|---|---|---|---|---|
| Ethanol | 3 | 0 | 3 | 1 | 1 | 1 | 1 |
| Phenol | 7 | 0 | 7 | 1 | 1 | 1 | 1 |
| Acetamide | 4 | 0 | 4 | 1 | 1 | 1 | 1 |
| Pyridine | 6 | 0 | 6 | 1 | 1 | 1 | 1 |
| Chlorobenzene | 7 | 0 | 7 | 1 | 1 | 1 | 1 |
| Cyclohexane | 6 | 0 | 6 | 2 | 2 | 2 | 2 |
| Cycloheptane | 7 | 0 | 7 | 1 | 1 | 2 | 3 |
| Oxane | 6 | 0 | 6 | 1 | 2 | 3 | 4 |
| Piperidine | 6 | 0 | 6 | 1 | 2 | 3 | 3 |
| Cyclohexanone | 7 | 0 | 7 | 3 | 3 | 3 | 5 |
| Cyclooctane | 8 | 0 | 8 | 3 | 3 | 3 | 5 |
| Cyclooctanone | 9 | 0 | 9 | 8 | 9 | 11 | 16 |
| Cyclodecane | 10 | 0 | 10 | 9 | 9 | 13 | 18 |
| Alanine | 6 | 1 | 16 | 2 | 2 | 2 | 2 |
| Ethyl methyl sulfide | 4 | 1 | 16 | 4 | 4 | 4 | 5 |
| Glycerol | 6 | 2 | 52 | 9 | 5 | 7 | 9 |
| Alpha-D-glucopyranose | 12 | 1 | 16 | 12 | 10 | 12 | 28 |
| Steroid acetate | 21 | 1 | 21 | 6 | 4 | 6 | 6 |

For the rigid aromatics and acyclic small molecules the kept count saturates at 1 however large
the request, so a larger floor does not change the outcome for them. For the saturated and medium
rings it does not saturate: cyclooctanone keeps 8 of 9 requested, 9 of 10, 11 of 16 and 16 of 52,
and cyclodecane (previous budget already 10) keeps 9 of 10, 13 of 16 and 18 of 52 -- the pool is still growing when the
previous, heavy-atom-only floor stopped requesting more. The glycerol and alpha-D-glucopyranose
rows reproduce the figures already recorded elsewhere for this package, confirming this
measurement was made the same way.

The large rigid steroid acetate is the case for keeping the heavy-atom term alongside a constant
floor rather than replacing it: at 21 heavy atoms and one rotatable bond it keeps 6 of 21
requested today, but only 4 if its budget were cut to a bare constant floor of 10 -- large rigid
polycyclic molecules are exactly where "one request per heavy atom" happens to track the number
of ring-pucker degrees of freedom, so losing that term would cost conformers for them even while
gaining conformers for the smaller rotor-free rings above.

## Benchmark molecules affected

Applying a constant minimum of 16, for any molecule with two or more heavy atoms, to the 24
drug-like molecules in this package's optimization benchmark changes the requested count for five
of them -- the five with no rotatable bonds under either rotor-count convention -- and leaves the
other nineteen unchanged, because at one rotatable bond the formula already requests 16 or more:

| Molecule set | SMILES | Heavy atoms | Rotatable bonds (RDKit default, strict) | Rotatable bonds (RDKit non-strict) | Previous budget | New budget |
|---|---|---|---|---|---|---|
| small | `CCO` | 3 | 0 | 0 | 3 | 16 |
| small | `c1ccccc1O` | 7 | 0 | 0 | 7 | 16 |
| small | `CC(=O)N` | 4 | 0 | 0 | 4 | 16 |
| small | `CC(N)C(=O)O` | 6 | 1 | 1 | 16 | 16 |
| small | `c1ccncc1` | 6 | 0 | 0 | 6 | 16 |
| small | `CSCC` | 4 | 1 | 1 | 16 | 16 |
| small | `OCC(O)CO` | 6 | 2 | 2 | 52 | 52 |
| small | `Clc1ccccc1` | 7 | 0 | 0 | 7 | 16 |
| medium | `CC(=O)Nc1ccc(O)cc1` | 11 | 1 | 2 | 16 | 16 |
| medium | `CN1CCC[C@H]1c1cccnc1` | 12 | 1 | 1 | 16 | 16 |
| medium | `c1ccc2[nH]ccc2c1C(=O)NCC` | 14 | 2 | 3 | 52 | 52 |
| medium | `OC(=O)c1ccccc1OC(C)=O` | 13 | 2 | 3 | 52 | 52 |
| medium | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` | 15 | 4 | 4 | 165 | 165 |
| medium | `Fc1ccc(cc1)C(=O)CCCN1CCCCC1` | 18 | 5 | 5 | 238 | 238 |
| medium | `CCN(CC)CCNC(=O)c1ccc(N)cc1` | 17 | 6 | 7 | 321 | 321 |
| medium | `CC(C)(C)NCC(O)c1ccc(O)c(CO)c1` | 17 | 4 | 5 | 165 | 165 |
| large | `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` | 23 | 4 | 5 | 165 | 165 |
| large | `CN1C2CCC1C(C(=O)OC)C(OC(=O)c1ccccc1)C2` | 22 | 3 | 5 | 103 | 103 |
| large | `Cc1ccc(cc1)S(=O)(=O)NC(=O)NN1CCCCCC1` | 21 | 3 | 5 | 103 | 103 |
| large | `CC(=O)OC1CC2CCC3C(CCC4(C)C3CCC24C)C1` | 21 | 1 | 2 | 21 | 21 |
| large | `COc1cc2c(cc1OC)C(=O)C(CC2)Cc1ccc(OC)cc1` | 24 | 5 | 5 | 238 | 238 |
| large | `OC(=O)C1CCCN1C(=O)C(Cc1ccccc1)NC(=O)OCc1ccccc1` | 29 | 7 | 10 | 414 | 414 |
| large | `CN(C)CCCN1c2ccccc2CCc2ccccc21` | 21 | 4 | 4 | 165 | 165 |
| large | `CC(C)NCC(O)COc1cccc2ccccc12` | 19 | 6 | 6 | 321 | 321 |

The two molecules the package's documentation already quotes a budget for, glycerol (two
rotatable bonds, budget 52, nine conformers kept) and alpha-D-glucopyranose (one rotatable bond,
budget 16, twelve conformers kept), are unaffected because both budgets already sit at or above
the new floor.

## What remains unverified

Three facts about the cited paper could not be confirmed, because its own text was never reached
by any route tried (listed above): the physical unit of its retained-conformer energy window
(almost certainly kcal/mol, inferred only from a companion file's half-unit steps); the
conformer-generation program, optimization level and energy method the paper itself used to
decide which conformers to retain; and the paper's own stated definition of a rotatable bond
(inferred here only by finding which of the deposit's two rotor-count columns reproduces the
fitted constants, not by reading the paper's definition directly). None of these affect the
numbers used in this package's budget, which were checked against the deposited data rather than
the text, but a reader who later gains access to the paper's text should confirm them against it.
