# Assessment: Proper Thermodynamic Framing for MAGEMin-Based Label Generation

## Core Issue

Current pipeline generates training labels using **random equilibrium snapshots**:

> bulk composition + (P, T, H₂O) → equilibrium assemblage

This produces a **magma-in-progress distribution**, not a **final rock distribution**.

As a result:
- ~70% mean melt fraction (liq) in training data
- Model learns crystallization-state systems, not hand-sample mineralogy
- Mismatch with target use case (room-T, fully crystallized rocks)

---

## Key Insight

Igneous rocks are **path-dependent systems**, not single-state equilibria.

Correct mapping is:

> bulk composition → cooling path → final modal mineralogy

Single (P, T) equilibrium is insufficient because it ignores:
- fractional crystallization
- melt removal
- kinetic arrest
- glass formation vs full crystallization

---

## What PetThermoTools Demonstrates

Found a resource that uses MAGEMin to simulate crystallization: https://petthermotools.readthedocs.io/en/latest/Examples/CrystallisationTests/BasalticCrystallisation_withMAGEMin.html

The referenced MAGEMin crystallization example implements:

- **fractional crystallization**
- stepwise cooling
- liquid-line-of-descent evolution
- progressive crystal removal (`Frac_solid = True`)

This is fundamentally different from:
- single-point equilibrium calculations

However:
- it still emphasizes magma evolution, not final room-temperature rock state

---

## Recommended Correction Strategies

### Option 1 (Minimal Fix): Near-Solidus Equilibrium
Replace arbitrary T sampling with:

- Estimate solidus per bulk composition
- Sample temperatures that are 20 to 100 C below the solidus 
- Compute equilibrium assemblage
- Discard melt phase

Pros:
- Simple
- Stable
- Removes bulk of melt-dominated labels

Cons:
- Still not path-dependent
- May over-equilibrate mineral assemblages

---

### Option 2 (Preferred): Fractional Crystallization Path

Simulate cooling explicitly:

1. Start above liquidus
2. Stepwise cooling
3. Fractional crystallization at each step
4. Track residual melt
5. Stop when melt fraction approaches 0


Final output:
- fully crystallized assemblage for plutonic rocks
- glass retained for volcanic endmembers if desired

Pros:
- Physically correct
- Captures igneous differentiation
- Produces realistic modal mineralogy

Cons:
- More compute expensive
- Requires careful implementation

---

## Volcanic vs Plutonic Handling

Critical ambiguity:

- volcanic rocks → melt quenched → glass is real phase
- plutonic rocks → melt crystallizes → no glass present

Without external metadata, this is unidentifiable from bulk chemistry alone.

Recommended handling:
- Default: predict fully crystalline modal mineralogy
- Optional extension: separate "glass fraction" head or auxiliary label

---

## Partial Melts / Quenched Rocks

Examples:
- ignimbrites
- welded tuffs
- vitrophyres

These require mixed representation:
- crystalline phases + glass fraction

Best handled as:
- secondary output or future extension
- not primary label target in v1

---

## Failure Modes in Current Approach

1. **Melt inflation**
- equilibrium at high T produces unrealistically high liquid fractions

2. **Metamorphic drift**
- low-T equilibration pushes assemblages into greenschist/amphibolite regimes

3. **Path ignorance**
- ignores fractional crystallization history

4. **Non-unique inversion**
- bulk → modal mineralogy is not uniquely invertible without thermodynamic priors

---

## Recommendation Summary

- Do NOT use broad random (P, T) equilibrium sampling
- Do NOT use deep subsolidus equilibration (e.g., 400°C granites)
- DO use either:
- near-solidus equilibration (fast baseline fix), or
- fractional crystallization path modeling (preferred, physically correct)

---

## Bottom Line

The current dataset is sampling the wrong thermodynamic object.

The correct object is not:
> equilibrium state at (P, T)

but:
> endpoint of a cooling/crystallization path

Fixing this alignment will likely resolve the dominant melt-fraction bias and improve physical realism of predicted modal mineralogy.