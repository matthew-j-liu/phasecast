# Figure manifest

One row per paper figure. Every figure regenerates from one entry-point
script in `figures/src/`, reading only existing tracked metrics/JSON files
(never rerunning instruments that write into `runs/`, `results/`, or
exploration results). Render mode: **PNG during iteration, PDF at
finalization** -- current mode: PNG.

Rules of the road:

- `results/trajectory.png` is internal-only, never paper material.
- The climatology-null panels (`nll_distribution.png`, `presence_brier.png`
  in `results/<tag>/`) are internal-only; no paper figure may score against
  the climatology null.
- A coverage panel is gated on #59 unless a calibration claim actually
  ships.
- Any quartz-improvement visual is gated on #57.
- Thin classes (ledger instrument stanza) are caveated in captions wherever
  a per-class slice appears.
- Card-family drawing conventions: see the realrock-figure-conventions
  memory (CARD_INTERVAL_Q vs share_interval default; the `_unit_bootstrap`
  subset footgun).

| figure | script | sources | checkpoint / instrument / slice | status |
|---|---|---|---|---|
| (none built yet -- candidates listed in `paper/outline.md`, pending owner approval of the structure) | | | | |
