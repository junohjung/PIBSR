# Additional figure reproduction

These commands run from the repository root. NumPy, SciPy, Matplotlib and
Pillow are required for Python postprocessing. Volume rendering additionally
uses ParaView 5.13.2 (`pvpython` on PATH).

## Figure 8

```sh
python3 reproducibility/scripts/make_figure8.py
```

The five fixed coarse observations include actual pressure. Five matched
21-by-21 reference frames and five 201-by-201 display frames are supplied.
Pressure errors use independently mean-aligned fields on the matched grid.
The script also recalculates the appendix flow-discretization diagnostics.
The zero pressure placeholder in the velocity-fitting trajectory is not used.
No Firedrake solver, environment or data-generation script is included.

## Figures 11 and 12

```sh
python3 reproducibility/scripts/verify_volume_fields.py
pvpython reproducibility/scripts/render_selected_volumes.py burgers
pvpython reproducibility/scripts/render_selected_volumes.py cavity
python3 reproducibility/scripts/make_figure11.py
pvpython reproducibility/scripts/render_selected_volumes.py burgers --slices
pvpython reproducibility/scripts/render_selected_volumes.py cavity --slices
python3 reproducibility/scripts/make_figure12_slices.py
python3 reproducibility/scripts/make_figure12.py
```

Outputs are written under `reproducibility/work/figures/`. The independent
SciPy decoder writes reconstructed arrays and its comparison report under
`reproducibility/work/volume_verification/`. It evaluates the same cubic,
uniform, clamped spline spaces as the released selected models. The plain
Burgers coefficient array is a lossless export of its released checkpoint.
The existing cavity coefficient file is read directly.

ParaView states preserve the original cameras, transfer functions and slice
planes, with only input paths relocated. Rendering uses the fixed decoded
VTK grids; the NumPy grids allow independent coefficient-to-field checks.
Rendering can vary slightly across graphics drivers. Figure 12 is a portable
composition, not a pixel-identical copy of the final typesetter artwork.
Its third cavity slice is at t=5.005, displayed as 5.0 in the submitted
artwork. The new label retains the exact sampled time.

The separate `pinn/` extension provides PINN checkpoint
evaluation. Alternative control-lattice optimizations and full timing/rank
campaigns are not covered by these figure scripts. This must not be described
as complete end-to-end reproduction of every experiment.
