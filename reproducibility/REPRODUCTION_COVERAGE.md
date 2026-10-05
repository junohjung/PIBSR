# Reproduction coverage

## Supported operations

- `scripts/verify_package.sh`: verifies file integrity, point roles, archived
  numerical results, and Python tests. It does not retrain the models.
- `scripts/build_and_test.sh all`: builds the three case executables and runs
  18 focused C++ tests.
- `scripts/run_selected.py CASE --execute`: refits the selected model from
  the released fixed Stage-A initialization. This does not rerun the entire
  initialization and parameter-selection campaign.
- `calculations/lid_driven_cavity/postprocess_raw.py`: recomputes 40 checks
  from retained field, pressure, and residual arrays.
- `scripts/make_figures.py all`: creates eight main-text numerical figures
  from retained inputs and curve tables. It does not retrain contextual PINNs.
- `scripts/make_figure5.py`: regenerates all 30 Figure 5 panels from the
  observations and reference snapshots, using the production layout.

## Limits

The optional `pinn/` subset includes the original 1D, five-seed Burgers and
five-seed cavity checkpoints. Evaluation reproduces the main-text PINN curves;
training from initialization is not included. See `pinn/README.md`. Alternative
control-lattice fits, complete rank/conditioning and timing sweeps, and some
supplementary PINN evaluations are still not standalone workflows. Lightweight
result audits must not be described as rerunning those experiments.

Figure 8 now has the actual coarse pressure, matched reference and fine display
frames, together with a standalone renderer and metric calculator. The fitting
pressure column remains an inactive zero placeholder and is not used.
Figures 11 and 12 have portable ParaView states, decoded grids, independent
coefficient-to-volume verification, and composition scripts. Figure 12 is not
pixel-identical to the typesetter artwork. See EXTENDED_FIGURES.md for commands.
Not all supplementary figure workflows are included.

The cavity observations are provided as fixed data. No Firedrake solver,
environment, or data-generation scripts are distributed.

## Presentation corrections

The 1D initial-profile comparison now uses the continuous idealization; the
fitting initial target remains the stored first snapshot. Contour extensions
show out-of-range colors without blank regions, and the 1D regional-balance
axes include all plotted values. The 400 snapshots retain their values and
the reconstruction time assignment `t_j = 0.4*j/399`; historical source times
were `0.001*j`. Figure 5 uses the per-time error limits in FIGURE5_UPDATE.md.
No observation, model coefficient, numerical solver, or result-table value
was changed by these presentation corrections.
