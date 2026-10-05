# Figure reproduction

For Figure 5 with the corrected error scales and production typography, see
[Figure 5 reproduction](FIGURE5_UPDATE.md). The dedicated command is
`python3 reproducibility/scripts/make_figure5.py`.

The package includes a compact NumPy/Matplotlib renderer for eight numerical
figures in the main article. From the repository root, create every supported
figure with:

```sh
python3 reproducibility/scripts/make_figures.py all
```

The command writes a PDF, a 300-dpi PNG preview, and
`figure_manifest.json` under `reproducibility/work/figures/`. The generated
`work/` tree is intentionally excluded from the release checksum inventory and
ZIP; users create it locally from the hash-checked inputs.

NumPy and Matplotlib are required. They can be installed in a virtual
environment with:

```sh
python3 -m pip install -r reproducibility/requirements-figures.txt
```

Case groups and individual figures can also be selected:

```sh
python3 reproducibility/scripts/make_figures.py one_d
python3 reproducibility/scripts/make_figures.py burgers
python3 reproducibility/scripts/make_figures.py cavity
python3 reproducibility/scripts/make_figures.py burgers_discrepancy
python3 reproducibility/scripts/make_figures.py --list
```

Use `--output-dir /path/to/output` to choose another destination.

## Supported figures

| Generated file | Submitted figure | Packaged numerical source |
|---|---|---|
| `one_d_observations.pdf` | `Figure2_1D_case1.pdf` | the 400 released observation frames and the documented analytical series |
| `one_d_weight_selection.pdf` | `Figure40_1D_strong_weight_selection_geometry_edits.pdf` | the 26-candidate strong-form sweep and sealed Pareto geometry |
| `one_d_balance_histories.pdf` | `1D_integralbalance2.pdf` | consolidated GL8 regional and whole-domain histories |
| `burgers_discrepancy.pdf` | `Figure9_2Dburger_datapdf_absErr_FD_RK4.pdf` | released forced observations and unforced FD/RK4 reference frames |
| `burgers_weight_selection.pdf` | `Figure36_Burgers_three_stage_weight_selection_edits.pdf` | beta, kappa, and gamma selection tables |
| `burgers_balance_histories.pdf` | `2D_integralbalance_Burgers2.pdf` | consolidated Design-B balance histories, including the contextual PINN curve |
| `cavity_weight_selection.pdf` | `weight_model_selection_cavity2.pdf` | the complete 4-by-4 family-selection coordinate table |
| `cavity_balance_histories.pdf` | `2D_integralbalance_cavity2.pdf` | deterministic GL8 histories and contextual five-seed PINN aggregates |

The submitted PDFs received final typography and layout edits in Adobe
Illustrator. The generated files therefore reproduce the numerical content,
panel structure, selections, scales, and scientific labels, but are not
expected to be byte-identical to the submitted artwork.

## Deliberate limits

The 1D dashed initial curve represents the continuous initial-profile
idealization, not the interpolated first fitting snapshot. The 400 released
snapshot values are unchanged and use reconstruction times `0.4*j/399`.
Their historical source times were `0.001*j`, for `j=0,...,399`.
Filled contours use colorbar extensions to mark values outside the displayed
range; values are not removed or numerically clipped. Regional balance plot
limits include every plotted data point.

The workflow diagram is an explanatory drawing rather than a numerical
output. The coarse/high-resolution cavity comparison is not regenerated
because the complete high-resolution trajectory is not distributed. The
selected-field composite and three-dimensional space-time rendering are also
not regenerated because their final visualization/export inputs are outside
the curated release. In particular, no cavity flow solver, solver environment,
or data-generation program is introduced by the figure command; it reads only
the distributed data and result tables.
