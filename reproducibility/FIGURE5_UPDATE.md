# Figure 5 presentation correction

This correction adds a production-layout renderer for Figure 5. Numerical
inputs, model coefficients, solver code, and tabulated results are unchanged.
All 30 data panels are regenerated from the released Burgers snapshots.
The layout template contains blank data panels and embedded fonts.

From the repository root, using Python 3.12 or later:

```sh
python3 -m pip install -r reproducibility/requirements-figure5.txt
python3 reproducibility/scripts/make_figure5.py
```

The PDF and a JSON report of input hashes, error maxima, and color limits are
written to `reproducibility/work/figures/Figure5.pdf` and `Figure5.json`.
With Poppler installed, a preview can be generated with:

```sh
pdftoppm -r 180 -singlefile -png reproducibility/work/figures/Figure5.pdf reproducibility/work/figures/Figure5
```

The absolute-error ranges are [0, 0.00001], [0, 0.07], [0, 0.26],
[0, 0.30], and [0, 0.30] at t = 0, 0.5, 1, 1.5, and 2, respectively.
Each range is shared by u and v at the same time; ranges differ across time.
None of the raw absolute errors is clipped. Bilinear interpolation is used
only for displaying the fields; it does not alter the numerical inputs.
The general `make_figures.py burgers_discrepancy` command uses these same
ranges with its Matplotlib layout. Use `make_figure5.py` for the production
fonts, panel positions, and page size.
