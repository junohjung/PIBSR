# 1-D observation generation

No byte-identifiable source for the historical 1-D observation generator was
found in the manuscript revision archives. The reproducible input is therefore
the packaged endpoint-inclusive 400-frame observation archive under
`../../../data/one_d_convection_diffusion`, together with its
`SELECTION_INPUT_MANIFEST.json`.

This directory intentionally contains no substitute generator: recreating one
from the prose-level discretization description would be a new implementation,
not the calculation that produced the submitted results.

`reference_resolution.py` is a separate, portable adaptation of the exact
analytical-series evaluator used by the final V12 postselection. It compares
the submitted 900-mode/4097-node reference with the doubled
1800-mode/8193-node evaluation on the packaged `60 x 400` grid:

```bash
python3 reference_resolution.py --dry-run
python3 reference_resolution.py
```

The full calculation requires NumPy; dry-run path validation does not.
