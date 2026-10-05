# Package-wide verification calculations

Run the standard-library-only audit from this directory:

```bash
python3 audit_results.py
```

The five full-precision values in `reported_metrics.json` are reconstructed
from the archived source records, whose hashes are checked before any values
are accepted.

| Verification | Packaged source | Interpretation |
| --- | --- | --- |
| Burgers strong-term directional derivative | `results/verification/burgers_strong_directional_gradient.log` | Three independent perturbation directions; maximum relative discrepancy |
| Navier--Stokes strong-term directional derivative | `results/verification/cavity_strong_directional_gradient_historical.log` | Five independent directions; maximum relative discrepancy |
| Burgers integral-balance closure | `results/two_d_burgers/stage_b_interface_compatibility/GL_BALANCE_SUMMARY.json` | Maximum direct-flux versus integrated-strong mismatch |
| Cavity cross-implementation comparison | `results/lid_driven_cavity/protected_gl8_v4/strong_only/run_audit.json` | Maximum pointwise absolute difference across mass and momentum components |
| Physical-coordinate derivative check | `results/two_d_burgers/postselection/POSTSELECTION_SUMMARY.json` | Maximum scaled centered-difference discrepancy across the five audited models |

The two text logs are byte-identical copies of the compact implementation
checks under the V8 reviewer-reproducibility archive. They retain historical
absolute paths as inert log text so that their original bytes remain
verifiable. In particular, the cavity log used full pressure data and checks
the Navier--Stokes strong-term implementation; it is not a run of the active
ramped-cavity fitting protocol, which excludes pressure observations.

The archived logs are sufficient to audit the two directional-derivative
values. Re-executing those compiled checks requires the historical evaluator
build and is outside the portable, read-only calculation layer.
