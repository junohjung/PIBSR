#!/usr/bin/env python3
"""Independently decode the selected models on the Figure 11/12 volume grids."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]


def basis(points, count, length):
    knots = np.r_[np.zeros(4), np.arange(1, count - 3) / (count - 3), np.ones(4)]
    return BSpline.design_matrix(points / length, knots, 3).toarray()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'work/volume_verification')
    args = parser.parse_args()
    data_root = ROOT / 'figure_data/selected_volumes'
    for name, digest in json.loads((data_root / 'manifest.json').read_text()).items():
        if hashlib.sha256((data_root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Checksum mismatch: {name}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for case, counts, lengths, coefficient_path in (
        ('burgers', (30, 30, 30), (2., 2., 2.), data_root / 'burgers_coefficients.txt'),
        ('cavity', (50, 20, 20), (10., 1., 1.), ROOT / 'checkpoints/lid_driven_cavity/stage_b/gl4_hybrid__gamma_0p04/final_coefficients_v1.txt'),
    ):
        with np.load(data_root / f'{case}.npz', allow_pickle=False) as archive:
            grids = [archive[name] for name in ('t', 'y', 'x')]
            reference = archive['speed'].copy()
        coefficients = np.loadtxt(coefficient_path).reshape((-1, *counts))[:2]
        matrices = [basis(points, count, length) for points, count, length in zip(grids, counts, lengths)]
        velocity = np.einsum('ta,yb,xc,fabc->ftyx', *matrices, coefficients, optimize=True)
        speed = np.hypot(*velocity)
        difference = float(np.max(np.abs(speed - reference)))
        if not np.all(np.isfinite(speed)) or difference > 1e-7:
            raise ValueError(f'{case}: decoded speed mismatch {difference:.17g}')
        report[case] = {'shape': list(speed.shape), 'maximum_absolute_difference': difference,
                        'comparison_tolerance': 1e-7, 'coefficient_sha256': hashlib.sha256(coefficient_path.read_bytes()).hexdigest()}
        np.savez_compressed(args.output_dir / f'{case}_decoded.npz', t=grids[0], y=grids[1], x=grids[2], u=velocity[0], v=velocity[1], speed=speed)
    (args.output_dir / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
