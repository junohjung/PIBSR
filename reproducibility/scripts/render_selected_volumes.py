#!/usr/bin/env pvpython
"""Render the Figure 11/12 volume panels with ParaView 5.13.2."""
import argparse
import hashlib
import json
from pathlib import Path

from paraview.simple import GetActiveView, LoadState, Render, SaveScreenshot

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=('burgers', 'cavity'))
    parser.add_argument('--slices', action='store_true', help='Add the Figure 12 slice planes.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'work/figures')
    args = parser.parse_args()
    data = ROOT / 'figure_data/selected_volumes'
    hashes = json.loads((data / 'manifest.json').read_text())
    state_name = args.case + ('_slices' if args.slices else '') + '.pvsm'
    for name in (state_name, args.case + '.vtk'):
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError('Checksum mismatch: ' + name)
    LoadState(str(data / state_name), data_directory=str(data), restrict_to_data_directory=True)
    view = GetActiveView()
    Render(view)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_volume_slices.png' if args.slices else '_volume.png'
    SaveScreenshot(str(args.output_dir / (args.case + suffix)), view, ImageResolution=[2800, 2200])


if __name__ == '__main__':
    main()
