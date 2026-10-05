#!/usr/bin/env python3
"""Compose Figure 12 from rendered volume panels and fixed temporal slices."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image

from make_figure12_slices import (ROOT, BURGERS_TIMES, CAVITY_TIMES, BURGERS_RANGE,
                                CAVITY_RANGE, exact_slices, linear_time_slices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'work/figures')
    args = parser.parse_args()
    fig = plt.figure(figsize=(10.7, 5.4))
    outer = fig.add_gridspec(1, 2, left=.045, right=.92, bottom=.10, top=.96, wspace=.23)
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
    for column, (case, times, limits, length) in enumerate((
        ('burgers', BURGERS_TIMES, BURGERS_RANGE, 2.), ('cavity', CAVITY_TIMES, CAVITY_RANGE, 1.),
    )):
        grid = outer[column].subgridspec(2, 4, height_ratios=[4., 1.], hspace=.02, wspace=.08)
        ax = fig.add_subplot(grid[0, :])
        ax.imshow(Image.open(args.output_dir / (case + '_volume_slices.png')))
        ax.set_axis_off()
        ax.text(.02, .96, '(' + 'ab'[column] + ')', transform=ax.transAxes)
        ax.text(.72, .90, '$x$', transform=ax.transAxes)
        ax.text(.23, .90, '$y$', transform=ax.transAxes)
        ax.text(.02, .47, '$t$' if case == 'burgers' else '$0.1t$', rotation=90, transform=ax.transAxes)
        ticks = [0., .25, .5, .75, 1.] if case == 'burgers' else [0., .2, .4, .6, .75]
        colorbar = fig.colorbar(ScalarMappable(norm=Normalize(*limits), cmap='turbo'), ax=ax, fraction=.025, pad=.015, shrink=.7, ticks=ticks)
        colorbar.set_label(r'$\|\mathbf{u}_h\|_2$')
        colorbar.ax.tick_params(labelsize=8)
        with np.load(ROOT / 'figure_data/selected_volumes' / (case + '.npz'), allow_pickle=False) as archive:
            if case == 'burgers':
                planes, _ = exact_slices(archive['speed'], archive['t'], times)
            else:
                planes, _ = linear_time_slices(archive['speed'], archive['t'], times, display_time_scale=.1)
        for i, (plane, time) in enumerate(zip(planes, times)):
            axis = fig.add_subplot(grid[1, i])
            axis.imshow(plane, origin='lower', extent=(0., length, 0., length), cmap='turbo', vmin=limits[0], vmax=limits[1], interpolation='bilinear')
            axis.set_title(f'$t={time:g}$', fontsize=8)
            axis.set_xlabel('$x$', fontsize=8)
            axis.set_xticks([0., length / 2, length])
            axis.set_yticks([0., length / 2, length])
            axis.tick_params(labelsize=7)
            axis.text(.05, .80, str(i + 1), color='white', transform=axis.transAxes, fontsize=8)
            if i:
                axis.tick_params(labelleft=False)
            else:
                axis.set_ylabel('$y$', fontsize=8)
    fig.savefig(args.output_dir / 'Figure12.pdf', dpi=400)
    fig.savefig(args.output_dir / 'Figure12.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
