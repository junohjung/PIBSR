"""Regression checks for the additional fixed-data figure inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class ExtendedFigureTests(unittest.TestCase):
    def test_volume_input_hashes(self):
        directory = ROOT / 'figure_data/selected_volumes'
        for name, expected in json.loads((directory / 'manifest.json').read_text()).items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected)

    def test_cavity_frame_roles(self):
        directory = ROOT / 'figure_data/cavity_comparison'
        entries = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(len(entries), 15)
        for time in (.01, 2.5, 5., 7.5, 10.):
            self.assertEqual({e['role'] for e in entries if e['time'] == time},
                             {'fine_reference', 'matched_reference', 'coarse_observation'})
        for entry in entries:
            self.assertEqual(hashlib.sha256((directory / entry['path']).read_bytes()).hexdigest(), entry['sha256'])

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('matplotlib'), 'plotting dependencies unavailable')
    def test_exact_cavity_slice_time(self):
        import numpy as np
        from make_figure12_slices import CAVITY_TIMES, linear_time_slices
        with np.load(ROOT / 'figure_data/selected_volumes/cavity.npz', allow_pickle=False) as archive:
            _, records = linear_time_slices(archive['speed'], archive['t'], CAVITY_TIMES, display_time_scale=.1)
        self.assertEqual(records[2]['physical_time'], 5.005)
        self.assertEqual(records[2]['array_sha256'], '9888bece9cdf61388f97524c253b195e4797f736ec02a51b1307cfaa1e760d44')


if __name__ == '__main__':
    unittest.main()
